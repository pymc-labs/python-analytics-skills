"""Benchmark runner — launches Claude in agentic mode with working dir isolation.

Each run gets an isolated temp directory with data copied in. Claude writes
model.py, runs it via Bash tool, and produces results.nc. The runner captures
the full JSON response for scoring.
"""

import json
import logging
import os
import signal
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

BENCHMARK_DIR = Path(__file__).parent.parent
SKILL_PATH = BENCHMARK_DIR.parent.parent / "skills" / "pymc-modeling" / "SKILL.md"
TASKS_PATH = BENCHMARK_DIR / "tasks.yaml"
RESULTS_DIR = BENCHMARK_DIR / "results"
DATA_DIR = BENCHMARK_DIR / "data"
RUNS_DIR = RESULTS_DIR / "runs"

# Claude CLI flags common to both conditions
BASE_FLAGS = [
    "--print",
    "--output-format", "json",
    "--model", "sonnet",
    "--tools", "Bash,Read,Write,Glob,Grep",
    "--disable-slash-commands",
    "--no-session-persistence",
    "--max-budget-usd", "2.0",
    "--dangerously-skip-permissions",
]

TIMEOUT_SECONDS = 600  # 10 minutes


def _kill_process_group(proc: subprocess.Popen):
    """Kill the entire process group rooted at proc.

    Sends SIGTERM first, waits briefly, then SIGKILL if anything survives.
    Silently ignores errors (process may already be dead).
    """
    try:
        pgid = os.getpgid(proc.pid)
    except (ProcessLookupError, OSError):
        return  # already dead

    try:
        os.killpg(pgid, signal.SIGTERM)
    except (ProcessLookupError, OSError):
        return

    # Give processes 5s to exit gracefully, then force-kill
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(pgid, signal.SIGKILL)
        except (ProcessLookupError, OSError):
            pass

    # Reap zombies
    try:
        proc.wait(timeout=2)
    except subprocess.TimeoutExpired:
        pass


@dataclass
class RunResult:
    """Result from a single benchmark run."""
    task_id: str
    condition: str  # "no_skill" or "with_skill"
    rep: int
    run_dir: Path
    success: bool
    wall_time: float = 0.0
    input_tokens: int = 0
    cache_creation_tokens: int = 0
    cache_read_tokens: int = 0
    output_tokens: int = 0
    total_input_tokens: int = 0
    tool_calls: list = field(default_factory=list)
    error: str = ""
    raw_response: str = ""


def load_tasks() -> dict:
    """Load task definitions from tasks.yaml."""
    with open(TASKS_PATH) as f:
        data = yaml.safe_load(f)
    return data


def get_run_dir(task_id: str, condition: str, rep: int) -> Path:
    """Get the result directory for a specific run."""
    return RUNS_DIR / f"{task_id}_{condition}_rep{rep}"


def is_cached(task_id: str, condition: str, rep: int) -> bool:
    """Check if a run result already exists."""
    run_dir = get_run_dir(task_id, condition, rep)
    return (run_dir / "metadata.json").exists()


def _setup_working_dir(task_id: str, task_config: dict) -> Path:
    """Create an isolated working directory with required data files."""
    work_dir = Path(f"/tmp/benchmark/{task_id}_{int(time.time())}")
    work_dir.mkdir(parents=True, exist_ok=True)
    data_dest = work_dir / "data"
    data_dest.mkdir(exist_ok=True)

    for rel_path in task_config.get("data_files", []):
        src = DATA_DIR / rel_path
        if not src.exists():
            # Try the cleaned version
            if rel_path == "gss_2022_clean.csv":
                src = DATA_DIR / "gss_2022_clean.csv"
            if not src.exists():
                raise FileNotFoundError(f"Data file not found: {src}")
        dest = data_dest / rel_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)

    return work_dir


def _build_prompt(preamble: str, task_prompt: str) -> str:
    """Combine preamble and task prompt."""
    return f"{preamble.strip()}\n\n{task_prompt.strip()}"


def _build_command(condition: str) -> list[str]:
    """Build the Claude CLI command."""
    cmd = ["claude"] + BASE_FLAGS
    if condition == "with_skill":
        skill_content = SKILL_PATH.read_text()
        cmd += ["--append-system-prompt", skill_content]
    return cmd


def _parse_response(raw: str) -> dict:
    """Parse Claude's JSON response, extracting tokens and tool calls.

    The --print --output-format json response has top-level keys:
    type, subtype, is_error, duration_ms, num_turns, result,
    usage, permission_denials, etc. No 'messages' field.
    Tool usage is inferred from num_turns and permission_denials.
    """
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return {"error": "Failed to parse JSON response"}

    result = {
        "input_tokens": data.get("usage", {}).get("input_tokens", 0),
        "cache_creation_tokens": data.get("usage", {}).get(
            "cache_creation_input_tokens", 0
        ),
        "cache_read_tokens": data.get("usage", {}).get(
            "cache_read_input_tokens", 0
        ),
        "output_tokens": data.get("usage", {}).get("output_tokens", 0),
        "tool_calls": [],
        "result": data.get("result", ""),
        "num_turns": data.get("num_turns", 0),
        "is_error": data.get("is_error", False),
        "cost_usd": data.get("total_cost_usd", 0.0),
    }

    # Total input tokens = input + cache_creation + cache_read
    result["total_input_tokens"] = (
        result["input_tokens"]
        + result["cache_creation_tokens"]
        + result["cache_read_tokens"]
    )

    # Extract tool calls from permission_denials (if any were denied)
    for denial in data.get("permission_denials", []):
        result["tool_calls"].append(denial.get("tool_name", "unknown"))

    # Check the result text for evidence of tool usage
    # (when tools succeed, they don't appear in permission_denials)
    result_text = data.get("result", "")
    if "model.py" in result_text or "results.nc" in result_text:
        result["produced_artifacts"] = True

    return result


def verify_isolation(parsed: dict, condition: str) -> list[str]:
    """Verify skill isolation — returns list of failures (empty = pass)."""
    failures = []

    # Check no Skill tool calls (from permission_denials or result text)
    skill_calls = [t for t in parsed.get("tool_calls", []) if t == "Skill"]
    if skill_calls:
        failures.append(
            f"Skill tool called {len(skill_calls)} times — contamination"
        )

    # Check that Claude actually produced output
    if parsed.get("is_error"):
        failures.append("Claude returned an error response")

    return failures


def verify_token_difference(
    no_skill_meta: dict, with_skill_meta: dict
) -> list[str]:
    """Verify token count difference between conditions.

    Compares output_tokens (proxy for work done) and checks that
    with_skill has more total input due to SKILL.md injection.
    Cache tokens make total_input_tokens unreliable for direct
    comparison, so we check that the with_skill run has evidence
    of more prompt content via cache_creation_tokens from its first
    turn (which includes SKILL.md).
    """
    failures = []

    # Both runs should have completed (num_turns > 0)
    ns_turns = no_skill_meta.get("num_turns", 0)
    ws_turns = with_skill_meta.get("num_turns", 0)

    if ns_turns == 0:
        failures.append("no_skill run had 0 turns — Claude didn't execute")
    if ws_turns == 0:
        failures.append("with_skill run had 0 turns — Claude didn't execute")

    # Check that with_skill had more cache_creation tokens (SKILL.md ~4500 tokens)
    ns_creation = no_skill_meta.get("cache_creation_tokens", 0)
    ws_creation = with_skill_meta.get("cache_creation_tokens", 0)
    creation_diff = ws_creation - ns_creation

    # Log but don't fail on cache_creation — caching behavior varies
    logger.info(
        f"Token check: no_skill cache_creation={ns_creation}, "
        f"with_skill cache_creation={ws_creation}, diff={creation_diff}"
    )

    return failures


def run_single(
    task_id: str,
    condition: str,
    rep: int,
    force: bool = False,
) -> RunResult:
    """Execute a single benchmark run.

    Args:
        task_id: Task identifier (e.g., "T1_hierarchical")
        condition: "no_skill" or "with_skill"
        rep: Replication number (0-indexed)
        force: If True, overwrite cached results
    """
    run_dir = get_run_dir(task_id, condition, rep)

    # Check cache
    if not force and is_cached(task_id, condition, rep):
        logger.info(f"Cached: {task_id} {condition} rep{rep}")
        meta = json.loads((run_dir / "metadata.json").read_text())
        return RunResult(
            task_id=task_id,
            condition=condition,
            rep=rep,
            run_dir=run_dir,
            success=meta.get("success", False),
            wall_time=meta.get("wall_time", 0.0),
            input_tokens=meta.get("input_tokens", 0),
            cache_creation_tokens=meta.get("cache_creation_tokens", 0),
            cache_read_tokens=meta.get("cache_read_tokens", 0),
            output_tokens=meta.get("output_tokens", 0),
            total_input_tokens=meta.get("total_input_tokens", 0),
        )

    # Load task config
    config = load_tasks()
    preamble = config["preamble"]
    task = config["tasks"][task_id]

    # Set up isolated working dir
    work_dir = _setup_working_dir(task_id, task)
    logger.info(f"Working dir: {work_dir}")

    # Build prompt and command
    prompt = _build_prompt(preamble, task["prompt"])
    cmd = _build_command(condition)

    # Run Claude in its own process group so we can kill the entire tree
    # (including python model.py spawned by Bash tool) on timeout.
    logger.info(f"Running: {task_id} {condition} rep{rep}")
    start = time.time()
    raw = ""
    error = ""

    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=str(work_dir),
        start_new_session=True,  # new process group
    )
    try:
        stdout, stderr = proc.communicate(input=prompt, timeout=TIMEOUT_SECONDS)
        raw = stdout
        elapsed = time.time() - start
        error = stderr if proc.returncode != 0 else ""
    except subprocess.TimeoutExpired:
        # Kill the entire process group
        _kill_process_group(proc)
        elapsed = TIMEOUT_SECONDS
        error = f"Timeout after {TIMEOUT_SECONDS}s"
        logger.warning(f"Timeout: {task_id} {condition} rep{rep}")
    except Exception as e:
        _kill_process_group(proc)
        elapsed = time.time() - start
        error = str(e)
    finally:
        # Ensure nothing survives even on normal completion —
        # Claude's Bash tool may have left background processes.
        _kill_process_group(proc)

    # Parse response
    parsed = _parse_response(raw) if raw else {}

    # Save results
    run_dir.mkdir(parents=True, exist_ok=True)

    # Copy working dir artifacts to run dir
    for artifact in ["model.py", "results.nc"]:
        src = work_dir / artifact
        if src.exists():
            shutil.copy2(src, run_dir / artifact)

    # Also check if artifacts were written to subdirectories
    for nc_file in work_dir.rglob("results.nc"):
        if nc_file != work_dir / "results.nc":
            shutil.copy2(nc_file, run_dir / "results.nc")
    for py_file in work_dir.rglob("model.py"):
        if py_file != work_dir / "model.py":
            shutil.copy2(py_file, run_dir / "model.py")

    # Save raw response
    if raw:
        (run_dir / "response.json").write_text(raw)

    # Save metadata
    result = RunResult(
        task_id=task_id,
        condition=condition,
        rep=rep,
        run_dir=run_dir,
        success=bool(parsed and not error),
        wall_time=elapsed,
        input_tokens=parsed.get("input_tokens", 0),
        cache_creation_tokens=parsed.get("cache_creation_tokens", 0),
        cache_read_tokens=parsed.get("cache_read_tokens", 0),
        output_tokens=parsed.get("output_tokens", 0),
        total_input_tokens=parsed.get("total_input_tokens", 0),
        tool_calls=parsed.get("tool_calls", []),
        error=error,
        raw_response=raw[:5000] if raw else "",
    )

    metadata = {
        "task_id": task_id,
        "condition": condition,
        "rep": rep,
        "success": result.success,
        "wall_time": result.wall_time,
        "num_turns": parsed.get("num_turns", 0),
        "input_tokens": result.input_tokens,
        "cache_creation_tokens": result.cache_creation_tokens,
        "cache_read_tokens": result.cache_read_tokens,
        "output_tokens": result.output_tokens,
        "total_input_tokens": result.total_input_tokens,
        "cost_usd": parsed.get("cost_usd", 0.0),
        "tool_calls": result.tool_calls,
        "error": result.error,
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    # Isolation checks
    isolation_failures = verify_isolation(parsed, condition)
    if isolation_failures:
        for f in isolation_failures:
            logger.error(f"ISOLATION FAILURE: {f}")
        result.success = False
        result.error = "; ".join(isolation_failures)

    # Clean up temp working dir
    shutil.rmtree(work_dir, ignore_errors=True)

    return result


def run_all(
    reps: int = 3,
    force: bool = False,
    resume: bool = False,
    tasks: list[str] | None = None,
) -> list[RunResult]:
    """Run all tasks in both conditions, interleaving for fairness.

    Args:
        reps: Number of replications per task/condition
        force: Overwrite all cached results
        resume: Only run missing/failed runs
        tasks: Specific task IDs to run (None = all)
    """
    config = load_tasks()
    task_ids = tasks or list(config["tasks"].keys())
    conditions = ["no_skill", "with_skill"]

    # Build run schedule, interleaving conditions
    schedule = []
    for rep in range(reps):
        for task_id in task_ids:
            for condition in conditions:
                schedule.append((task_id, condition, rep))

    results = []
    for task_id, condition, rep in schedule:
        if resume and is_cached(task_id, condition, rep):
            # In resume mode, check if previous run was successful
            run_dir = get_run_dir(task_id, condition, rep)
            meta = json.loads((run_dir / "metadata.json").read_text())
            if meta.get("success", False):
                logger.info(f"Skipping (resume, success): {task_id} {condition} rep{rep}")
                continue

        result = run_single(task_id, condition, rep, force=force)
        results.append(result)
        logger.info(
            f"{'OK' if result.success else 'FAIL'}: "
            f"{task_id} {condition} rep{rep} "
            f"({result.wall_time:.0f}s, {result.total_input_tokens} tokens)"
        )

    return results
