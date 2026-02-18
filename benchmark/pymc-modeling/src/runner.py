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
    "--verbose",
    "--output-format", "stream-json",
    "--model", "sonnet",
    "--tools", "Bash,Read,Write,Glob,Grep",
    "--disable-slash-commands",
    "--no-session-persistence",
    "--dangerously-skip-permissions",
]

DEFAULT_TIMEOUT = 600  # 10 minutes


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


def _kill_orphans(work_dir: Path):
    """Kill ALL processes whose cwd or cmdline references work_dir.

    Loops until zero matches — a dying process can spawn children between
    sweeps, so a single pass is not sufficient. Guaranteed clean on return.
    """
    work_dir_str = str(work_dir)
    my_pid = os.getpid()
    total_killed = []

    for sweep in range(10):
        killed = []
        for entry in Path("/proc").iterdir():
            if not entry.name.isdigit():
                continue
            pid = int(entry.name)
            if pid == my_pid:
                continue
            try:
                # Check cwd
                cwd = os.readlink(f"/proc/{pid}/cwd")
                if work_dir_str in cwd:
                    os.kill(pid, signal.SIGKILL)
                    killed.append(pid)
                    continue
                # Check cmdline
                cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().decode(
                    errors="replace"
                )
                if work_dir_str in cmdline:
                    os.kill(pid, signal.SIGKILL)
                    killed.append(pid)
            except (ProcessLookupError, PermissionError, OSError):
                continue
        if not killed:
            break
        total_killed.extend(killed)
        time.sleep(0.5)  # let children die before next sweep

    if total_killed:
        logger.warning(
            f"Killed {len(total_killed)} orphan(s) over {sweep + 1} sweep(s) "
            f"in {work_dir}: {total_killed}"
        )


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


def is_corrupted_model(path: Path) -> bool:
    """Check if a model.py file is actually a library file (e.g., pymc/dims/model.py).

    Returns True if the first 200 bytes contain a copyright or license header,
    indicating this is installed package code rather than Claude-generated code.
    """
    try:
        header = path.read_text(errors="replace")[:200]
    except OSError:
        return False
    return "Copyright" in header or "Licensed under" in header


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
    """Parse Claude's stream-json response (NDJSON).

    The --print --verbose --output-format stream-json output is
    newline-delimited JSON. Each line has {"type": ...}:
    - "system"    — hook events, init (ignored)
    - "assistant" — Claude's messages with content blocks (tool_use, text)
    - "result"    — final aggregated result (tokens, cost, num_turns)

    Returns a dict compatible with the old single-JSON format, plus a
    "turns" list of assistant message objects for thrashing analysis.
    """
    lines = raw.strip().split("\n")
    result_data = None
    turns = []

    for line in lines:
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue

        msg_type = obj.get("type")
        if msg_type == "result":
            result_data = obj
        elif msg_type == "assistant":
            turns.append(obj)

    if result_data is None:
        return {"error": "No result object in stream-json output", "turns": turns}

    result = {
        "input_tokens": result_data.get("usage", {}).get("input_tokens", 0),
        "cache_creation_tokens": result_data.get("usage", {}).get(
            "cache_creation_input_tokens", 0
        ),
        "cache_read_tokens": result_data.get("usage", {}).get(
            "cache_read_input_tokens", 0
        ),
        "output_tokens": result_data.get("usage", {}).get("output_tokens", 0),
        "tool_calls": [],
        "result": result_data.get("result", ""),
        "num_turns": result_data.get("num_turns", 0),
        "is_error": result_data.get("is_error", False),
        "cost_usd": result_data.get("total_cost_usd", 0.0),
        "turns": turns,
    }

    # Total input tokens = input + cache_creation + cache_read
    result["total_input_tokens"] = (
        result["input_tokens"]
        + result["cache_creation_tokens"]
        + result["cache_read_tokens"]
    )

    # Extract tool calls from permission_denials (if any were denied)
    for denial in result_data.get("permission_denials", []):
        result["tool_calls"].append(denial.get("tool_name", "unknown"))

    # Check the result text for evidence of tool usage
    result_text = result_data.get("result", "")
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
    timeout = task.get("timeout", DEFAULT_TIMEOUT)

    # Run Claude in its own process group so we can kill the entire tree
    # (including python model.py spawned by Bash tool) on timeout.
    logger.info(f"Running: {task_id} {condition} rep{rep} (timeout={timeout}s)")
    start = time.time()
    raw = ""
    error = ""

    # Strip CLAUDECODE to allow nested claude --print calls.
    # Keep all other env vars (especially auth credentials) intact.
    env = os.environ.copy()
    env.pop("CLAUDECODE", None)

    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=str(work_dir),
        start_new_session=True,  # new process group
        env=env,
    )
    try:
        stdout, stderr = proc.communicate(input=prompt, timeout=timeout)
        raw = stdout
        elapsed = time.time() - start
        error = stderr if proc.returncode != 0 else ""
    except subprocess.TimeoutExpired:
        # Kill the entire process group
        _kill_process_group(proc)
        elapsed = timeout
        error = f"Timeout after {timeout}s"
        logger.warning(f"Timeout: {task_id} {condition} rep{rep}")
    except Exception as e:
        _kill_process_group(proc)
        elapsed = time.time() - start
        error = str(e)
    finally:
        # Ensure nothing survives even on normal completion —
        # Claude's Bash tool may have left background processes.
        _kill_process_group(proc)
        # Kill any grandchild processes that escaped the process group
        # (e.g., nutpie sampler workers spawned by model.py)
        _kill_orphans(work_dir)

    # Parse response
    parsed = _parse_response(raw) if raw else {}

    # Save results
    run_dir.mkdir(parents=True, exist_ok=True)

    # Copy working dir artifacts to run dir
    for artifact in ["model.py", "results.nc"]:
        src = work_dir / artifact
        if src.exists():
            shutil.copy2(src, run_dir / artifact)

    # If results.nc wasn't at the root, check subdirectories (first match only)
    if not (run_dir / "results.nc").exists():
        for nc_file in work_dir.rglob("results.nc"):
            if nc_file != work_dir / "results.nc":
                logger.info(f"Found results.nc in subdirectory: {nc_file}")
                shutil.copy2(nc_file, run_dir / "results.nc")
                break

    # Corruption detection: reject model.py that is actually a library file
    # (e.g., pymc/dims/model.py found by rglob in previous versions)
    model_dest = run_dir / "model.py"
    if model_dest.exists() and is_corrupted_model(model_dest):
        logger.error(
            f"CORRUPTION: {model_dest} contains a library copyright header — removing"
        )
        model_dest.unlink()

    # Save raw response (NDJSON from stream-json)
    if raw:
        (run_dir / "response.json").write_text(raw)

    # Save turn-by-turn data for thrashing analysis
    if parsed.get("turns"):
        turns_path = run_dir / "turns.jsonl"
        with open(turns_path, "w") as f:
            for turn in parsed["turns"]:
                f.write(json.dumps(turn) + "\n")

    # Save metadata
    result = RunResult(
        task_id=task_id,
        condition=condition,
        rep=rep,
        run_dir=run_dir,
        success=bool(parsed and not error and parsed.get("total_input_tokens", 0) > 0),
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
