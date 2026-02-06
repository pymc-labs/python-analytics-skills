"""Phase 1: Generate PyMC code via Claude CLI.

Uses claude --print mode with --append-system-prompt to inject the skill
content directly into the system prompt for the with_skill condition.

The no_skill condition omits the --append-system-prompt flag entirely,
giving Claude only its base knowledge.
"""

from __future__ import annotations

import json
import subprocess
import time
from datetime import datetime
from pathlib import Path

import yaml

BENCHMARK_DIR = Path(__file__).resolve().parent.parent
TASKS_FILE = BENCHMARK_DIR / "tasks.yaml"
RAW_DIR = BENCHMARK_DIR / "results" / "raw"

SKILL_NAME = "pymc-modeling"
SKILL_DIR = Path.home() / ".claude" / "skills" / SKILL_NAME
SKILL_FILE = SKILL_DIR / "SKILL.md"

# Minimum token count that proves skill content was loaded.
# SKILL.md is ~18KB which is ~4500 tokens. The base system prompt for
# --print mode is ~25K-50K tokens. With the skill appended, we expect
# a clearly measurable increase. This threshold is conservative:
# if total input tokens (including cache) aren't at least this much
# higher than the no_skill baseline, something is wrong.
SKILL_MIN_CHARS = 15000  # SKILL.md is ~18K chars; sanity floor


def has_raw_result(task_id: str, condition: str, replication: int) -> bool:
    """Check if a raw result already exists for this scenario."""
    if not RAW_DIR.exists():
        return False
    return any(RAW_DIR.glob(f"{task_id}_{condition}_rep{replication}_*.json"))


PROMPT_PREAMBLE = """\
Write a complete, self-contained Python script that can be run with `python script.py`.
Do NOT use marimo notebooks or Jupyter notebooks. Include all imports.
At the end, save the InferenceData object to 'results.nc' using idata.to_netcdf('results.nc').
Print a summary of convergence diagnostics to stdout.
Do NOT ask for confirmation — just provide the complete script.

"""


def load_tasks() -> dict:
    """Load task definitions from YAML."""
    with open(TASKS_FILE) as f:
        return yaml.safe_load(f)["tasks"]


def load_skill_content() -> str:
    """Read the SKILL.md file and return its content.

    Raises FileNotFoundError if the skill is not installed.
    """
    if not SKILL_FILE.exists():
        raise FileNotFoundError(
            f"Skill file not found at {SKILL_FILE}. "
            "Install the pymc-modeling skill first."
        )
    content = SKILL_FILE.read_text()
    if len(content) < SKILL_MIN_CHARS:
        raise ValueError(
            f"Skill file suspiciously small ({len(content)} chars, "
            f"expected >= {SKILL_MIN_CHARS}). File may be corrupted: {SKILL_FILE}"
        )
    return content


def check_skill_status() -> dict:
    """Check current status of the pymc-modeling skill."""
    skill_content_len = 0
    if SKILL_FILE.exists():
        skill_content_len = len(SKILL_FILE.read_text())
    return {
        "skill_installed": SKILL_FILE.exists(),
        "skill_path": str(SKILL_FILE),
        "skill_content_chars": skill_content_len,
    }


def _parse_usage(output: dict) -> dict:
    """Extract comprehensive token usage from Claude JSON output."""
    usage = output.get("usage", {})
    return {
        "input_tokens": usage.get("input_tokens", 0),
        "cache_creation_input_tokens": usage.get("cache_creation_input_tokens", 0),
        "cache_read_input_tokens": usage.get("cache_read_input_tokens", 0),
        "output_tokens": usage.get("output_tokens", 0),
        "total_input_tokens": (
            usage.get("input_tokens", 0)
            + usage.get("cache_creation_input_tokens", 0)
            + usage.get("cache_read_input_tokens", 0)
        ),
    }


def get_result_path(task_id: str, condition: str, replication: int) -> Path:
    """Generate path for a result file."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return RAW_DIR / f"{task_id}_{condition}_rep{replication}_{timestamp}.json"


def run_claude(
    prompt: str,
    condition: str,
    model: str = "sonnet",
    max_budget: float = 1.0,
    timeout_minutes: int = 10,
) -> dict:
    """Run Claude CLI in print mode and capture results.

    For 'with_skill': injects SKILL.md via --append-system-prompt.
    For 'no_skill': no skill content is appended.
    """
    cmd = [
        "claude",
        "--print",
        "--output-format",
        "json",
        "--model",
        model,
        "--max-budget-usd",
        str(max_budget),
        "--no-session-persistence",
    ]

    skill_injected = False
    skill_chars = 0

    if condition == "with_skill":
        skill_content = load_skill_content()
        cmd.extend(["--append-system-prompt", skill_content])
        skill_injected = True
        skill_chars = len(skill_content)

    result = {
        "start_time": datetime.now().isoformat(),
        "prompt": prompt,
        "condition": condition,
        "skill_injected": skill_injected,
        "skill_chars_injected": skill_chars,
        "success": False,
        "error": None,
        "response": None,
        "response_text": None,
        "execution_time_seconds": None,
        "input_tokens": None,
        "output_tokens": None,
        "total_input_tokens": None,
        "cost_usd": None,
        "model": model,
    }

    start = time.time()

    try:
        proc = subprocess.run(
            cmd,
            input=prompt,
            capture_output=True,
            text=True,
            timeout=timeout_minutes * 60,
        )

        elapsed = time.time() - start
        result["execution_time_seconds"] = round(elapsed, 2)
        result["end_time"] = datetime.now().isoformat()
        result["response"] = proc.stdout

        try:
            output = json.loads(proc.stdout)
            usage = _parse_usage(output)
            result["input_tokens"] = usage["input_tokens"]
            result["output_tokens"] = usage["output_tokens"]
            result["total_input_tokens"] = usage["total_input_tokens"]
            result["cache_creation_input_tokens"] = usage["cache_creation_input_tokens"]
            result["cache_read_input_tokens"] = usage["cache_read_input_tokens"]
            result["cost_usd"] = output.get("total_cost_usd")
            result["response_text"] = output.get("result", "")
            result["num_turns"] = output.get("num_turns")
            result["session_id"] = output.get("session_id")

            if output.get("is_error"):
                result["error"] = output.get("result", f"Exit code: {proc.returncode}")
            elif proc.returncode == 0:
                result["success"] = True
            else:
                result["error"] = proc.stderr or f"Exit code: {proc.returncode}"
        except json.JSONDecodeError:
            if proc.returncode == 0:
                result["success"] = True
                result["response_text"] = proc.stdout
            else:
                result["error"] = proc.stderr or f"Exit code: {proc.returncode}"

    except subprocess.TimeoutExpired:
        result["error"] = f"Timeout after {timeout_minutes} minutes"
        result["execution_time_seconds"] = timeout_minutes * 60
    except FileNotFoundError:
        result["error"] = "Claude CLI not found. Is 'claude' in PATH?"

    return result


def run_task(
    task_id: str,
    condition: str,
    replication: int = 1,
    model: str = "sonnet",
    max_budget: float = 1.0,
    dry_run: bool = False,
    force: bool = False,
) -> dict | None:
    """Run a single benchmark task.

    Returns None if the scenario was skipped (already has results).
    """
    tasks = load_tasks()

    if task_id not in tasks:
        raise ValueError(f"Unknown task: {task_id}. Valid: {sorted(tasks.keys())}")

    if condition not in ("with_skill", "no_skill"):
        raise ValueError(f"Unknown condition: {condition!r}")

    if not force and not dry_run and has_raw_result(task_id, condition, replication):
        print(f"  SKIP {task_id} ({condition}, rep {replication}) — result exists")
        return None

    task = tasks[task_id]
    prompt = PROMPT_PREAMBLE + task["prompt"].strip()

    # Pre-flight: verify skill is readable before starting a with_skill run
    if condition == "with_skill" and not dry_run:
        load_skill_content()  # raises if missing or corrupt

    print(f"\n{'=' * 60}")
    print(f"Task: {task_id} - {task['name']}")
    print(f"Condition: {condition}")
    print(f"Replication: {replication}")
    print(f"Tier: {task['tier']}")
    if condition == "with_skill":
        print(f"Skill: {SKILL_FILE} ({SKILL_FILE.stat().st_size} bytes)")
    print(f"{'=' * 60}")

    if dry_run:
        print("\n[DRY RUN] Prompt:")
        print("-" * 40)
        print(prompt)
        if condition == "with_skill":
            skill = load_skill_content()
            print("-" * 40)
            print(f"[DRY RUN] Skill content ({len(skill)} chars) would be appended")
        print("-" * 40)
        return {"dry_run": True, "task_id": task_id, "condition": condition}

    result = run_claude(prompt, condition=condition, model=model, max_budget=max_budget)

    # Add metadata
    result["task_id"] = task_id
    result["task_name"] = task["name"]
    result["tier"] = task["tier"]
    result["replication"] = replication
    result["condition"] = condition
    result["skill_coverage"] = task.get("skill_coverage", [])
    result["dataset"] = task.get("dataset")

    # Save result
    result_path = get_result_path(task_id, condition, replication)
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nResult saved to: {result_path}")
    print(f"Skill injected: {result['skill_injected']}")
    print(f"Skill chars: {result['skill_chars_injected']}")
    print(f"Total input tokens: {result.get('total_input_tokens', 'N/A')}")
    print(f"Execution time: {result.get('execution_time_seconds', 'N/A')}s")
    print(f"Success: {result.get('success', False)}")
    if result.get("error"):
        print(f"Error: {result['error']}")

    return result


def run_all(
    condition: str | None = None,
    replications: int = 3,
    model: str = "sonnet",
    max_budget: float = 1.0,
    dry_run: bool = False,
    force: bool = False,
) -> list[dict]:
    """Run all benchmark tasks, skipping scenarios that already have results."""
    tasks = load_tasks()
    conditions = [condition] if condition else ["no_skill", "with_skill"]
    results = []

    total = len(tasks) * len(conditions) * replications
    current = 0

    for cond in conditions:
        for task_id in sorted(tasks.keys()):
            for rep in range(1, replications + 1):
                current += 1
                print(f"\n[{current}/{total}] {task_id} ({cond}, rep {rep})")

                result = run_task(
                    task_id,
                    cond,
                    rep,
                    model=model,
                    max_budget=max_budget,
                    dry_run=dry_run,
                    force=force,
                )
                if result is not None:
                    results.append(result)

                if result is not None and not dry_run:
                    time.sleep(2)

    return results
