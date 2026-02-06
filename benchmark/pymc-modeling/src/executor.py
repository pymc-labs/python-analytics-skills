"""Phase 2: Execute generated PyMC scripts.

Runs extracted Python scripts in a pixi-managed environment,
capturing execution results and InferenceData (.nc) files.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import time
from pathlib import Path
from tempfile import TemporaryDirectory

BENCHMARK_DIR = Path(__file__).resolve().parent.parent
CODE_DIR = BENCHMARK_DIR / "results" / "code"
EXEC_DIR = BENCHMARK_DIR / "results" / "execution"
DATA_DIR = BENCHMARK_DIR / "data"

# Environment variables for deterministic execution
EXEC_ENV = {
    "MPLBACKEND": "Agg",
    "PYTENSOR_FLAGS": "device=cpu",
    "OMP_NUM_THREADS": "4",
}

# Timeout for script execution (seconds)
DEFAULT_TIMEOUT = 600  # 10 minutes


def classify_error(stderr: str, return_code: int) -> str:
    """Classify the type of execution error."""
    if return_code == -9 or return_code == 137:
        return "timeout"

    stderr_lower = stderr.lower()
    if "syntaxerror" in stderr_lower:
        return "syntax_error"
    if "modulenotfounderror" in stderr_lower or "importerror" in stderr_lower:
        return "import_error"
    if "samplingerror" in stderr_lower or "divergen" in stderr_lower:
        return "sampling_error"
    if "valueerror" in stderr_lower or "typeerror" in stderr_lower:
        return "runtime_error"
    if "memoryerror" in stderr_lower:
        return "memory_error"

    return "runtime_error"


def execute_script(
    code_path: Path,
    task_id: str,
    condition: str,
    replication: int,
    timeout: int = DEFAULT_TIMEOUT,
) -> dict:
    """Execute a single generated script.

    Creates a temporary working directory, copies data files into it,
    runs the script, and captures results including InferenceData.
    """
    EXEC_DIR.mkdir(parents=True, exist_ok=True)

    result = {
        "task_id": task_id,
        "condition": condition,
        "replication": replication,
        "code_file": str(code_path),
        "execution_success": False,
        "return_code": None,
        "wall_time_seconds": None,
        "stdout": None,
        "stderr": None,
        "idata_path": None,
        "idata_exists": False,
        "error_type": None,
    }

    if not code_path.exists():
        result["error_type"] = "missing_script"
        result["stderr"] = f"Script not found: {code_path}"
        return result

    with TemporaryDirectory(prefix="benchmark_") as tmpdir:
        tmpdir = Path(tmpdir)

        # Copy data directory into tmpdir
        tmp_data = tmpdir / "data"
        shutil.copytree(DATA_DIR, tmp_data)

        # Copy the script
        script_path = tmpdir / "script.py"
        shutil.copy2(code_path, script_path)

        # Build environment
        import os

        env = os.environ.copy()
        env.update(EXEC_ENV)

        # Run the script using pixi
        start = time.time()
        try:
            proc = subprocess.run(
                ["pixi", "run", "python", str(script_path)],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(tmpdir),
                env=env,
            )

            elapsed = time.time() - start
            result["wall_time_seconds"] = round(elapsed, 2)
            result["return_code"] = proc.returncode
            result["stdout"] = proc.stdout[-10000:] if proc.stdout else ""
            result["stderr"] = proc.stderr[-10000:] if proc.stderr else ""

            if proc.returncode == 0:
                result["execution_success"] = True
            else:
                result["error_type"] = classify_error(
                    proc.stderr or "", proc.returncode
                )

        except subprocess.TimeoutExpired:
            elapsed = time.time() - start
            result["wall_time_seconds"] = round(elapsed, 2)
            result["error_type"] = "timeout"
            result["stderr"] = f"Timeout after {timeout} seconds"

        # Check for InferenceData output
        nc_file = tmpdir / "results.nc"
        if not nc_file.exists():
            # Try to find any .nc file
            nc_files = list(tmpdir.glob("*.nc"))
            if nc_files:
                nc_file = nc_files[0]

        if nc_file.exists():
            result["idata_exists"] = True
            # Copy to results/execution/
            dest_name = f"{task_id}_{condition}_rep{replication}.nc"
            dest_path = EXEC_DIR / dest_name
            shutil.copy2(nc_file, dest_path)
            result["idata_path"] = str(dest_path)

        elif result["execution_success"]:
            # Script ran OK but no InferenceData saved
            result["error_type"] = "no_output"

    # Save execution metadata
    meta_name = f"{task_id}_{condition}_rep{replication}.json"
    meta_path = EXEC_DIR / meta_name
    with open(meta_path, "w") as f:
        json.dump(result, f, indent=2)

    return result


def execute_all(timeout: int = DEFAULT_TIMEOUT) -> list[dict]:
    """Execute all extracted scripts."""
    if not CODE_DIR.exists():
        print("No extracted code found. Run 'extract' first.")
        return []

    scripts = sorted(CODE_DIR.glob("*.py"))
    if not scripts:
        print("No scripts found in results/code/")
        return []

    results = []
    for i, script_path in enumerate(scripts, 1):
        # Parse task_id, condition, replication from filename
        # Format: T1_with_skill_rep1.py
        stem = script_path.stem
        parts = stem.split("_")

        task_id = parts[0]
        # Find rep number
        rep_idx = next(
            (j for j, p in enumerate(parts) if p.startswith("rep")), len(parts) - 1
        )
        condition = "_".join(parts[1:rep_idx])
        try:
            replication = int(parts[rep_idx].replace("rep", ""))
        except (ValueError, IndexError):
            replication = 1

        print(f"\n[{i}/{len(scripts)}] Executing {stem}")

        result = execute_script(
            script_path, task_id, condition, replication, timeout=timeout
        )

        status = (
            "OK" if result["execution_success"] else f"FAIL ({result['error_type']})"
        )
        nc_status = "idata saved" if result["idata_exists"] else "no idata"
        time_str = (
            f"{result['wall_time_seconds']:.1f}s"
            if result["wall_time_seconds"]
            else "N/A"
        )
        print(f"  {status} | {nc_status} | {time_str}")

        results.append(result)

    success = sum(1 for r in results if r["execution_success"])
    idata = sum(1 for r in results if r["idata_exists"])
    print(
        f"\nExecution: {success}/{len(results)} succeeded, {idata}/{len(results)} produced InferenceData"
    )
    return results
