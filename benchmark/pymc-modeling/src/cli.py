"""CLI entry point for the PyMC skill benchmark.

Subcommands:
  run        — Run benchmark tasks
  score      — Score completed runs
  analyze    — Generate analysis report
  validate   — Gate: run T1 in both conditions, verify everything works
  list-tasks — Show available tasks
  status     — Show run/score status
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import yaml

from src.runner import (
    RESULTS_DIR,
    RUNS_DIR,
    TASKS_PATH,
    get_run_dir,
    is_cached,
    load_tasks,
    run_all,
    run_single,
    verify_token_difference,
)
from src.scorer import score_all, score_run
from src.analysis import generate_report, load_scores

logger = logging.getLogger("benchmark")


def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.getLogger("arviz.preview").setLevel(logging.WARNING)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def cmd_run(args):
    """Run benchmark tasks."""
    config = load_tasks()
    task_ids = list(config["tasks"].keys()) if args.all else [args.task]

    if not args.all and args.task not in config["tasks"]:
        print(f"Unknown task: {args.task}")
        print(f"Available: {', '.join(config['tasks'].keys())}")
        return 1

    results = run_all(
        reps=args.reps,
        force=args.force,
        resume=args.resume,
        tasks=task_ids,
    )

    # Summary
    success = sum(1 for r in results if r.success)
    total = len(results)
    print(f"\nCompleted: {success}/{total} runs succeeded")

    for r in results:
        status = "OK" if r.success else "FAIL"
        print(f"  {status}: {r.task_id} {r.condition} rep{r.rep} ({r.wall_time:.0f}s)")
        if r.error:
            print(f"         Error: {r.error[:200]}")

    return 0 if success == total else 1


def cmd_score(args):
    """Score completed runs."""
    if args.all:
        results = score_all()
    else:
        run_dir = get_run_dir(args.task, args.condition, args.rep)
        if not run_dir.exists():
            print(f"No run found at {run_dir}")
            return 1
        result = score_run(run_dir, args.task, args.condition, args.rep)
        results = [result]

    for r in results:
        print(
            f"{r.task_id} {r.condition} rep{r.rep}: "
            f"produced={r.model_produced} conv={r.convergence} "
            f"approp={r.model_appropriateness} bp={r.best_practices} "
            f"total={r.total}/20"
        )

    return 0


def cmd_analyze(args):
    """Generate analysis report."""
    report = generate_report()
    print(report)
    return 0


def cmd_validate(args):
    """Validation gate: run T1 in both conditions, verify everything works."""
    print("=" * 60)
    print("VALIDATION RUN — T1 hierarchical, both conditions")
    print("=" * 60)

    log_lines = []

    def log(msg):
        print(msg)
        log_lines.append(msg)

    checks = []

    # Run T1 in both conditions
    for condition in ["no_skill", "with_skill"]:
        log(f"\n--- Running T1 {condition} ---")
        start = time.time()
        result = run_single("T1_hierarchical", condition, rep=0, force=args.force)
        elapsed = time.time() - start
        log(f"Wall time: {elapsed:.0f}s")
        log(f"Success: {result.success}")
        log(f"Tokens: input={result.input_tokens} cache_creation={result.cache_creation_tokens} "
            f"cache_read={result.cache_read_tokens} output={result.output_tokens} "
            f"total_input={result.total_input_tokens}")
        log(f"Tool calls: {result.tool_calls}")
        if result.error:
            log(f"Error: {result.error[:500]}")

    # Load results for checks
    no_skill_dir = get_run_dir("T1_hierarchical", "no_skill", 0)
    with_skill_dir = get_run_dir("T1_hierarchical", "with_skill", 0)

    # Check 1: Token difference / runs completed
    no_meta = json.loads((no_skill_dir / "metadata.json").read_text())
    ws_meta = json.loads((with_skill_dir / "metadata.json").read_text())
    token_failures = verify_token_difference(no_meta, ws_meta)
    if token_failures:
        for f in token_failures:
            log(f"FAIL: {f}")
        checks.append(("runs_completed", False, token_failures))
    else:
        ns_turns = no_meta.get("num_turns", 0)
        ws_turns = ws_meta.get("num_turns", 0)
        log(f"PASS: Both runs completed (no_skill={ns_turns} turns, with_skill={ws_turns} turns)")
        checks.append(("runs_completed", True, []))

    # Check 2: No Skill tool calls
    for cond, meta in [("no_skill", no_meta), ("with_skill", ws_meta)]:
        skill_calls = [t for t in meta.get("tool_calls", []) if t == "Skill"]
        if skill_calls:
            log(f"FAIL: Skill tool called in {cond} condition")
            checks.append((f"no_skill_tool_{cond}", False, ["Skill tool detected"]))
        else:
            log(f"PASS: No Skill tool calls in {cond}")
            checks.append((f"no_skill_tool_{cond}", True, []))

    # Check 3: Tools used (num_turns > 1 means tools were used)
    for cond, meta in [("no_skill", no_meta), ("with_skill", ws_meta)]:
        num_turns = meta.get("num_turns", 0)
        if num_turns <= 1:
            log(f"FAIL: Only {num_turns} turn(s) in {cond} — Claude didn't use tools")
            checks.append((f"tools_used_{cond}", False, [f"Only {num_turns} turns"]))
        else:
            log(f"PASS: {num_turns} turns in {cond} (tools were used)")
            checks.append((f"tools_used_{cond}", True, []))

    # Check 4: model.py produced
    for cond, run_dir in [("no_skill", no_skill_dir), ("with_skill", with_skill_dir)]:
        model_py = run_dir / "model.py"
        if model_py.exists() and model_py.stat().st_size > 100:
            log(f"PASS: model.py exists in {cond} ({model_py.stat().st_size} bytes)")
            checks.append((f"model_py_{cond}", True, []))
        else:
            log(f"FAIL: model.py missing or too small in {cond}")
            checks.append((f"model_py_{cond}", False, ["model.py missing/empty"]))

    # Check 5: results.nc produced
    for cond, run_dir in [("no_skill", no_skill_dir), ("with_skill", with_skill_dir)]:
        nc = run_dir / "results.nc"
        if nc.exists():
            try:
                import arviz as az
                idata = az.from_netcdf(str(nc))
                has_posterior = hasattr(idata, "posterior") and idata.posterior is not None
                if has_posterior:
                    log(f"PASS: results.nc has posterior in {cond}")
                    checks.append((f"results_nc_{cond}", True, []))
                else:
                    log(f"FAIL: results.nc has no posterior in {cond}")
                    checks.append((f"results_nc_{cond}", False, ["no posterior"]))
            except Exception as e:
                log(f"FAIL: results.nc load error in {cond}: {e}")
                checks.append((f"results_nc_{cond}", False, [str(e)]))
        else:
            log(f"FAIL: results.nc missing in {cond}")
            checks.append((f"results_nc_{cond}", False, ["missing"]))

    # Check 6: Scorer runs
    for cond, run_dir in [("no_skill", no_skill_dir), ("with_skill", with_skill_dir)]:
        try:
            score = score_run(run_dir, "T1_hierarchical", cond, 0)
            log(f"PASS: Scorer returned total={score.total}/20 for {cond}")
            checks.append((f"scorer_{cond}", True, []))
        except Exception as e:
            log(f"FAIL: Scorer error in {cond}: {e}")
            checks.append((f"scorer_{cond}", False, [str(e)]))

    # Check 7: Working dir isolation
    for cond, run_dir in [("no_skill", no_skill_dir), ("with_skill", with_skill_dir)]:
        contaminated = []
        for bad_dir in [".claude-plugin", "hooks", ".claude"]:
            if (run_dir / bad_dir).exists():
                contaminated.append(bad_dir)
        if contaminated:
            log(f"FAIL: Working dir contaminated in {cond}: {contaminated}")
            checks.append((f"isolation_{cond}", False, contaminated))
        else:
            log(f"PASS: Working dir clean in {cond}")
            checks.append((f"isolation_{cond}", True, []))

    # Summary
    log("\n" + "=" * 60)
    passed = sum(1 for _, ok, _ in checks if ok)
    total = len(checks)
    log(f"Validation: {passed}/{total} checks passed")

    for name, ok, failures in checks:
        status = "PASS" if ok else "FAIL"
        detail = f" — {'; '.join(failures)}" if failures else ""
        log(f"  [{status}] {name}{detail}")

    # Save validation log
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "validation.log").write_text("\n".join(log_lines))

    if passed < total:
        log(f"\nValidation FAILED ({total - passed} failures). Fix issues before running full suite.")
        return 1

    log("\nValidation PASSED. Safe to run full suite.")
    return 0


def cmd_list_tasks(args):
    """List available tasks."""
    config = load_tasks()
    for tid, task in config["tasks"].items():
        data = ", ".join(task.get("data_files", [])) or "embedded"
        print(f"  {tid}: {task['name']} (data: {data})")


def cmd_status(args):
    """Show status of all runs and scores."""
    config = load_tasks()
    task_ids = list(config["tasks"].keys())
    conditions = ["no_skill", "with_skill"]

    print(f"{'Task':<25} {'Condition':<12} {'Rep':>4} {'Run':>6} {'Score':>6}")
    print("-" * 60)

    for task_id in task_ids:
        for condition in conditions:
            for rep in range(3):
                run_dir = get_run_dir(task_id, condition, rep)
                has_run = (run_dir / "metadata.json").exists()
                has_score = (RESULTS_DIR / "scores" / f"{task_id}_{condition}_rep{rep}.json").exists()

                run_status = "done" if has_run else "-"
                score_status = "done" if has_score else "-"

                if has_run:
                    meta = json.loads((run_dir / "metadata.json").read_text())
                    if not meta.get("success"):
                        run_status = "FAIL"

                print(f"  {task_id:<23} {condition:<12} {rep:>4} {run_status:>6} {score_status:>6}")


def main():
    parser = argparse.ArgumentParser(
        description="PyMC Skill Benchmark",
        prog="python -m src.cli",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    # run
    p_run = sub.add_parser("run", help="Run benchmark tasks")
    p_run.add_argument("--all", action="store_true", help="Run all tasks")
    p_run.add_argument("--task", default="T1_hierarchical", help="Task ID to run")
    p_run.add_argument("--reps", type=int, default=3, help="Replications per condition")
    p_run.add_argument("--force", action="store_true", help="Overwrite cached results")
    p_run.add_argument("--resume", action="store_true", help="Re-run only missing/failed")

    # score
    p_score = sub.add_parser("score", help="Score completed runs")
    p_score.add_argument("--all", action="store_true", help="Score all runs")
    p_score.add_argument("--task", default="T1_hierarchical")
    p_score.add_argument("--condition", default="no_skill")
    p_score.add_argument("--rep", type=int, default=0)

    # analyze
    sub.add_parser("analyze", help="Generate analysis report")

    # validate
    p_val = sub.add_parser("validate", help="Validation gate (T1, both conditions)")
    p_val.add_argument("--force", action="store_true", help="Force re-run")

    # list-tasks
    sub.add_parser("list-tasks", help="List available tasks")

    # status
    sub.add_parser("status", help="Show run/score status")

    args = parser.parse_args()
    setup_logging(args.verbose)

    commands = {
        "run": cmd_run,
        "score": cmd_score,
        "analyze": cmd_analyze,
        "validate": cmd_validate,
        "list-tasks": cmd_list_tasks,
        "status": cmd_status,
    }

    sys.exit(commands[args.command](args))


if __name__ == "__main__":
    main()
