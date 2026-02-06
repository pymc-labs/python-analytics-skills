"""Unified CLI for the PyMC skill benchmark suite.

Usage:
    pixi run python -m src.cli <command> [options]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BENCHMARK_DIR = Path(__file__).resolve().parent.parent


def cmd_run(args):
    """Run Claude to generate code (Phase 1)."""
    from .runner import load_tasks, run_all, run_task

    if args.all:
        run_all(
            condition=args.condition,
            replications=args.reps,
            model=args.model,
            max_budget=args.max_budget,
            dry_run=args.dry_run,
            force=args.force,
        )
    elif args.task:
        if not args.condition:
            print("Error: --condition required when running a single task")
            sys.exit(1)
        run_task(
            args.task,
            args.condition,
            replication=args.rep,
            model=args.model,
            max_budget=args.max_budget,
            dry_run=args.dry_run,
            force=args.force,
        )
    else:
        print("Specify --task or --all")
        sys.exit(1)


def cmd_extract(args):
    """Extract code from Claude output (Phase 1.5)."""
    from .extractor import extract_all

    print("Extracting code from raw results...")
    extract_all()


def cmd_execute(args):
    """Execute generated scripts (Phase 2)."""
    from .executor import execute_all

    print("Executing extracted scripts...")
    execute_all(timeout=args.timeout)


def cmd_diagnose(args):
    """Compute MCMC diagnostics (Phase 3)."""
    from .diagnostics import diagnose_all

    print("Computing diagnostics from InferenceData files...")
    diagnose_all()


def cmd_score(args):
    """Score all results (Phase 4)."""
    from .scorer import score_all

    print("Scoring results...")
    score_all(use_llm_judge=not args.no_llm)


def cmd_pipeline(args):
    """Run the full pipeline for specified tasks."""
    from .diagnostics import diagnose_result
    from .executor import execute_script
    from .extractor import extract_from_result, save_extracted_code
    from .runner import has_raw_result, load_tasks, run_task
    from .scorer import score_result

    tasks = load_tasks()
    task_ids = sorted(tasks.keys()) if args.all else [args.task]
    conditions = [args.condition] if args.condition else ["no_skill", "with_skill"]
    reps = args.reps

    total = len(task_ids) * len(conditions) * reps
    current = 0

    for cond in conditions:
        for task_id in task_ids:
            for rep in range(1, reps + 1):
                current += 1

                # Skip scenarios that already have results
                if (
                    not args.force
                    and not args.dry_run
                    and has_raw_result(task_id, cond, rep)
                ):
                    print(
                        f"  SKIP [{current}/{total}] {task_id} ({cond}, rep {rep})"
                        " — result exists"
                    )
                    continue

                print(f"\n{'#' * 60}")
                print(f"# Pipeline [{current}/{total}]: {task_id} ({cond}, rep {rep})")
                print(f"{'#' * 60}")

                # Phase 1: Generate
                result = run_task(
                    task_id,
                    cond,
                    rep,
                    model=args.model,
                    max_budget=args.max_budget,
                    dry_run=args.dry_run,
                    force=args.force,
                )
                if args.dry_run:
                    continue

                # Phase 1.5: Extract
                print("\n--- Extracting code ---")
                extraction = extract_from_result(result)
                code_path = save_extracted_code(extraction)
                if not extraction["success"]:
                    print(f"  Extraction failed: {extraction['error']}")
                    continue
                print(f"  Extracted to: {code_path}")

                # Phase 2: Execute
                print("\n--- Executing script ---")
                exec_result = execute_script(
                    code_path,
                    task_id,
                    cond,
                    rep,
                    timeout=args.timeout,
                )
                status = (
                    "OK"
                    if exec_result["execution_success"]
                    else f"FAIL ({exec_result['error_type']})"
                )
                print(f"  {status}")

                # Phase 3: Diagnose
                if exec_result.get("idata_exists"):
                    print("\n--- Computing diagnostics ---")
                    diag = diagnose_result(task_id, cond, rep)
                    conv = diag.get("convergence", {})
                    print(f"  Converged: {conv.get('converged', 'N/A')}")
                    print(f"  r_hat_max: {conv.get('r_hat_max', 'N/A')}")

                # Phase 4: Score
                print("\n--- Scoring ---")
                score = score_result(
                    task_id,
                    cond,
                    rep,
                    use_llm_judge=not args.no_llm,
                )
                print(f"  Total: {score['total_score']}/25")
                for k, v in score["scores"].items():
                    print(f"    {k}: {v}/5")

    if not args.dry_run:
        print(f"\n{'=' * 60}")
        print("Pipeline complete.")


def cmd_analyze(args):
    """Analyze results and generate reports."""
    from .analysis import print_effects, print_summary, write_report

    if args.summary:
        print_summary()
    elif args.effects:
        print_effects()
    elif args.report:
        write_report()
    else:
        print_summary()


def cmd_list_tasks(args):
    """List all benchmark tasks."""
    from .runner import load_tasks

    tasks = load_tasks()
    print(f"\nBenchmark Tasks ({len(tasks)}):")
    print("-" * 60)
    for task_id in sorted(tasks.keys()):
        task = tasks[task_id]
        print(f"  {task_id}: {task['name']} [{task['tier']}]")
        if task.get("dataset"):
            print(f"       dataset: {task['dataset']}")


def cmd_status(args):
    """Show completion status matrix."""
    from .diagnostics import DIAG_DIR
    from .executor import EXEC_DIR
    from .extractor import CODE_DIR
    from .runner import RAW_DIR, load_tasks
    from .scorer import SCORE_DIR

    tasks = load_tasks()
    conditions = ["no_skill", "with_skill"]

    print("\nBenchmark Status")
    print("=" * 80)
    print(
        f"{'Task':<6} {'Condition':<12} {'Raw':>5} {'Code':>5} {'Exec':>5} {'Diag':>5} {'Score':>5}"
    )
    print("-" * 80)

    for task_id in sorted(tasks.keys()):
        for condition in conditions:
            raw_count = (
                len(list(RAW_DIR.glob(f"{task_id}_{condition}_*.json")))
                if RAW_DIR.exists()
                else 0
            )
            code_count = (
                len(list(CODE_DIR.glob(f"{task_id}_{condition}_*.py")))
                if CODE_DIR.exists()
                else 0
            )
            exec_count = (
                len(list(EXEC_DIR.glob(f"{task_id}_{condition}_*.json")))
                if EXEC_DIR.exists()
                else 0
            )
            diag_count = (
                len(list(DIAG_DIR.glob(f"{task_id}_{condition}_*.json")))
                if DIAG_DIR.exists()
                else 0
            )
            score_count = (
                len(list(SCORE_DIR.glob(f"{task_id}_{condition}_*.json")))
                if SCORE_DIR.exists()
                else 0
            )

            print(
                f"  {task_id:<4} {condition:<12} "
                f"{raw_count:>5} {code_count:>5} {exec_count:>5} "
                f"{diag_count:>5} {score_count:>5}"
            )


def cmd_cleanup(args):
    """Check skill installation status."""
    from .runner import check_skill_status

    status = check_skill_status()
    print("Skill status:")
    print(f"  Installed: {status['skill_installed']}")
    print(f"  Path: {status['skill_path']}")
    print(f"  Content size: {status['skill_content_chars']} chars")
    if not status["skill_installed"]:
        print("  WARNING: Skill not installed. with_skill runs will fail.")


def main():
    parser = argparse.ArgumentParser(
        description="PyMC Skill Benchmark Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # run
    p_run = subparsers.add_parser("run", help="Generate code via Claude CLI")
    p_run.add_argument("--task", type=str, help="Task ID (T1-T12)")
    p_run.add_argument("--condition", choices=["with_skill", "no_skill"])
    p_run.add_argument("--rep", type=int, default=1, help="Replication number")
    p_run.add_argument("--all", action="store_true", help="Run all tasks")
    p_run.add_argument("--reps", type=int, default=3, help="Replications per task")
    p_run.add_argument("--model", default="sonnet", help="Claude model")
    p_run.add_argument("--max-budget", type=float, default=1.0, help="Max USD per call")
    p_run.add_argument("--dry-run", action="store_true", help="Print prompts only")
    p_run.add_argument(
        "--force", action="store_true", help="Re-run even if results exist"
    )
    p_run.set_defaults(func=cmd_run)

    # extract
    p_extract = subparsers.add_parser("extract", help="Extract code from results")
    p_extract.add_argument("--all", action="store_true", required=True)
    p_extract.set_defaults(func=cmd_extract)

    # execute
    p_execute = subparsers.add_parser("execute", help="Execute generated scripts")
    p_execute.add_argument("--all", action="store_true", required=True)
    p_execute.add_argument(
        "--timeout", type=int, default=600, help="Timeout per script (seconds)"
    )
    p_execute.set_defaults(func=cmd_execute)

    # diagnose
    p_diagnose = subparsers.add_parser("diagnose", help="Compute MCMC diagnostics")
    p_diagnose.add_argument("--all", action="store_true", required=True)
    p_diagnose.set_defaults(func=cmd_diagnose)

    # score
    p_score = subparsers.add_parser("score", help="Score all results")
    p_score.add_argument("--all", action="store_true", required=True)
    p_score.add_argument(
        "--no-llm",
        action="store_true",
        help="Skip LLM-as-judge, use automated fallback",
    )
    p_score.set_defaults(func=cmd_score)

    # pipeline
    p_pipe = subparsers.add_parser(
        "pipeline", help="Full pipeline: run→extract→execute→diagnose→score"
    )
    p_pipe.add_argument("--task", type=str, help="Single task ID")
    p_pipe.add_argument("--condition", choices=["with_skill", "no_skill"])
    p_pipe.add_argument("--all", action="store_true", help="All tasks")
    p_pipe.add_argument("--reps", type=int, default=3, help="Replications")
    p_pipe.add_argument("--model", default="sonnet", help="Claude model")
    p_pipe.add_argument(
        "--max-budget", type=float, default=1.0, help="Max USD per call"
    )
    p_pipe.add_argument("--timeout", type=int, default=600, help="Execution timeout")
    p_pipe.add_argument("--no-llm", action="store_true", help="Skip LLM-as-judge")
    p_pipe.add_argument("--dry-run", action="store_true", help="Print prompts only")
    p_pipe.add_argument(
        "--force", action="store_true", help="Re-run even if results exist"
    )
    p_pipe.set_defaults(func=cmd_pipeline)

    # analyze
    p_analyze = subparsers.add_parser("analyze", help="Analyze results")
    p_analyze.add_argument("--summary", action="store_true", help="Print summary")
    p_analyze.add_argument("--effects", action="store_true", help="Print effect sizes")
    p_analyze.add_argument(
        "--report", action="store_true", help="Generate markdown report"
    )
    p_analyze.set_defaults(func=cmd_analyze)

    # list-tasks
    p_list = subparsers.add_parser("list-tasks", help="List benchmark tasks")
    p_list.set_defaults(func=cmd_list_tasks)

    # status
    p_status = subparsers.add_parser("status", help="Show completion matrix")
    p_status.set_defaults(func=cmd_status)

    # cleanup
    p_cleanup = subparsers.add_parser("cleanup", help="Restore skill from backup")
    p_cleanup.set_defaults(func=cmd_cleanup)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
