"""Benchmark analysis — Cohen's d effect sizes, Polars reports.

Compares with_skill vs no_skill conditions across all tasks, computing
effect sizes and generating markdown reports.
"""

import json
import math
from pathlib import Path

import polars as pl

RESULTS_DIR = Path(__file__).parent.parent / "results"
RUNS_DIR = RESULTS_DIR / "runs"
SCORES_DIR = RESULTS_DIR / "scores"
ANALYSIS_DIR = RESULTS_DIR / "analysis"

TIMEOUT_CAP = 600  # seconds — winsorize wall_time at this value

CRITERIA = [
    "model_produced", "convergence", "model_appropriateness",
    "best_practices", "thrashing", "efficiency", "total",
]


def load_scores(scores_dir: Path | None = None) -> pl.DataFrame:
    """Load all score JSONs into a Polars DataFrame."""
    if scores_dir is None:
        scores_dir = SCORES_DIR

    records = []
    for f in sorted(scores_dir.glob("*.json")):
        data = json.loads(f.read_text())
        task_id = data["task_id"]
        condition = data["condition"]
        rep = data["rep"]

        # Load wall_time and cost from metadata
        run_name = f"{task_id}_{condition}_rep{rep}"
        meta_path = RUNS_DIR / run_name / "metadata.json"
        wall_time = 0.0
        cost_usd = 0.0
        num_turns = 0
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            wall_time = float(meta.get("wall_time", 0.0))
            cost_usd = float(meta.get("cost_usd", 0.0))
            num_turns = int(meta.get("num_turns", 0))

        # Winsorize wall_time at the timeout cap
        wall_time_winsorized = min(wall_time, TIMEOUT_CAP)

        records.append({
            "task_id": task_id,
            "condition": condition,
            "rep": rep,
            "model_produced": data["model_produced"],
            "convergence": data["convergence"],
            "model_appropriateness": data["model_appropriateness"],
            "best_practices": data["best_practices"],
            "thrashing": data.get("thrashing", 0),
            "efficiency": data.get("efficiency", 0),
            "total": data["total"],
            "passed": data.get("passed", False),
            "retries": data.get("retries", 0),
            "wall_time": wall_time,
            "wall_time_winsorized": wall_time_winsorized,
            "cost_usd": cost_usd,
            "num_turns": num_turns,
        })

    if not records:
        return pl.DataFrame(schema={
            "task_id": pl.Utf8,
            "condition": pl.Utf8,
            "rep": pl.Int64,
            "model_produced": pl.Int64,
            "convergence": pl.Int64,
            "model_appropriateness": pl.Int64,
            "best_practices": pl.Int64,
            "thrashing": pl.Int64,
            "efficiency": pl.Int64,
            "total": pl.Int64,
            "passed": pl.Boolean,
            "retries": pl.Int64,
            "wall_time": pl.Float64,
            "wall_time_winsorized": pl.Float64,
            "cost_usd": pl.Float64,
            "num_turns": pl.Int64,
        })

    return pl.DataFrame(records)


def cohens_d(group1: list[float], group2: list[float]) -> float:
    """Compute Cohen's d effect size (group2 - group1).

    Positive d means group2 scored higher than group1.
    Uses pooled standard deviation.
    """
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return float("nan")

    mean1 = sum(group1) / n1
    mean2 = sum(group2) / n2

    var1 = sum((x - mean1) ** 2 for x in group1) / (n1 - 1)
    var2 = sum((x - mean2) ** 2 for x in group2) / (n2 - 1)

    pooled_var = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)

    if pooled_var == 0:
        # Zero variance in both groups — d is 0 if means match, undefined otherwise
        return 0.0 if mean1 == mean2 else float("nan")

    return (mean2 - mean1) / math.sqrt(pooled_var)


def compute_effect_sizes(df: pl.DataFrame) -> pl.DataFrame:
    """Compute Cohen's d for each task and criterion.

    Returns a DataFrame with columns: task_id, criterion, d, no_skill_mean,
    with_skill_mean, no_skill_sd, with_skill_sd, n_no_skill, n_with_skill.
    """
    records = []

    task_ids = df.get_column("task_id").unique().sort().to_list()

    for task_id in task_ids:
        task_df = df.filter(pl.col("task_id") == task_id)

        for criterion in CRITERIA:
            no_skill = (
                task_df.filter(pl.col("condition") == "no_skill")
                .get_column(criterion)
                .to_list()
            )
            with_skill = (
                task_df.filter(pl.col("condition") == "with_skill")
                .get_column(criterion)
                .to_list()
            )

            no_skill_f = [float(x) for x in no_skill]
            with_skill_f = [float(x) for x in with_skill]

            d = cohens_d(no_skill_f, with_skill_f)

            records.append({
                "task_id": task_id,
                "criterion": criterion,
                "d": round(d, 3) if not math.isnan(d) else None,
                "no_skill_mean": round(sum(no_skill_f) / len(no_skill_f), 2) if no_skill_f else None,
                "with_skill_mean": round(sum(with_skill_f) / len(with_skill_f), 2) if with_skill_f else None,
                "n_no_skill": len(no_skill_f),
                "n_with_skill": len(with_skill_f),
            })

    return pl.DataFrame(records)


def _interpret_d(d: float | None) -> str:
    """Interpret Cohen's d magnitude."""
    if d is None or math.isnan(d):
        return "insufficient data"
    abs_d = abs(d)
    direction = "skill helps" if d > 0 else "skill hurts"
    if abs_d < 0.2:
        return f"negligible ({direction})"
    if abs_d < 0.5:
        return f"small ({direction})"
    if abs_d < 0.8:
        return f"medium ({direction})"
    return f"large ({direction})"


def summary_table(df: pl.DataFrame) -> pl.DataFrame:
    """Create a summary table: mean score by task and condition."""
    return (
        df.group_by(["task_id", "condition"])
        .agg([
            pl.col("model_produced").mean().alias("model_produced_mean"),
            pl.col("convergence").mean().alias("convergence_mean"),
            pl.col("model_appropriateness").mean().alias("appropriateness_mean"),
            pl.col("best_practices").mean().alias("best_practices_mean"),
            pl.col("thrashing").mean().alias("thrashing_mean"),
            pl.col("efficiency").mean().alias("efficiency_mean"),
            pl.col("total").mean().alias("total_mean"),
            pl.col("total").count().alias("n"),
        ])
        .sort(["task_id", "condition"])
    )


def pass_rate_table(df: pl.DataFrame) -> pl.DataFrame:
    """Compute pass rate by task and condition."""
    return (
        df.group_by(["task_id", "condition"])
        .agg([
            pl.col("passed").sum().alias("n_passed"),
            pl.col("passed").count().alias("n_total"),
            pl.col("passed").mean().alias("pass_rate"),
        ])
        .sort(["task_id", "condition"])
    )


def retries_table(df: pl.DataFrame) -> pl.DataFrame:
    """Compute retry statistics by task and condition."""
    return (
        df.group_by(["task_id", "condition"])
        .agg([
            pl.col("retries").mean().alias("mean_retries"),
            pl.col("retries").min().alias("min_retries"),
            pl.col("retries").max().alias("max_retries"),
        ])
        .sort(["task_id", "condition"])
    )


def generate_report(
    scores_dir: Path | None = None,
    output_dir: Path | None = None,
) -> str:
    """Generate a full markdown analysis report."""
    if output_dir is None:
        output_dir = ANALYSIS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_scores(scores_dir)

    if df.is_empty():
        return "No scores found. Run scoring first."

    summary = summary_table(df)
    effects = compute_effect_sizes(df)

    lines = [
        "# PyMC Skill Benchmark — Analysis Report",
        "",
        "## Summary",
        "",
        f"Total runs scored: {len(df)}",
        f"Tasks: {df.get_column('task_id').n_unique()}",
        f"Replications per condition: {df.group_by(['task_id', 'condition']).len().get_column('len').max()}",
        "",
        "## Scores by Task and Condition",
        "",
    ]

    # Summary table as markdown
    lines.append("| Task | Condition | N | Produced | Convergence | Appropriateness | Best Practices | Thrashing | Efficiency | Total |")
    lines.append("|------|-----------|---|----------|-------------|-----------------|----------------|-----------|------------|-------|")

    for row in summary.iter_rows(named=True):
        lines.append(
            f"| {row['task_id']} | {row['condition']} | {row['n']} | "
            f"{row['model_produced_mean']:.1f} | {row['convergence_mean']:.1f} | "
            f"{row['appropriateness_mean']:.1f} | {row['best_practices_mean']:.1f} | "
            f"{row['thrashing_mean']:.1f} | {row['efficiency_mean']:.1f} | "
            f"{row['total_mean']:.1f} |"
        )

    # Pass/Fail Summary
    pass_rates = pass_rate_table(df)
    lines.extend([
        "",
        "## Pass/Fail Summary",
        "",
        "A run passes if: sampling completed (produced >= 4), convergence acceptable",
        "(convergence >= 3), and posterior estimates are finite and non-degenerate.",
        "",
        "| Task | Condition | Passed | Total | Pass Rate |",
        "|------|-----------|--------|-------|-----------|",
    ])
    for row in pass_rates.iter_rows(named=True):
        n_passed = int(row["n_passed"])
        n_total = int(row["n_total"])
        rate = row["pass_rate"]
        rate_str = f"{rate:.0%}" if rate is not None else "N/A"
        lines.append(
            f"| {row['task_id']} | {row['condition']} | {n_passed} | {n_total} | {rate_str} |"
        )

    # Retry Summary
    retries = retries_table(df)
    lines.extend([
        "",
        "## Retry Summary",
        "",
        "Number of error-fix cycles (model.py rewrites) per run. Lower = better.",
        "",
        "| Task | Condition | Mean Retries | Min | Max |",
        "|------|-----------|-------------|-----|-----|",
    ])
    for row in retries.iter_rows(named=True):
        lines.append(
            f"| {row['task_id']} | {row['condition']} | "
            f"{row['mean_retries']:.1f} | {row['min_retries']} | {row['max_retries']} |"
        )

    # Cost and Efficiency table (wall_time winsorized at TIMEOUT_CAP)
    efficiency_stats = (
        df.group_by(["task_id", "condition"])
        .agg([
            pl.col("num_turns").mean().alias("mean_turns"),
            pl.col("wall_time_winsorized").mean().alias("mean_wall_time"),
            pl.col("cost_usd").mean().alias("mean_cost"),
        ])
        .sort(["task_id", "condition"])
    )

    lines.extend([
        "",
        "## Cost and Efficiency",
        "",
        f"Wall times winsorized at {TIMEOUT_CAP}s timeout cap.",
        "",
        "| Task | Condition | Mean Turns | Mean Wall Time | Mean Cost |",
        "|------|-----------|-----------|----------------|-----------|",
    ])
    for row in efficiency_stats.iter_rows(named=True):
        lines.append(
            f"| {row['task_id']} | {row['condition']} | "
            f"{row['mean_turns']:.1f} | {row['mean_wall_time']:.0f}s | "
            f"${row['mean_cost']:.2f} |"
        )

    # Overall averages
    for cond in ["no_skill", "with_skill"]:
        cond_df = df.filter(pl.col("condition") == cond)
        if not cond_df.is_empty():
            lines.append(
                f"| **All tasks** | **{cond}** | "
                f"**{cond_df.get_column('num_turns').mean():.1f}** | "
                f"**{cond_df.get_column('wall_time_winsorized').mean():.0f}s** | "
                f"**${cond_df.get_column('cost_usd').mean():.2f}** |"
            )

    lines.extend(["", "## Effect Sizes (Cohen's d)", ""])
    lines.append("Positive d = skill helps. |d| interpretation: <0.2 negligible, 0.2-0.5 small, 0.5-0.8 medium, >0.8 large.")
    lines.append("")
    lines.append("| Task | Criterion | d | no_skill mean | with_skill mean | Interpretation |")
    lines.append("|------|-----------|---|---------------|-----------------|----------------|")

    for row in effects.iter_rows(named=True):
        d_val = row["d"]
        d_str = "N/A" if d_val is None or math.isnan(d_val) else f"{d_val:.2f}"
        ns_mean = f"{row['no_skill_mean']:.1f}" if row["no_skill_mean"] is not None else "N/A"
        ws_mean = f"{row['with_skill_mean']:.1f}" if row["with_skill_mean"] is not None else "N/A"
        interp = _interpret_d(row["d"])
        lines.append(
            f"| {row['task_id']} | {row['criterion']} | {d_str} | "
            f"{ns_mean} | {ws_mean} | {interp} |"
        )

    # Overall effect
    lines.extend(["", "## Overall Effect", ""])
    no_skill_total = df.filter(pl.col("condition") == "no_skill").get_column("total").to_list()
    with_skill_total = df.filter(pl.col("condition") == "with_skill").get_column("total").to_list()

    if no_skill_total and with_skill_total:
        overall_d = cohens_d(
            [float(x) for x in no_skill_total],
            [float(x) for x in with_skill_total],
        )
        ns_mean = sum(no_skill_total) / len(no_skill_total)
        ws_mean = sum(with_skill_total) / len(with_skill_total)
        lines.append(f"- no_skill mean total: {ns_mean:.1f}/30")
        lines.append(f"- with_skill mean total: {ws_mean:.1f}/30")
        lines.append(f"- Cohen's d: {overall_d:.2f} ({_interpret_d(overall_d)})")

        # Pass rates
        no_skill_passed = df.filter(pl.col("condition") == "no_skill").get_column("passed")
        with_skill_passed = df.filter(pl.col("condition") == "with_skill").get_column("passed")
        ns_pass_n = int(no_skill_passed.sum())
        ns_pass_total = len(no_skill_passed)
        ws_pass_n = int(with_skill_passed.sum())
        ws_pass_total = len(with_skill_passed)
        ns_pass_pct = ns_pass_n / ns_pass_total * 100 if ns_pass_total else 0
        ws_pass_pct = ws_pass_n / ws_pass_total * 100 if ws_pass_total else 0
        lines.append(f"- no_skill pass rate: {ns_pass_pct:.0f}% ({ns_pass_n}/{ns_pass_total})")
        lines.append(f"- with_skill pass rate: {ws_pass_pct:.0f}% ({ws_pass_n}/{ws_pass_total})")

        # Retries
        no_skill_retries = df.filter(pl.col("condition") == "no_skill").get_column("retries").to_list()
        with_skill_retries = df.filter(pl.col("condition") == "with_skill").get_column("retries").to_list()
        ns_retry_mean = sum(no_skill_retries) / len(no_skill_retries) if no_skill_retries else 0
        ws_retry_mean = sum(with_skill_retries) / len(with_skill_retries) if with_skill_retries else 0
        lines.append(f"- no_skill mean retries: {ns_retry_mean:.1f}")
        lines.append(f"- with_skill mean retries: {ws_retry_mean:.1f}")

    report = "\n".join(lines)

    # Save report and data
    (output_dir / "report.md").write_text(report)
    summary.write_csv(str(output_dir / "summary.csv"))
    effects.write_csv(str(output_dir / "effects.csv"))

    return report
