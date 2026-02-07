"""Benchmark analysis — Cohen's d effect sizes, Polars reports.

Compares with_skill vs no_skill conditions across all tasks, computing
effect sizes and generating markdown reports.
"""

import json
import math
from pathlib import Path

import polars as pl

RESULTS_DIR = Path(__file__).parent.parent / "results"
SCORES_DIR = RESULTS_DIR / "scores"
ANALYSIS_DIR = RESULTS_DIR / "analysis"

CRITERIA = ["model_produced", "convergence", "model_appropriateness", "best_practices", "total"]


def load_scores(scores_dir: Path | None = None) -> pl.DataFrame:
    """Load all score JSONs into a Polars DataFrame."""
    if scores_dir is None:
        scores_dir = SCORES_DIR

    records = []
    for f in sorted(scores_dir.glob("*.json")):
        data = json.loads(f.read_text())
        records.append({
            "task_id": data["task_id"],
            "condition": data["condition"],
            "rep": data["rep"],
            "model_produced": data["model_produced"],
            "convergence": data["convergence"],
            "model_appropriateness": data["model_appropriateness"],
            "best_practices": data["best_practices"],
            "total": data["total"],
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
            "total": pl.Int64,
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
    pooled_sd = math.sqrt(pooled_var) if pooled_var > 0 else 1e-10

    return (mean2 - mean1) / pooled_sd


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
            pl.col("total").mean().alias("total_mean"),
            pl.col("total").count().alias("n"),
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
    lines.append("| Task | Condition | N | Produced | Convergence | Appropriateness | Best Practices | Total |")
    lines.append("|------|-----------|---|----------|-------------|-----------------|----------------|-------|")

    for row in summary.iter_rows(named=True):
        lines.append(
            f"| {row['task_id']} | {row['condition']} | {row['n']} | "
            f"{row['model_produced_mean']:.1f} | {row['convergence_mean']:.1f} | "
            f"{row['appropriateness_mean']:.1f} | {row['best_practices_mean']:.1f} | "
            f"{row['total_mean']:.1f} |"
        )

    lines.extend(["", "## Effect Sizes (Cohen's d)", ""])
    lines.append("Positive d = skill helps. |d| interpretation: <0.2 negligible, 0.2-0.5 small, 0.5-0.8 medium, >0.8 large.")
    lines.append("")
    lines.append("| Task | Criterion | d | no_skill mean | with_skill mean | Interpretation |")
    lines.append("|------|-----------|---|---------------|-----------------|----------------|")

    for row in effects.iter_rows(named=True):
        d_str = f"{row['d']:.2f}" if row["d"] is not None else "N/A"
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
        lines.append(f"- no_skill mean total: {ns_mean:.1f}/20")
        lines.append(f"- with_skill mean total: {ws_mean:.1f}/20")
        lines.append(f"- Cohen's d: {overall_d:.2f} ({_interpret_d(overall_d)})")

    report = "\n".join(lines)

    # Save report and data
    (output_dir / "report.md").write_text(report)
    summary.write_csv(str(output_dir / "summary.csv"))
    effects.write_csv(str(output_dir / "effects.csv"))

    return report
