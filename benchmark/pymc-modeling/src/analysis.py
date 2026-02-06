"""Polars-based analysis and reporting.

Loads scored results and computes summary statistics, effect sizes,
and generates markdown reports.
"""

from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path

import polars as pl

BENCHMARK_DIR = Path(__file__).resolve().parent.parent
SCORE_DIR = BENCHMARK_DIR / "results" / "scores"
REPORT_DIR = BENCHMARK_DIR / "results" / "reports"


def load_scores() -> pl.DataFrame:
    """Load all score JSONs into a polars DataFrame."""
    if not SCORE_DIR.exists():
        return pl.DataFrame()

    records = []
    for path in sorted(SCORE_DIR.glob("*.json")):
        with open(path) as f:
            data = json.load(f)

        scores = data.get("scores", {})
        metrics = data.get("metrics", {})

        records.append(
            {
                "task_id": data.get("task_id"),
                "condition": data.get("condition"),
                "replication": data.get("replication"),
                "task_name": data.get("task_name"),
                "tier": data.get("tier"),
                "total_score": data.get("total_score", 0),
                "execution": scores.get("execution", 0),
                "convergence": scores.get("convergence", 0),
                "diagnostic_completeness": scores.get("diagnostic_completeness", 0),
                "statistical_quality": scores.get("statistical_quality", 0),
                "code_quality": scores.get("code_quality", 0),
                "generation_time": metrics.get("generation_time_seconds"),
                "execution_time": metrics.get("execution_time_seconds"),
                "input_tokens": metrics.get("input_tokens"),
                "output_tokens": metrics.get("output_tokens"),
                "cost_usd": metrics.get("cost_usd"),
                "idata_exists": metrics.get("idata_exists", False),
                "execution_success": metrics.get("execution_success", False),
                "error_type": metrics.get("error_type"),
            }
        )

    if not records:
        return pl.DataFrame()

    return pl.DataFrame(records)


def cohens_d(group1: pl.Series, group2: pl.Series) -> float:
    """Compute Cohen's d effect size."""
    n1 = group1.drop_nulls().len()
    n2 = group2.drop_nulls().len()

    if n1 < 2 or n2 < 2:
        return float("nan")

    mean1 = group1.drop_nulls().mean()
    mean2 = group2.drop_nulls().mean()
    var1 = group1.drop_nulls().var()
    var2 = group2.drop_nulls().var()

    pooled_std = math.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0.0

    return (mean2 - mean1) / pooled_std


def summary_by_condition(df: pl.DataFrame) -> pl.DataFrame:
    """Summary statistics by condition."""
    score_cols = [
        "total_score",
        "execution",
        "convergence",
        "diagnostic_completeness",
        "statistical_quality",
        "code_quality",
    ]
    metric_cols = ["generation_time", "execution_time", "output_tokens", "cost_usd"]

    aggs = [pl.col("task_id").count().alias("n")]
    for col in score_cols + metric_cols:
        aggs.extend(
            [
                pl.col(col).mean().round(2).alias(f"{col}_mean"),
                pl.col(col).std().round(2).alias(f"{col}_std"),
            ]
        )

    # Success rates
    aggs.append(
        pl.col("execution_success").mean().round(3).alias("execution_success_rate")
    )
    aggs.append(pl.col("idata_exists").mean().round(3).alias("idata_production_rate"))

    return df.group_by("condition").agg(aggs).sort("condition")


def summary_by_task(df: pl.DataFrame) -> pl.DataFrame:
    """Summary by task and condition."""
    return (
        df.group_by(["task_id", "task_name", "tier", "condition"])
        .agg(
            [
                pl.col("total_score").mean().round(1).alias("mean_score"),
                pl.col("total_score").std().round(1).alias("std_score"),
                pl.col("execution_success").mean().round(2).alias("exec_rate"),
                pl.col("idata_exists").mean().round(2).alias("idata_rate"),
                pl.col("task_id").count().alias("n"),
            ]
        )
        .sort(["task_id", "condition"])
    )


def summary_by_tier(df: pl.DataFrame) -> pl.DataFrame:
    """Summary by tier and condition."""
    return (
        df.group_by(["tier", "condition"])
        .agg(
            [
                pl.col("total_score").mean().round(1).alias("mean_score"),
                pl.col("total_score").std().round(1).alias("std_score"),
                pl.col("task_id").count().alias("n"),
            ]
        )
        .sort(["tier", "condition"])
    )


def compute_effect_sizes(df: pl.DataFrame) -> dict[str, float]:
    """Compute Cohen's d for each metric (with_skill vs no_skill)."""
    no_skill = df.filter(pl.col("condition") == "no_skill")
    with_skill = df.filter(pl.col("condition") == "with_skill")

    metrics = [
        "total_score",
        "execution",
        "convergence",
        "diagnostic_completeness",
        "statistical_quality",
        "code_quality",
        "generation_time",
        "execution_time",
        "output_tokens",
    ]

    effects = {}
    for metric in metrics:
        g1 = no_skill[metric].drop_nulls()
        g2 = with_skill[metric].drop_nulls()
        d = cohens_d(g1, g2)
        effects[metric] = round(d, 3) if not math.isnan(d) else None

    return effects


def failure_analysis(df: pl.DataFrame) -> pl.DataFrame:
    """Categorize failure modes by condition."""
    failures = df.filter(~pl.col("execution_success"))
    if failures.is_empty():
        return pl.DataFrame({"message": ["No failures"]})

    return (
        failures.group_by(["condition", "error_type"])
        .agg(pl.col("task_id").count().alias("count"))
        .sort(["condition", "error_type"])
    )


def generate_report(df: pl.DataFrame) -> str:
    """Generate a markdown benchmark report."""
    lines = [
        "# PyMC Skill Benchmark Report",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
    ]

    if df.is_empty():
        lines.append("No results to report.")
        return "\n".join(lines)

    # Executive summary
    effects = compute_effect_sizes(df)
    by_cond = summary_by_condition(df)

    lines.extend(
        [
            "## Executive Summary",
            "",
            "| Metric | No Skill | With Skill | Effect Size (d) |",
            "|--------|----------|------------|-----------------|",
        ]
    )

    no_skill = by_cond.filter(pl.col("condition") == "no_skill")
    with_skill = by_cond.filter(pl.col("condition") == "with_skill")

    key_metrics = [
        ("Total Score", "total_score"),
        ("Execution", "execution"),
        ("Convergence", "convergence"),
        ("Diagnostics", "diagnostic_completeness"),
        ("Statistical Quality", "statistical_quality"),
        ("Code Quality", "code_quality"),
    ]

    for label, col in key_metrics:
        ns_mean = _get_val(no_skill, f"{col}_mean")
        ns_std = _get_val(no_skill, f"{col}_std")
        ws_mean = _get_val(with_skill, f"{col}_mean")
        ws_std = _get_val(with_skill, f"{col}_std")
        d = effects.get(col, "N/A")
        d_str = f"{d:.2f}" if isinstance(d, float) else "N/A"

        lines.append(
            f"| {label} | {ns_mean} ± {ns_std} | {ws_mean} ± {ws_std} | {d_str} |"
        )

    lines.append("")

    # Sample sizes
    lines.extend(
        [
            "## Sample Size",
            "",
        ]
    )
    for row in by_cond.iter_rows(named=True):
        lines.append(f"- **{row['condition']}**: {row['n']} runs")
    lines.append("")

    # Success rates
    lines.extend(
        [
            "## Success Rates",
            "",
            "| Condition | Execution Success | InferenceData Produced |",
            "|-----------|-------------------|------------------------|",
        ]
    )
    for row in by_cond.iter_rows(named=True):
        exec_rate = f"{row['execution_success_rate']:.0%}"
        idata_rate = f"{row['idata_production_rate']:.0%}"
        lines.append(f"| {row['condition']} | {exec_rate} | {idata_rate} |")
    lines.append("")

    # By task
    by_task = summary_by_task(df)
    lines.extend(
        [
            "## Results by Task",
            "",
            "| Task | Name | Tier | Condition | Mean Score | Std | Exec Rate | iData Rate | N |",
            "|------|------|------|-----------|------------|-----|-----------|------------|---|",
        ]
    )
    for row in by_task.iter_rows(named=True):
        lines.append(
            f"| {row['task_id']} | {row['task_name']} | {row['tier']} | "
            f"{row['condition']} | {row['mean_score']} | {row['std_score']} | "
            f"{row['exec_rate']:.0%} | {row['idata_rate']:.0%} | {row['n']} |"
        )
    lines.append("")

    # By tier
    by_tier = summary_by_tier(df)
    lines.extend(
        [
            "## Results by Tier",
            "",
            "| Tier | Condition | Mean Score | Std | N |",
            "|------|-----------|------------|-----|---|",
        ]
    )
    for row in by_tier.iter_rows(named=True):
        lines.append(
            f"| {row['tier']} | {row['condition']} | "
            f"{row['mean_score']} | {row['std_score']} | {row['n']} |"
        )
    lines.append("")

    # Failure analysis
    failures = failure_analysis(df)
    if "message" not in failures.columns:
        lines.extend(
            [
                "## Failure Analysis",
                "",
                "| Condition | Error Type | Count |",
                "|-----------|------------|-------|",
            ]
        )
        for row in failures.iter_rows(named=True):
            lines.append(
                f"| {row['condition']} | {row['error_type']} | {row['count']} |"
            )
        lines.append("")

    # Effect size interpretation
    lines.extend(
        [
            "## Effect Size Interpretation",
            "",
            "Cohen's d benchmarks: small=0.2, medium=0.5, large=0.8",
            "",
            "Positive d means **with_skill** scored higher (better).",
            "Negative d means **no_skill** scored higher.",
            "",
            "## Methodology",
            "",
            "- **Conditions**: no_skill (skill directory moved) vs with_skill (skill auto-loads)",
            "- **Scoring**: Automated from real execution + MCMC diagnostics + LLM-as-judge",
            "- **Metrics**: 5 criteria x 0-5 scale = 25 points max",
            "",
        ]
    )

    return "\n".join(lines)


def _get_val(df: pl.DataFrame, col: str) -> str:
    """Get a single value from a DataFrame column, formatted."""
    if df.is_empty() or col not in df.columns:
        return "N/A"
    val = df[col][0]
    if val is None:
        return "N/A"
    if isinstance(val, float):
        return f"{val:.2f}"
    return str(val)


def write_report() -> Path:
    """Load scores, generate report, and write to file."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_scores()
    report = generate_report(df)

    report_path = REPORT_DIR / "benchmark_report.md"
    report_path.write_text(report)
    print(f"Report written to: {report_path}")
    return report_path


def print_summary() -> None:
    """Print summary statistics to stdout."""
    df = load_scores()
    if df.is_empty():
        print("No scores found.")
        return

    print("\n=== Benchmark Summary ===\n")

    by_cond = summary_by_condition(df)
    print(by_cond)

    print("\n=== Effect Sizes (Cohen's d) ===\n")
    effects = compute_effect_sizes(df)
    for metric, d in effects.items():
        d_str = f"{d:+.3f}" if d is not None else "N/A"
        print(f"  {metric:30s} {d_str}")


def print_effects() -> None:
    """Print effect sizes to stdout."""
    df = load_scores()
    if df.is_empty():
        print("No scores found.")
        return

    effects = compute_effect_sizes(df)
    print("\n=== Effect Sizes (Cohen's d) ===")
    print("Positive = with_skill better, Negative = no_skill better\n")
    for metric, d in effects.items():
        if d is None:
            interp = "insufficient data"
        elif abs(d) < 0.2:
            interp = "negligible"
        elif abs(d) < 0.5:
            interp = "small"
        elif abs(d) < 0.8:
            interp = "medium"
        else:
            interp = "large"
        d_str = f"{d:+.3f}" if d is not None else "  N/A"
        print(f"  {metric:30s} d = {d_str}  ({interp})")
