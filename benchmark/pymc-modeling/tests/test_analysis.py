"""Tests for the benchmark analysis module."""

import json
from pathlib import Path

import polars as pl
import pytest

from src.analysis import (
    cohens_d,
    compute_effect_sizes,
    generate_report,
    load_scores,
    pass_rate_table,
    retries_table,
    summary_table,
)


def _create_score_files(scores_dir: Path, task_id: str, reps: int = 5):
    """Helper: create synthetic score files for testing."""
    scores_dir.mkdir(parents=True, exist_ok=True)

    for condition in ["no_skill", "with_skill"]:
        for rep in range(reps):
            # with_skill scores higher; add rep-based jitter for nonzero variance
            bonus = 1 if condition == "with_skill" else 0
            jitter = rep % 2  # 0, 1, 0 across reps
            score = {
                "task_id": task_id,
                "condition": condition,
                "rep": rep,
                "model_produced": 3 + bonus + jitter,
                "convergence": 3 + bonus,
                "model_appropriateness": 2 + bonus + jitter,
                "best_practices": 2 + bonus,
                "workflow": 3 + bonus,
                "parameter_recovery": 3 + bonus + jitter,
                "total": 16 + 6 * bonus + 3 * jitter,
                "passed": condition == "with_skill",
                "retries": 3 - bonus * 2,
            }
            fname = f"{task_id}_{condition}_rep{rep}.json"
            (scores_dir / fname).write_text(json.dumps(score))


class TestCohensD:
    def test_identical_groups(self):
        d = cohens_d([1, 2, 3], [1, 2, 3])
        assert abs(d) < 0.01

    def test_positive_effect(self):
        d = cohens_d([1, 2, 1], [5, 6, 5])
        assert d > 0  # group2 higher

    def test_negative_effect(self):
        d = cohens_d([5, 6, 5], [1, 2, 1])
        assert d < 0  # group2 lower

    def test_large_effect(self):
        d = cohens_d([0, 1, 0], [10, 11, 10])
        assert abs(d) > 0.8

    def test_zero_variance_different_means(self):
        import math
        d = cohens_d([1, 1, 1], [5, 5, 5])
        assert math.isnan(d)  # undefined when pooled SD is zero

    def test_zero_variance_same_means(self):
        d = cohens_d([3, 3, 3], [3, 3, 3])
        assert d == 0.0

    def test_insufficient_data(self):
        import math
        d = cohens_d([1], [2])
        assert math.isnan(d)


class TestLoadScores:
    def test_empty_dir(self, tmp_path):
        scores_dir = tmp_path / "scores"
        scores_dir.mkdir()
        df = load_scores(scores_dir)
        assert df.is_empty()

    def test_load_scores(self, tmp_path):
        scores_dir = tmp_path / "scores"
        _create_score_files(scores_dir, "T1_hierarchical")
        df = load_scores(scores_dir)
        assert len(df) == 10  # 2 conditions * 5 reps
        assert "task_id" in df.columns
        assert "total" in df.columns


class TestComputeEffectSizes:
    def test_effect_sizes(self, tmp_path):
        scores_dir = tmp_path / "scores"
        _create_score_files(scores_dir, "T1_hierarchical")
        df = load_scores(scores_dir)
        effects = compute_effect_sizes(df)
        assert not effects.is_empty()

        # Total effect should be positive (with_skill scores higher)
        total_effect = effects.filter(
            (pl.col("task_id") == "T1_hierarchical") & (pl.col("criterion") == "total")
        )
        assert len(total_effect) == 1
        d_value = total_effect.get_column("d")[0]
        assert d_value > 0  # with_skill should beat no_skill


class TestLoadScoresNewFields:
    def test_passed_and_retries_loaded(self, tmp_path):
        scores_dir = tmp_path / "scores"
        _create_score_files(scores_dir, "T1_hierarchical")
        df = load_scores(scores_dir)
        assert "passed" in df.columns
        assert "retries" in df.columns
        # with_skill runs should pass, no_skill should not
        ws = df.filter(pl.col("condition") == "with_skill")
        ns = df.filter(pl.col("condition") == "no_skill")
        assert ws.get_column("passed").sum() == 5
        assert ns.get_column("passed").sum() == 0

    def test_backward_compat_defaults(self, tmp_path):
        """Old score JSONs without passed/retries/workflow/parameter_recovery get defaults."""
        scores_dir = tmp_path / "scores"
        scores_dir.mkdir(parents=True)
        old_score = {
            "task_id": "T1_hierarchical",
            "condition": "no_skill",
            "rep": 0,
            "model_produced": 3,
            "convergence": 3,
            "model_appropriateness": 2,
            "best_practices": 2,
            "total": 10,
        }
        (scores_dir / "T1_hierarchical_no_skill_rep0.json").write_text(
            json.dumps(old_score)
        )
        df = load_scores(scores_dir)
        assert df.get_column("passed")[0] is False
        assert df.get_column("retries")[0] == 0
        assert df.get_column("workflow")[0] == 0
        assert df.get_column("parameter_recovery")[0] == 0


class TestSummaryTable:
    def test_summary(self, tmp_path):
        scores_dir = tmp_path / "scores"
        _create_score_files(scores_dir, "T1_hierarchical")
        df = load_scores(scores_dir)
        summary = summary_table(df)
        assert len(summary) == 2  # 2 conditions
        assert "total_mean" in summary.columns


class TestGenerateReport:
    def test_report_with_data(self, tmp_path):
        scores_dir = tmp_path / "scores"
        output_dir = tmp_path / "analysis"
        _create_score_files(scores_dir, "T1_hierarchical")
        _create_score_files(scores_dir, "T2_ordinal")

        report = generate_report(scores_dir=scores_dir, output_dir=output_dir)
        assert "Analysis Report" in report
        assert "T1_hierarchical" in report
        assert "Cohen's d" in report
        assert (output_dir / "report.md").exists()
        assert (output_dir / "summary.csv").exists()

    def test_report_has_pass_fail_summary(self, tmp_path):
        scores_dir = tmp_path / "scores"
        output_dir = tmp_path / "analysis"
        _create_score_files(scores_dir, "T1_hierarchical")
        report = generate_report(scores_dir=scores_dir, output_dir=output_dir)
        assert "Pass/Fail Summary" in report
        assert "pass rate" in report.lower()

    def test_report_has_retry_summary(self, tmp_path):
        scores_dir = tmp_path / "scores"
        output_dir = tmp_path / "analysis"
        _create_score_files(scores_dir, "T1_hierarchical")
        report = generate_report(scores_dir=scores_dir, output_dir=output_dir)
        assert "Retry Summary" in report
        assert "error-fix cycles" in report.lower()

    def test_report_empty(self, tmp_path):
        scores_dir = tmp_path / "scores"
        scores_dir.mkdir()
        report = generate_report(scores_dir=scores_dir, output_dir=tmp_path / "analysis")
        assert "No scores found" in report


class TestPassRateTable:
    def test_pass_rates(self, tmp_path):
        scores_dir = tmp_path / "scores"
        _create_score_files(scores_dir, "T1_hierarchical")
        df = load_scores(scores_dir)
        pr = pass_rate_table(df)
        assert len(pr) == 2  # 2 conditions
        assert "n_passed" in pr.columns
        assert "pass_rate" in pr.columns
        # with_skill should have 100% pass rate, no_skill 0%
        ws = pr.filter(pl.col("condition") == "with_skill")
        assert ws.get_column("pass_rate")[0] == 1.0
        ns = pr.filter(pl.col("condition") == "no_skill")
        assert ns.get_column("pass_rate")[0] == 0.0


class TestRetriesTable:
    def test_retries(self, tmp_path):
        scores_dir = tmp_path / "scores"
        _create_score_files(scores_dir, "T1_hierarchical")
        df = load_scores(scores_dir)
        rt = retries_table(df)
        assert len(rt) == 2
        assert "mean_retries" in rt.columns
        # no_skill has retries=3, with_skill has retries=1
        ns = rt.filter(pl.col("condition") == "no_skill")
        ws = rt.filter(pl.col("condition") == "with_skill")
        assert ns.get_column("mean_retries")[0] == 3.0
        assert ws.get_column("mean_retries")[0] == 1.0
