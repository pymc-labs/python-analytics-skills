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


def _create_score_files(scores_dir: Path, task_id: str, reps: int = 3):
    """Helper: create synthetic score files for testing."""
    scores_dir.mkdir(parents=True, exist_ok=True)

    for condition in ["no_skill", "with_skill"]:
        for rep in range(reps):
            bonus = 1 if condition == "with_skill" else 0
            jitter = rep % 2
            score = {
                "task_id": task_id,
                "condition": condition,
                "rep": rep,
                "iterative_approach": 2 + bonus + jitter,
                "prior_predictive": 1 + bonus + jitter,
                "model_criticism": 1 + bonus,
                "diagnostics": 3 + bonus,
                "reporting": 2 + bonus + jitter,
                "efficiency": 3 + bonus,
                "total": 12 + 6 * bonus + 3 * jitter,
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
        assert d > 0

    def test_negative_effect(self):
        d = cohens_d([5, 6, 5], [1, 2, 1])
        assert d < 0

    def test_large_effect(self):
        d = cohens_d([0, 1, 0], [10, 11, 10])
        assert abs(d) > 0.8

    def test_zero_variance_different_means(self):
        import math
        d = cohens_d([1, 1, 1], [5, 5, 5])
        assert math.isnan(d)

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
        _create_score_files(scores_dir, "W1_radon")
        df = load_scores(scores_dir)
        assert len(df) == 6  # 2 conditions * 3 reps
        assert "task_id" in df.columns
        assert "total" in df.columns
        assert "iterative_approach" in df.columns


class TestComputeEffectSizes:
    def test_effect_sizes(self, tmp_path):
        scores_dir = tmp_path / "scores"
        _create_score_files(scores_dir, "W1_radon")
        df = load_scores(scores_dir)
        effects = compute_effect_sizes(df)
        assert not effects.is_empty()

        total_effect = effects.filter(
            (pl.col("task_id") == "W1_radon") & (pl.col("criterion") == "total")
        )
        assert len(total_effect) == 1
        d_value = total_effect.get_column("d")[0]
        assert d_value > 0  # with_skill should beat no_skill


class TestSummaryTable:
    def test_summary(self, tmp_path):
        scores_dir = tmp_path / "scores"
        _create_score_files(scores_dir, "W1_radon")
        df = load_scores(scores_dir)
        summary = summary_table(df)
        assert len(summary) == 2
        assert "total_mean" in summary.columns
        assert "iterative_mean" in summary.columns


class TestGenerateReport:
    def test_report_with_data(self, tmp_path):
        scores_dir = tmp_path / "scores"
        output_dir = tmp_path / "analysis"
        _create_score_files(scores_dir, "W1_radon")
        _create_score_files(scores_dir, "W2_dose_response")

        report = generate_report(scores_dir=scores_dir, output_dir=output_dir)
        assert "Analysis Report" in report
        assert "W1_radon" in report
        assert "Cohen's d" in report
        assert (output_dir / "report.md").exists()
        assert (output_dir / "summary.csv").exists()

    def test_report_has_pass_fail_summary(self, tmp_path):
        scores_dir = tmp_path / "scores"
        output_dir = tmp_path / "analysis"
        _create_score_files(scores_dir, "W1_radon")
        report = generate_report(scores_dir=scores_dir, output_dir=output_dir)
        assert "Pass/Fail Summary" in report

    def test_report_has_retry_summary(self, tmp_path):
        scores_dir = tmp_path / "scores"
        output_dir = tmp_path / "analysis"
        _create_score_files(scores_dir, "W1_radon")
        report = generate_report(scores_dir=scores_dir, output_dir=output_dir)
        assert "Retry Summary" in report

    def test_report_empty(self, tmp_path):
        scores_dir = tmp_path / "scores"
        scores_dir.mkdir()
        report = generate_report(scores_dir=scores_dir, output_dir=tmp_path / "analysis")
        assert "No scores found" in report


class TestPassRateTable:
    def test_pass_rates(self, tmp_path):
        scores_dir = tmp_path / "scores"
        _create_score_files(scores_dir, "W1_radon")
        df = load_scores(scores_dir)
        pr = pass_rate_table(df)
        assert len(pr) == 2
        assert "n_passed" in pr.columns
        ws = pr.filter(pl.col("condition") == "with_skill")
        assert ws.get_column("pass_rate")[0] == 1.0
        ns = pr.filter(pl.col("condition") == "no_skill")
        assert ns.get_column("pass_rate")[0] == 0.0


class TestRetriesTable:
    def test_retries(self, tmp_path):
        scores_dir = tmp_path / "scores"
        _create_score_files(scores_dir, "W1_radon")
        df = load_scores(scores_dir)
        rt = retries_table(df)
        assert len(rt) == 2
        assert "mean_retries" in rt.columns
        ns = rt.filter(pl.col("condition") == "no_skill")
        ws = rt.filter(pl.col("condition") == "with_skill")
        assert ns.get_column("mean_retries")[0] == 3.0
        assert ws.get_column("mean_retries")[0] == 1.0
