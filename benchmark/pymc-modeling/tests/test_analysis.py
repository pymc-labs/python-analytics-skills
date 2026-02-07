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
    summary_table,
)


def _create_score_files(scores_dir: Path, task_id: str, reps: int = 3):
    """Helper: create synthetic score files for testing."""
    scores_dir.mkdir(parents=True, exist_ok=True)

    for condition in ["no_skill", "with_skill"]:
        for rep in range(reps):
            # with_skill scores slightly higher
            bonus = 1 if condition == "with_skill" else 0
            score = {
                "task_id": task_id,
                "condition": condition,
                "rep": rep,
                "model_produced": 3 + bonus,
                "convergence": 3 + bonus,
                "model_appropriateness": 2 + bonus,
                "best_practices": 2 + bonus,
                "total": 10 + 4 * bonus,
            }
            fname = f"{task_id}_{condition}_rep{rep}.json"
            (scores_dir / fname).write_text(json.dumps(score))


class TestCohensD:
    def test_identical_groups(self):
        d = cohens_d([1, 2, 3], [1, 2, 3])
        assert abs(d) < 0.01

    def test_positive_effect(self):
        d = cohens_d([1, 1, 1], [5, 5, 5])
        assert d > 0  # group2 higher

    def test_negative_effect(self):
        d = cohens_d([5, 5, 5], [1, 1, 1])
        assert d < 0  # group2 lower

    def test_large_effect(self):
        d = cohens_d([0, 0, 0], [10, 10, 10])
        assert abs(d) > 0.8

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
        assert len(df) == 6  # 2 conditions * 3 reps
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

    def test_report_empty(self, tmp_path):
        scores_dir = tmp_path / "scores"
        scores_dir.mkdir()
        report = generate_report(scores_dir=scores_dir, output_dir=tmp_path / "analysis")
        assert "No scores found" in report
