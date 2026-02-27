"""Tests for the benchmark scorer."""

import json
from pathlib import Path

import numpy as np
import pytest

from src.scorer import (
    _extract_judge_json,
    count_retries,
    evaluate_pass_fail,
    score_diagnostics,
    score_efficiency,
)
from src.extractor import extract_workflow_trace


@pytest.fixture
def run_dir(tmp_path):
    """Create a temporary run directory."""
    return tmp_path


def _write_model_py(run_dir: Path, code: str):
    """Helper: write model.py to run_dir."""
    (run_dir / "model.py").write_text(code)


def _write_metadata(run_dir, num_turns=10, success=True):
    """Helper: write metadata.json."""
    (run_dir / "metadata.json").write_text(json.dumps({
        "task_id": "W1_radon",
        "condition": "no_skill",
        "rep": 0,
        "success": success,
        "num_turns": num_turns,
    }))


def _create_idata(run_dir: Path, n_chains=4, n_draws=1000, n_divergent=0,
                   has_pp=False, has_ll=False):
    """Helper: create a synthetic InferenceData and save to results.nc."""
    import arviz as az
    import xarray as xr

    rng = np.random.default_rng(42)

    mu = rng.normal(0, 1, (n_chains, n_draws))
    sigma = np.abs(rng.normal(1, 0.1, (n_chains, n_draws)))
    posterior = xr.Dataset(
        {
            "mu": (["chain", "draw"], mu),
            "sigma": (["chain", "draw"], sigma),
        },
        coords={"chain": range(n_chains), "draw": range(n_draws)},
    )

    groups = {"posterior": posterior}

    diverging = np.zeros((n_chains, n_draws), dtype=bool)
    if n_divergent > 0:
        flat = diverging.ravel()
        indices = rng.choice(len(flat), min(n_divergent, len(flat)), replace=False)
        flat[indices] = True
        diverging = flat.reshape(n_chains, n_draws)

    sample_stats = xr.Dataset(
        {"diverging": (["chain", "draw"], diverging)},
        coords={"chain": range(n_chains), "draw": range(n_draws)},
    )
    groups["sample_stats"] = sample_stats

    if has_pp:
        pp = xr.Dataset(
            {"y_pred": (["chain", "draw"], rng.normal(0, 1, (n_chains, n_draws)))},
            coords={"chain": range(n_chains), "draw": range(n_draws)},
        )
        groups["posterior_predictive"] = pp

    if has_ll:
        ll = xr.Dataset(
            {"y": (["chain", "draw", "y_dim_0"], rng.normal(0, 1, (n_chains, n_draws, 10)))},
            coords={"chain": range(n_chains), "draw": range(n_draws), "y_dim_0": range(10)},
        )
        groups["log_likelihood"] = ll

    idata = az.InferenceData(**groups)
    idata.to_netcdf(str(run_dir / "results.nc"))
    return idata


def _make_assistant_turn(tool_name, tool_input):
    """Helper: create an assistant turn with a single tool_use block."""
    return {
        "type": "assistant",
        "message": {
            "content": [
                {"type": "tool_use", "name": tool_name, "input": tool_input},
            ]
        },
    }


def _write_turns_jsonl(run_dir, turns):
    """Helper: write turns.jsonl with assistant messages."""
    path = run_dir / "turns.jsonl"
    with open(path, "w") as f:
        for turn in turns:
            f.write(json.dumps(turn) + "\n")


class TestExtractJudgeJson:
    def test_valid_json(self):
        result = _extract_judge_json('{"score": 3, "reasoning": "good"}')
        assert result["score"] == 3

    def test_json_in_fences(self):
        result = _extract_judge_json('```json\n{"score": 4, "reasoning": "great"}\n```')
        assert result["score"] == 4

    def test_score_extraction(self):
        result = _extract_judge_json('The score is {"score": 2, "reasoning": "ok"}')
        assert result["score"] == 2

    def test_invalid_response(self):
        result = _extract_judge_json("no json here")
        assert result is None

    def test_partial_json(self):
        result = _extract_judge_json('Some text with "score": 5 in it')
        assert result is not None
        assert result["score"] == 5


class TestScoreDiagnostics:
    def test_no_nc(self, run_dir):
        trace = extract_workflow_trace(run_dir)
        score, details = score_diagnostics(run_dir, trace)
        assert score == 0

    def test_good_convergence(self, run_dir):
        _create_idata(run_dir, n_chains=4, n_draws=1000, n_divergent=0)
        trace = extract_workflow_trace(run_dir)
        score, details = score_diagnostics(run_dir, trace)
        assert score >= 2

    def test_divergences_lower_score(self, run_dir):
        _create_idata(run_dir, n_chains=4, n_draws=1000, n_divergent=200)
        trace = extract_workflow_trace(run_dir)
        score, details = score_diagnostics(run_dir, trace)
        assert score <= 2

    def test_trace_diagnostics_bonus(self, run_dir):
        """Trace evidence of diagnostic checking gives bonus."""
        _create_idata(run_dir, n_chains=4, n_draws=1000, n_divergent=0)
        turns = [
            _make_assistant_turn("Bash", {"command": "python -c 'az.summary(idata)'"}),
            _make_assistant_turn("Bash", {"command": "python -c 'az.plot_trace(idata)'"}),
            _make_assistant_turn("Bash", {"command": "python -c 'az.ess(idata)'"}),
        ]
        _write_turns_jsonl(run_dir, turns)
        trace = extract_workflow_trace(run_dir)
        score, details = score_diagnostics(run_dir, trace)
        assert score >= 4  # base 3 + bonus from trace diagnostics


class TestScoreEfficiency:
    def test_fast_run(self, run_dir):
        """10 turns → score 5."""
        _write_metadata(run_dir, num_turns=10)
        score, details = score_efficiency(run_dir)
        assert score == 5

    def test_moderate_run(self, run_dir):
        """20 turns → score 3."""
        _write_metadata(run_dir, num_turns=20)
        score, details = score_efficiency(run_dir)
        assert score == 3

    def test_slow_run(self, run_dir):
        """40 turns → score 1."""
        _write_metadata(run_dir, num_turns=40)
        score, details = score_efficiency(run_dir)
        assert score == 1

    def test_timeout(self, run_dir):
        """0 turns → score 0."""
        _write_metadata(run_dir, num_turns=0)
        score, details = score_efficiency(run_dir)
        assert score == 0

    def test_no_metadata(self, run_dir):
        """No metadata → score 0."""
        score, details = score_efficiency(run_dir)
        assert score == 0

    def test_very_slow_run(self, run_dir):
        """50 turns → score 0."""
        _write_metadata(run_dir, num_turns=50)
        score, details = score_efficiency(run_dir)
        assert score == 0


class TestEvaluatePassFail:
    def test_full_pass(self, run_dir):
        """Good idata, high scores → passed=True."""
        _create_idata(run_dir, n_chains=4, n_draws=1000, n_divergent=0)
        passed, details = evaluate_pass_fail(run_dir, diagnostics_score=4)
        assert passed is True
        assert details["sampling_completed"] is True
        assert details["diagnostics_acceptable"] is True
        assert details["non_degenerate"] is True

    def test_fail_no_nc(self, run_dir):
        """No results.nc → fail."""
        passed, details = evaluate_pass_fail(run_dir, diagnostics_score=4)
        assert passed is False

    def test_fail_low_diagnostics(self, run_dir):
        """Good idata but diagnostics_score=1 → fail."""
        _create_idata(run_dir, n_chains=4, n_draws=1000)
        passed, details = evaluate_pass_fail(run_dir, diagnostics_score=1)
        assert passed is False
        assert details["diagnostics_acceptable"] is False

    def test_fail_degenerate(self, run_dir):
        """All vars have zero std → fail."""
        import arviz as az
        import xarray as xr

        constant = np.full((4, 1000), 5.0)
        posterior = xr.Dataset(
            {"mu": (["chain", "draw"], constant)},
            coords={"chain": range(4), "draw": range(1000)},
        )
        idata = az.InferenceData(posterior=posterior)
        idata.to_netcdf(str(run_dir / "results.nc"))

        passed, details = evaluate_pass_fail(run_dir, diagnostics_score=3)
        assert passed is False
        assert "degenerate" in details["reason"]


class TestCountRetries:
    def test_no_rewrites(self, run_dir):
        """Single model file → 0 retries."""
        turns = [
            _make_assistant_turn("Write", {"file_path": "/tmp/work/model.py", "content": "v1"}),
            _make_assistant_turn("Bash", {"command": "python model.py"}),
        ]
        _write_turns_jsonl(run_dir, turns)
        retries, details = count_retries(run_dir)
        assert retries == 0

    def test_one_rewrite(self, run_dir):
        """Same file written twice → 1 rewrite."""
        turns = [
            _make_assistant_turn("Write", {"file_path": "/tmp/work/model.py", "content": "v1"}),
            _make_assistant_turn("Bash", {"command": "python model.py"}),
            _make_assistant_turn("Write", {"file_path": "/tmp/work/model.py", "content": "v2"}),
            _make_assistant_turn("Bash", {"command": "python model.py"}),
        ]
        _write_turns_jsonl(run_dir, turns)
        retries, details = count_retries(run_dir)
        assert retries == 1

    def test_new_model_versions_not_retries(self, run_dir):
        """Different model files are not retries."""
        turns = [
            _make_assistant_turn("Write", {"file_path": "/tmp/work/model.py", "content": "v1"}),
            _make_assistant_turn("Write", {"file_path": "/tmp/work/model_v2.py", "content": "v2"}),
        ]
        _write_turns_jsonl(run_dir, turns)
        retries, details = count_retries(run_dir)
        assert retries == 0  # new versions, not rewrites

    def test_fallback_from_metadata(self, run_dir):
        """No turns.jsonl, metadata num_turns=24 → retries=2."""
        _write_metadata(run_dir, num_turns=24)
        retries, details = count_retries(run_dir)
        assert retries == 2
        assert details["method"] == "fallback_num_turns"

    def test_no_data(self, run_dir):
        """No turns.jsonl, no metadata → retries=0."""
        retries, details = count_retries(run_dir)
        assert retries == 0
