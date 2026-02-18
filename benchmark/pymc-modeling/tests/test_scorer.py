"""Tests for the benchmark scorer."""

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from src.scorer import (
    _score_appropriateness_regex,
    count_retries,
    evaluate_pass_fail,
    score_best_practices,
    score_convergence,
    score_efficiency,
    score_model_produced,
    score_thrashing,
)


@pytest.fixture
def run_dir(tmp_path):
    """Create a temporary run directory."""
    return tmp_path


def _write_model_py(run_dir: Path, code: str):
    """Helper: write model.py to run_dir."""
    (run_dir / "model.py").write_text(code)


def _create_idata(run_dir: Path, n_chains=4, n_draws=1000, n_divergent=0,
                   has_pp=False, has_ll=False, rhat_noise=0.0):
    """Helper: create a synthetic InferenceData and save to results.nc."""
    import arviz as az
    import xarray as xr

    rng = np.random.default_rng(42)

    # Posterior
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

    # Sample stats with divergences
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


class TestScoreModelProduced:
    def test_no_model_py(self, run_dir):
        score, details = score_model_produced(run_dir)
        assert score == 0

    def test_model_py_no_nc(self, run_dir):
        _write_model_py(run_dir, "import pymc as pm\n" * 10)
        score, details = score_model_produced(run_dir)
        assert score == 1

    def test_nc_with_posterior(self, run_dir):
        _write_model_py(run_dir, "import pymc as pm\n" * 10)
        _create_idata(run_dir)
        score, details = score_model_produced(run_dir)
        assert score >= 3

    def test_nc_with_extras(self, run_dir):
        _write_model_py(run_dir, "import pymc as pm\n" * 10)
        _create_idata(run_dir, has_pp=True, has_ll=True)
        score, details = score_model_produced(run_dir)
        assert score == 5


class TestScoreConvergence:
    def test_no_nc(self, run_dir):
        score, details = score_convergence(run_dir)
        assert score == 0

    def test_good_convergence(self, run_dir):
        _create_idata(run_dir, n_chains=4, n_draws=1000, n_divergent=0)
        score, details = score_convergence(run_dir)
        assert score >= 4

    def test_some_divergences(self, run_dir):
        _create_idata(run_dir, n_chains=4, n_draws=1000, n_divergent=50)
        score, details = score_convergence(run_dir)
        assert 2 <= score <= 3

    def test_many_divergences(self, run_dir):
        _create_idata(run_dir, n_chains=4, n_draws=1000, n_divergent=500)
        score, details = score_convergence(run_dir)
        assert score <= 2


class TestScoreBestPractices:
    def test_no_model_py(self, run_dir):
        score, details = score_best_practices(run_dir, "T1_hierarchical")
        assert score == 0

    def test_all_patterns(self, run_dir):
        code = """
import pymc as pm
import nutpie

with pm.Model(coords={"school": range(8)}) as model:
    mu = pm.Normal("mu", dims="school")
    idata = pm.sample(nuts_sampler="nutpie", random_seed=42)
"""
        _write_model_py(run_dir, code)
        score, details = score_best_practices(run_dir, "T1_hierarchical")
        assert score >= 3

    def test_no_patterns(self, run_dir):
        _write_model_py(run_dir, "import pymc as pm\nwith pm.Model():\n    pass")
        score, details = score_best_practices(run_dir, "T1_hierarchical")
        assert score == 0


class TestScoreAppropriatenessRegex:
    def test_no_model(self):
        score, details = _score_appropriateness_regex("print('hello')", "T1_hierarchical", {})
        assert score == 0

    def test_basic_model(self):
        code = "with pm.Model() as model:\n    pass"
        score, details = _score_appropriateness_regex(code, "T1_hierarchical", {})
        assert score == 2

    def test_hierarchical_patterns(self):
        code = """
with pm.Model() as model:
    mu_mu = pm.Normal("mu_mu")
    offset = pm.Normal("offset")
    x = pm.Deterministic("x", mu_mu + offset)
"""
        score, details = _score_appropriateness_regex(code, "T1_hierarchical", {})
        assert score >= 4


def _write_turns_jsonl(run_dir, turns):
    """Helper: write turns.jsonl with assistant messages."""
    path = run_dir / "turns.jsonl"
    with open(path, "w") as f:
        for turn in turns:
            f.write(json.dumps(turn) + "\n")


def _write_metadata(run_dir, num_turns=10, success=True):
    """Helper: write metadata.json."""
    (run_dir / "metadata.json").write_text(json.dumps({
        "task_id": "T1_hierarchical",
        "condition": "no_skill",
        "rep": 0,
        "success": success,
        "num_turns": num_turns,
    }))


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


class TestScoreThrashing:
    def test_no_turns_file(self, run_dir):
        """No turns.jsonl — fallback score from metadata."""
        _write_metadata(run_dir, num_turns=10)
        score, details = score_thrashing(run_dir)
        assert details["method"] == "fallback_num_turns"
        assert score >= 1

    def test_no_metadata(self, run_dir):
        """No metadata at all — score 0."""
        score, details = score_thrashing(run_dir)
        assert score == 0

    def test_clean_run(self, run_dir):
        """Single write + single bash run → score 5."""
        _write_metadata(run_dir, num_turns=4)
        turns = [
            _make_assistant_turn("Write", {"file_path": "/tmp/work/model.py"}),
            _make_assistant_turn("Bash", {"command": "python model.py"}),
        ]
        _write_turns_jsonl(run_dir, turns)
        score, details = score_thrashing(run_dir)
        assert score == 5
        assert details["rewrites"] == 0

    def test_one_rewrite(self, run_dir):
        """Two writes + two bash runs → score 4."""
        _write_metadata(run_dir, num_turns=8)
        turns = [
            _make_assistant_turn("Write", {"file_path": "/tmp/work/model.py"}),
            _make_assistant_turn("Bash", {"command": "python model.py"}),
            _make_assistant_turn("Write", {"file_path": "/tmp/work/model.py"}),
            _make_assistant_turn("Bash", {"command": "python model.py"}),
        ]
        _write_turns_jsonl(run_dir, turns)
        score, details = score_thrashing(run_dir)
        assert score == 4
        assert details["rewrites"] == 1

    def test_many_rewrites(self, run_dir):
        """5+ writes → score 1."""
        _write_metadata(run_dir, num_turns=20)
        turns = []
        for _ in range(6):
            turns.append(_make_assistant_turn("Write", {"file_path": "/tmp/work/model.py"}))
            turns.append(_make_assistant_turn("Bash", {"command": "python model.py"}))
        _write_turns_jsonl(run_dir, turns)
        score, details = score_thrashing(run_dir)
        assert score == 1
        assert details["rewrites"] == 5

    def test_no_model_writes(self, run_dir):
        """Turns exist but no model.py writes → score 0."""
        _write_metadata(run_dir, num_turns=5)
        turns = [
            _make_assistant_turn("Read", {"file_path": "/tmp/work/data.csv"}),
            _make_assistant_turn("Bash", {"command": "ls -la"}),
        ]
        _write_turns_jsonl(run_dir, turns)
        score, details = score_thrashing(run_dir)
        assert score == 0

    def test_fallback_fast_run(self, run_dir):
        """Fallback: 5 turns → estimated 0 rewrites → score 5."""
        _write_metadata(run_dir, num_turns=5)
        score, details = score_thrashing(run_dir)
        assert score == 5
        assert details["method"] == "fallback_num_turns"

    def test_fallback_slow_run(self, run_dir):
        """Fallback: 30 turns → several estimated rewrites → low score."""
        _write_metadata(run_dir, num_turns=30)
        score, details = score_thrashing(run_dir)
        assert score <= 2
        assert details["method"] == "fallback_num_turns"


class TestScoreEfficiency:
    def test_fast_run(self, run_dir):
        """6 turns → score 5."""
        _write_metadata(run_dir, num_turns=6)
        score, details = score_efficiency(run_dir)
        assert score == 5

    def test_moderate_run(self, run_dir):
        """15 turns → score 3."""
        _write_metadata(run_dir, num_turns=15)
        score, details = score_efficiency(run_dir)
        assert score == 3

    def test_slow_run(self, run_dir):
        """30 turns → score 1."""
        _write_metadata(run_dir, num_turns=30)
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

    def test_very_fast_run(self, run_dir):
        """1 turn → score 5."""
        _write_metadata(run_dir, num_turns=1)
        score, details = score_efficiency(run_dir)
        assert score == 5

    def test_very_slow_run(self, run_dir):
        """40 turns → score 0."""
        _write_metadata(run_dir, num_turns=40)
        score, details = score_efficiency(run_dir)
        assert score == 0


def _create_idata_with_values(run_dir: Path, posterior_data: dict,
                               n_chains=4, n_draws=1000, n_divergent=0):
    """Helper: create idata with specific posterior values."""
    import arviz as az
    import xarray as xr

    data_vars = {}
    for name, values in posterior_data.items():
        data_vars[name] = (["chain", "draw"], values)

    posterior = xr.Dataset(
        data_vars,
        coords={"chain": range(n_chains), "draw": range(n_draws)},
    )

    diverging = np.zeros((n_chains, n_draws), dtype=bool)
    if n_divergent > 0:
        rng = np.random.default_rng(42)
        flat = diverging.ravel()
        indices = rng.choice(len(flat), min(n_divergent, len(flat)), replace=False)
        flat[indices] = True
        diverging = flat.reshape(n_chains, n_draws)

    sample_stats = xr.Dataset(
        {"diverging": (["chain", "draw"], diverging)},
        coords={"chain": range(n_chains), "draw": range(n_draws)},
    )

    idata = az.InferenceData(posterior=posterior, sample_stats=sample_stats)
    idata.to_netcdf(str(run_dir / "results.nc"))
    return idata


class TestEvaluatePassFail:
    def test_full_pass(self, run_dir):
        """Good idata, high scores → passed=True."""
        _create_idata(run_dir, n_chains=4, n_draws=1000, n_divergent=0)
        passed, details = evaluate_pass_fail(run_dir, model_produced_score=5, convergence_score=5)
        assert passed is True
        assert details["sampling_completed"] is True
        assert details["convergence_acceptable"] is True
        assert details["non_degenerate"] is True

    def test_fail_no_sampling(self, run_dir):
        """model_produced=1 → fail on sampling check."""
        passed, details = evaluate_pass_fail(run_dir, model_produced_score=1, convergence_score=0)
        assert passed is False
        assert details["sampling_completed"] is False

    def test_fail_convergence(self, run_dir):
        """Good idata but convergence_score=2 → fail on convergence check."""
        _create_idata(run_dir, n_chains=4, n_draws=1000)
        passed, details = evaluate_pass_fail(run_dir, model_produced_score=5, convergence_score=2)
        assert passed is False
        assert details["convergence_acceptable"] is False

    def test_fail_nan_estimates(self, run_dir):
        """Posterior with NaN means → fail on non-degenerate check."""
        rng = np.random.default_rng(42)
        mu = rng.normal(0, 1, (4, 1000))
        mu[0, 0] = np.nan  # inject NaN
        sigma = rng.normal(1, 0.1, (4, 1000))
        _create_idata_with_values(run_dir, {"mu": mu, "sigma": sigma})
        passed, details = evaluate_pass_fail(run_dir, model_produced_score=4, convergence_score=3)
        assert passed is False
        assert details["non_degenerate"] is False
        assert "non-finite" in details["reason"]

    def test_fail_degenerate(self, run_dir):
        """All vars have zero std (constant values) → fail on degenerate check."""
        constant = np.full((4, 1000), 5.0)
        _create_idata_with_values(run_dir, {"mu": constant, "sigma": constant})
        passed, details = evaluate_pass_fail(run_dir, model_produced_score=4, convergence_score=3)
        assert passed is False
        assert details["non_degenerate"] is False
        assert "degenerate" in details["reason"]

    def test_pass_marginal(self, run_dir):
        """Boundary: model_produced=4, convergence=3, valid idata → pass."""
        _create_idata(run_dir, n_chains=4, n_draws=1000, n_divergent=0)
        passed, details = evaluate_pass_fail(run_dir, model_produced_score=4, convergence_score=3)
        assert passed is True


class TestCountRetries:
    def test_clean_run(self, run_dir):
        """1 Write to model.py → retries=0."""
        turns = [
            _make_assistant_turn("Write", {"file_path": "/tmp/work/model.py"}),
            _make_assistant_turn("Bash", {"command": "python model.py"}),
        ]
        _write_turns_jsonl(run_dir, turns)
        retries, details = count_retries(run_dir)
        assert retries == 0
        assert details["method"] == "turns_analysis"

    def test_two_rewrites(self, run_dir):
        """3 Writes to model.py → retries=2."""
        turns = [
            _make_assistant_turn("Write", {"file_path": "/tmp/work/model.py"}),
            _make_assistant_turn("Bash", {"command": "python model.py"}),
            _make_assistant_turn("Write", {"file_path": "/tmp/work/model.py"}),
            _make_assistant_turn("Bash", {"command": "python model.py"}),
            _make_assistant_turn("Write", {"file_path": "/tmp/work/model.py"}),
            _make_assistant_turn("Bash", {"command": "python model.py"}),
        ]
        _write_turns_jsonl(run_dir, turns)
        retries, details = count_retries(run_dir)
        assert retries == 2
        assert details["method"] == "turns_analysis"

    def test_fallback_from_metadata(self, run_dir):
        """No turns.jsonl, metadata num_turns=20 → retries=2 (heuristic)."""
        _write_metadata(run_dir, num_turns=20)
        retries, details = count_retries(run_dir)
        assert retries == 2
        assert details["method"] == "fallback_num_turns"

    def test_no_data(self, run_dir):
        """No turns.jsonl, no metadata → retries=0."""
        retries, details = count_retries(run_dir)
        assert retries == 0
        assert details["method"] == "none"
