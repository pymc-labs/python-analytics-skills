"""Tests for the benchmark scorer."""

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from src.scorer import (
    _score_appropriateness_regex,
    score_best_practices,
    score_convergence,
    score_model_produced,
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
