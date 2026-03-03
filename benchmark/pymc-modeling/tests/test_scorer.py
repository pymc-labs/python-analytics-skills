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
    score_model_produced,
    score_parameter_recovery,
    score_workflow,
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


class TestScoreWorkflow:
    def test_no_model_py(self, run_dir):
        score, details = score_workflow(run_dir)
        assert score == 0

    def test_full_workflow(self, run_dir):
        code = """
import pymc as pm
import arviz as az

with pm.Model() as model:
    mu = pm.Normal("mu", 0, 1)
    y = pm.Normal("y", mu, 1, observed=[1, 2, 3])
    prior_pred = pm.sample_prior_predictive(draws=100)
    idata = pm.sample(1000)
    pm.sample_posterior_predictive(idata, extend_inferencedata=True)
    pm.compute_log_likelihood(idata)

idata.to_netcdf("results.nc")

summary = az.summary(idata)
print(summary)
n_div = idata.sample_stats["diverging"].sum()
print(f"Divergences: {n_div}")
"""
        _write_model_py(run_dir, code)
        score, details = score_workflow(run_dir)
        assert score == 5
        assert details["prior_predictive"] is True
        assert details["diagnostics"]["sufficient"] is True
        assert details["posterior_predictive"] is True
        assert details["model_comparison"]["sufficient"] is True
        assert details["early_save"]["is_early"] is True

    def test_minimal_workflow(self, run_dir):
        code = """
import pymc as pm
with pm.Model() as model:
    mu = pm.Normal("mu", 0, 1)
    idata = pm.sample(1000)
idata.to_netcdf("results.nc")
"""
        _write_model_py(run_dir, code)
        score, details = score_workflow(run_dir)
        # Only early save, no workflow steps
        assert score <= 1

    def test_diagnostics_only(self, run_dir):
        code = """
import pymc as pm
import arviz as az

with pm.Model() as model:
    mu = pm.Normal("mu", 0, 1)
    y = pm.Normal("y", mu, 1, observed=[1, 2])
    idata = pm.sample(1000)

idata.to_netcdf("results.nc")

summary = az.summary(idata)
n_div = idata.sample_stats["diverging"].sum()
r_hat = az.rhat(idata)
print("Done")
"""
        _write_model_py(run_dir, code)
        score, details = score_workflow(run_dir)
        # diagnostics + early save = 2
        assert score == 2
        assert details["diagnostics"]["sufficient"] is True
        assert details["early_save"]["is_early"] is True

    def test_late_save_not_rewarded(self, run_dir):
        """Save at very end of file doesn't count as early save."""
        code = """
import pymc as pm
with pm.Model() as model:
    mu = pm.Normal("mu", 0, 1)
    idata = pm.sample(1000)
print("lots of analysis here")
print("more analysis")
print("still more analysis")
print("even more analysis")
print("yet more analysis")
idata.to_netcdf("results.nc")
"""
        _write_model_py(run_dir, code)
        score, details = score_workflow(run_dir)
        assert details["early_save"]["is_early"] is False


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


class TestScoreParameterRecovery:
    def test_no_results_nc(self, run_dir):
        """No results.nc → score 0."""
        score, details = score_parameter_recovery(run_dir, "T1_hierarchical")
        assert score == 0

    def test_t4_good_recovery(self, run_dir):
        """T4: component means near [-5, 0, 5] → high score."""
        import arviz as az
        import xarray as xr

        rng = np.random.default_rng(42)
        # 3 component means (chain x draw x component)
        n_chains, n_draws = 4, 1000
        mu = np.stack([
            rng.normal([-5.0, 0.0, 5.0], 0.3, (n_draws, 3))
            for _ in range(n_chains)
        ])
        w = rng.dirichlet([10, 10, 10], (n_chains, n_draws))

        posterior = xr.Dataset(
            {
                "mu": (["chain", "draw", "component"], mu),
                "w": (["chain", "draw", "component"], w),
            },
            coords={
                "chain": range(n_chains),
                "draw": range(n_draws),
                "component": range(3),
            },
        )
        sample_stats = xr.Dataset(
            {"diverging": (["chain", "draw"],
                           np.zeros((n_chains, n_draws), dtype=bool))},
            coords={"chain": range(n_chains), "draw": range(n_draws)},
        )
        idata = az.InferenceData(posterior=posterior, sample_stats=sample_stats)
        idata.to_netcdf(str(run_dir / "results.nc"))

        score, details = score_parameter_recovery(run_dir, "T4_mixture")
        assert score >= 3
        assert details.get("centers_recovered", 0) >= 2

    def test_t5_shrinkage(self, run_dir):
        """T5: some coefficients near zero, some not → good shrinkage score."""
        rng = np.random.default_rng(42)
        n_chains, n_draws = 4, 1000
        # 11 predictors: 3 important, 8 shrunk to zero
        betas = np.zeros((n_chains, n_draws, 11))
        for c in range(n_chains):
            betas[c, :, 0] = rng.normal(0.5, 0.1, n_draws)   # important
            betas[c, :, 1] = rng.normal(-0.3, 0.08, n_draws)  # important
            betas[c, :, 2] = rng.normal(0.2, 0.05, n_draws)   # important
            for j in range(3, 11):
                betas[c, :, j] = rng.normal(0.0, 0.02, n_draws)  # shrunk

        import arviz as az
        import xarray as xr
        posterior = xr.Dataset(
            {"beta": (["chain", "draw", "predictor"], betas)},
            coords={
                "chain": range(n_chains),
                "draw": range(n_draws),
                "predictor": range(11),
            },
        )
        sample_stats = xr.Dataset(
            {"diverging": (["chain", "draw"],
                           np.zeros((n_chains, n_draws), dtype=bool))},
            coords={"chain": range(n_chains), "draw": range(n_draws)},
        )
        idata = az.InferenceData(posterior=posterior, sample_stats=sample_stats)
        idata.to_netcdf(str(run_dir / "results.nc"))

        score, details = score_parameter_recovery(run_dir, "T5_horseshoe")
        assert score >= 3
        assert details.get("shrinkage_pattern") == "good"

    def test_t4_false_positive_weight_names(self, run_dir):
        """T4: variables like 'mu_raw', 'spike' must NOT be treated as weights."""
        import arviz as az
        import xarray as xr

        rng = np.random.default_rng(42)
        n_chains, n_draws = 4, 500
        # Component means near [-5, 0, 5]
        mu = np.stack([
            rng.normal([-5.0, 0.0, 5.0], 0.3, (n_draws, 3))
            for _ in range(n_chains)
        ])
        # Variables whose names contain "w" or "pi" as substrings
        # but are NOT mixture weights
        mu_raw = rng.normal(0, 1, (n_chains, n_draws, 3))
        sigma_raw = rng.normal(0, 1, (n_chains, n_draws, 3))
        spike = rng.normal(0, 1, (n_chains, n_draws, 3))
        # Actual weights under an unambiguous name
        weights = rng.dirichlet([10, 10, 10], (n_chains, n_draws))

        posterior = xr.Dataset(
            {
                "mu": (["chain", "draw", "component"], mu),
                "mu_raw": (["chain", "draw", "component"], mu_raw),
                "sigma_raw": (["chain", "draw", "component"], sigma_raw),
                "spike": (["chain", "draw", "component"], spike),
                "weights": (["chain", "draw", "component"], weights),
            },
            coords={
                "chain": range(n_chains),
                "draw": range(n_draws),
                "component": range(3),
            },
        )
        sample_stats = xr.Dataset(
            {"diverging": (["chain", "draw"],
                           np.zeros((n_chains, n_draws), dtype=bool))},
            coords={"chain": range(n_chains), "draw": range(n_draws)},
        )
        idata = az.InferenceData(posterior=posterior, sample_stats=sample_stats)
        idata.to_netcdf(str(run_dir / "results.nc"))

        score, details = score_parameter_recovery(run_dir, "T4_mixture")
        # "weights" should be found, but "mu_raw", "sigma_raw", "spike" should not
        assert details.get("weights_valid") is True
        assert score >= 3

    def test_t4_no_false_weight_without_real_weight(self, run_dir):
        """T4: 'mu_raw' alone must NOT be mistaken for a weight variable."""
        import arviz as az
        import xarray as xr

        rng = np.random.default_rng(42)
        n_chains, n_draws = 4, 500
        mu = np.stack([
            rng.normal([-5.0, 0.0, 5.0], 0.3, (n_draws, 3))
            for _ in range(n_chains)
        ])
        # Only false-positive names, no real weight variable
        mu_raw = rng.normal(0, 50, (n_chains, n_draws, 3))  # not valid weights

        posterior = xr.Dataset(
            {
                "mu": (["chain", "draw", "component"], mu),
                "mu_raw": (["chain", "draw", "component"], mu_raw),
            },
            coords={
                "chain": range(n_chains),
                "draw": range(n_draws),
                "component": range(3),
            },
        )
        sample_stats = xr.Dataset(
            {"diverging": (["chain", "draw"],
                           np.zeros((n_chains, n_draws), dtype=bool))},
            coords={"chain": range(n_chains), "draw": range(n_draws)},
        )
        idata = az.InferenceData(posterior=posterior, sample_stats=sample_stats)
        idata.to_netcdf(str(run_dir / "results.nc"))

        score, details = score_parameter_recovery(run_dir, "T4_mixture")
        # mu_raw should NOT be matched as a weight variable
        assert "weights_valid" not in details

    def test_t3_false_positive_volatility_names(self, run_dir):
        """T3: variables like 'theta', 'alpha' must NOT be treated as volatility."""
        import arviz as az
        import xarray as xr

        rng = np.random.default_rng(42)
        n_chains, n_draws, n_time = 4, 500, 100
        # Variables with "h" as a substring but NOT volatility
        theta = rng.normal(0, 1, (n_chains, n_draws))
        alpha = rng.normal(0, 1, (n_chains, n_draws))
        # Actual volatility process
        volatility = rng.normal(0, 0.5, (n_chains, n_draws, n_time))
        nu = rng.exponential(10, (n_chains, n_draws)) + 2
        sigma = np.abs(rng.normal(0.1, 0.02, (n_chains, n_draws)))

        posterior = xr.Dataset(
            {
                "theta": (["chain", "draw"], theta),
                "alpha": (["chain", "draw"], alpha),
                "volatility": (["chain", "draw", "time"], volatility),
                "nu": (["chain", "draw"], nu),
                "sigma": (["chain", "draw"], sigma),
            },
            coords={
                "chain": range(n_chains),
                "draw": range(n_draws),
                "time": range(n_time),
            },
        )
        sample_stats = xr.Dataset(
            {"diverging": (["chain", "draw"],
                           np.zeros((n_chains, n_draws), dtype=bool))},
            coords={"chain": range(n_chains), "draw": range(n_draws)},
        )
        idata = az.InferenceData(posterior=posterior, sample_stats=sample_stats)
        idata.to_netcdf(str(run_dir / "results.nc"))

        score, details = score_parameter_recovery(run_dir, "T3_stochastic_volatility")
        # "volatility" should be found; "theta" and "alpha" should NOT
        vol_vars = details.get("volatility_vars", [])
        assert "volatility" in vol_vars
        assert "theta" not in vol_vars
        assert "alpha" not in vol_vars
        assert score >= 3

    def test_t3_h_exact_name_matches(self, run_dir):
        """T3: bare 'h' and 'h_t' should match as volatility variables."""
        import arviz as az
        import xarray as xr

        rng = np.random.default_rng(42)
        n_chains, n_draws, n_time = 4, 500, 100
        h = rng.normal(0, 0.5, (n_chains, n_draws, n_time))
        nu = rng.exponential(10, (n_chains, n_draws)) + 2
        sigma = np.abs(rng.normal(0.1, 0.02, (n_chains, n_draws)))

        posterior = xr.Dataset(
            {
                "h": (["chain", "draw", "time"], h),
                "nu": (["chain", "draw"], nu),
                "sigma": (["chain", "draw"], sigma),
            },
            coords={
                "chain": range(n_chains),
                "draw": range(n_draws),
                "time": range(n_time),
            },
        )
        sample_stats = xr.Dataset(
            {"diverging": (["chain", "draw"],
                           np.zeros((n_chains, n_draws), dtype=bool))},
            coords={"chain": range(n_chains), "draw": range(n_draws)},
        )
        idata = az.InferenceData(posterior=posterior, sample_stats=sample_stats)
        idata.to_netcdf(str(run_dir / "results.nc"))

        score, details = score_parameter_recovery(run_dir, "T3_stochastic_volatility")
        assert "h" in details.get("volatility_vars", [])
        assert score >= 3
