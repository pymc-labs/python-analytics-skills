"""Benchmark scorer — 6-criterion rubric (30 max).

Criteria:
1. Model produced (0-5): automated, checks results.nc
2. Convergence (0-5): automated, reads r_hat/ESS/divergences from .nc
3. Model appropriateness (0-5): LLM judge (Haiku) with task-specific rubric
4. Best practices (0-5): regex on model.py for coords/dims, nutpie, etc.
5. Workflow (0-5): evidence of Bayesian workflow steps in model.py
6. Parameter recovery (0-5): posterior covers known ground truth or plausible ranges
"""

import json
import logging
import os
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import arviz as az
import numpy as np

from src.runner import (
    DEFAULT_TIMEOUT,
    RESULTS_DIR,
    RUNS_DIR,
    load_tasks,
)

logger = logging.getLogger(__name__)

# LLM judge settings
JUDGE_MODEL = "haiku"
JUDGE_BUDGET = "0.50"
JUDGE_TIMEOUT = 60


@dataclass
class ScoreResult:
    """Scores for a single benchmark run."""
    task_id: str
    condition: str
    rep: int
    model_produced: int = 0
    convergence: int = 0
    model_appropriateness: int = 0
    best_practices: int = 0
    workflow: int = 0
    parameter_recovery: int = 0
    total: int = 0
    passed: bool = False
    retries: int = 0
    details: dict = field(default_factory=dict)

    def compute_total(self):
        self.total = (
            self.model_produced
            + self.convergence
            + self.model_appropriateness
            + self.best_practices
            + self.workflow
            + self.parameter_recovery
        )


def score_model_produced(run_dir: Path, idata=None) -> tuple[int, dict]:
    """Score criterion 1: Model produced (0-5).

    0 = no model.py or results.nc
    1 = model.py exists but no results.nc
    2 = results.nc exists but empty/invalid
    3 = results.nc has posterior group
    4 = posterior has reasonable shape (>100 draws)
    5 = posterior + posterior_predictive or log_likelihood present
    """
    details = {}
    model_py = run_dir / "model.py"
    nc_path = run_dir / "results.nc"

    if not model_py.exists():
        details["reason"] = "no model.py"
        return 0, details

    details["model_py_size"] = model_py.stat().st_size

    if not nc_path.exists():
        details["reason"] = "no results.nc"
        return 1, details

    try:
        if idata is None:
            idata = az.from_netcdf(str(nc_path))
    except Exception as e:
        details["reason"] = f"results.nc load error: {e}"
        return 2, details

    if not hasattr(idata, "posterior") or idata.posterior is None:
        details["reason"] = "no posterior group"
        return 2, details

    details["groups"] = list(idata.groups())

    # Check draw count
    n_draws = idata.posterior.sizes.get("draw", 0)
    n_chains = idata.posterior.sizes.get("chain", 0)
    details["n_draws"] = n_draws
    details["n_chains"] = n_chains

    if n_draws < 100:
        details["reason"] = f"only {n_draws} draws"
        return 3, details

    # Check for extra groups
    has_pp = hasattr(idata, "posterior_predictive")
    has_ll = hasattr(idata, "log_likelihood")
    details["has_posterior_predictive"] = has_pp
    details["has_log_likelihood"] = has_ll

    if has_pp or has_ll:
        return 5, details

    return 4, details


def score_convergence(run_dir: Path, idata=None) -> tuple[int, dict]:
    """Score criterion 2: Convergence (0-5).

    0 = no results.nc or can't compute diagnostics
    1 = sampling ran but severe convergence issues (r_hat > 1.1 for many params)
    2 = r_hat mostly OK but high divergences (>100)
    3 = r_hat < 1.05 for most params, some divergences (<100)
    4 = good convergence, few divergences (<10), adequate ESS
    5 = excellent: r_hat < 1.01, zero divergences, ESS > 400
    """
    details = {}
    nc_path = run_dir / "results.nc"

    if not nc_path.exists():
        return 0, details

    try:
        if idata is None:
            idata = az.from_netcdf(str(nc_path))

        if not hasattr(idata, "posterior") or idata.posterior is None:
            details["reason"] = "no posterior group"
            return 0, details

        # Check chain count — r_hat and ESS require >= 2 chains
        n_chains = idata.posterior.sizes.get("chain", 0)
        n_draws = idata.posterior.sizes.get("draw", 0)
        details["n_chains"] = n_chains
        details["n_draws"] = n_draws

        if n_chains < 2:
            # Can't compute r_hat/ESS with 1 chain. Score based on
            # divergences only; cap at 3 (can't confirm convergence).
            details["reason"] = f"only {n_chains} chain — r_hat/ESS unavailable"
            rhat_values = np.array([])
            ess_values = np.array([])
            pct_above_1_05 = 0.0
            pct_above_1_1 = 0.0
        else:
            # Compute r_hat
            try:
                rhat = az.rhat(idata)
                rhat_values = []
                for var in rhat.data_vars:
                    vals = rhat[var].values.flatten()
                    rhat_values.extend(vals[np.isfinite(vals)])
                rhat_values = np.array(rhat_values)
                details["rhat_max"] = float(np.max(rhat_values)) if len(rhat_values) > 0 else None
                details["rhat_mean"] = float(np.mean(rhat_values)) if len(rhat_values) > 0 else None
                pct_above_1_05 = float(np.mean(rhat_values > 1.05)) if len(rhat_values) > 0 else 1.0
                pct_above_1_1 = float(np.mean(rhat_values > 1.1)) if len(rhat_values) > 0 else 1.0
            except Exception:
                rhat_values = np.array([])
                pct_above_1_05 = 1.0
                pct_above_1_1 = 1.0

            # Compute ESS
            try:
                ess = az.ess(idata)
                ess_values = []
                for var in ess.data_vars:
                    vals = ess[var].values.flatten()
                    ess_values.extend(vals[np.isfinite(vals)])
                ess_values = np.array(ess_values)
                details["ess_min"] = float(np.min(ess_values)) if len(ess_values) > 0 else 0
                details["ess_median"] = float(np.median(ess_values)) if len(ess_values) > 0 else 0
            except Exception:
                ess_values = np.array([])

        # Count divergences
        n_divergent = 0
        if hasattr(idata, "sample_stats"):
            try:
                div = idata.sample_stats.get("diverging")
                if div is not None:
                    n_divergent = int(div.values.sum())
            except Exception:
                pass
        details["n_divergent"] = n_divergent

        # Score
        if pct_above_1_1 > 0.5:
            return 1, details
        if n_divergent > 100 or pct_above_1_05 > 0.3:
            return 2, details
        if n_divergent > 10 or pct_above_1_05 > 0.1:
            return 3, details

        # Single-chain runs: can't verify convergence, cap at 3
        if n_chains < 2:
            return min(3, 3 if n_divergent < 10 else 2), details

        min_ess = float(np.min(ess_values)) if len(ess_values) > 0 else 0
        max_rhat = float(np.max(rhat_values)) if len(rhat_values) > 0 else 999

        if n_divergent == 0 and max_rhat < 1.01 and min_ess > 400:
            return 5, details
        if n_divergent < 10 and max_rhat < 1.05 and min_ess > 100:
            return 4, details

        return 3, details

    except Exception as e:
        details["error"] = str(e)
        return 0, details


def score_best_practices(run_dir: Path, task_id: str) -> tuple[int, dict]:
    """Score criterion 4: Best practices (0-5).

    Checks model.py for patterns defined in tasks.yaml.
    Each matching pattern = 1 point, capped at 5.
    """
    details = {}
    model_py = run_dir / "model.py"

    if not model_py.exists():
        details["reason"] = "no model.py"
        return 0, details

    code = model_py.read_text()
    details["code_lines"] = len(code.splitlines())

    config = load_tasks()
    patterns = config["tasks"][task_id].get("best_practices_patterns", [])
    matches = {}
    score = 0

    for pattern in patterns:
        found = bool(re.search(pattern, code))
        matches[pattern] = found
        if found:
            score += 1

    details["pattern_matches"] = matches
    return min(score, 5), details


def _extract_judge_json(response: str) -> dict | None:
    """Extract {"score": N, "reasoning": "..."} from LLM judge response.

    Tries multiple strategies since Haiku may wrap the JSON in markdown
    fences, add extra text, or include newlines/quotes in reasoning.
    """
    # Strategy 1: parse the whole response as JSON
    try:
        data = json.loads(response)
        if "score" in data:
            return data
    except (json.JSONDecodeError, TypeError):
        pass

    # Strategy 2: find a JSON block inside markdown fences
    fenced = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response, re.DOTALL)
    if fenced:
        try:
            data = json.loads(fenced.group(1))
            if "score" in data:
                return data
        except (json.JSONDecodeError, TypeError):
            pass

    # Strategy 3: find the outermost { ... } allowing newlines
    brace = re.search(r'\{.*\}', response, re.DOTALL)
    if brace:
        try:
            data = json.loads(brace.group())
            if "score" in data:
                return data
        except (json.JSONDecodeError, TypeError):
            pass

    # Strategy 4: just extract the score number
    score_match = re.search(r'"score"\s*:\s*(\d)', response)
    if score_match:
        return {"score": int(score_match.group(1)), "reasoning": "extracted from partial JSON"}

    return None


def score_model_appropriateness_llm(
    run_dir: Path, task_id: str
) -> tuple[int, dict]:
    """Score criterion 3: Model appropriateness via LLM judge (0-5).

    Uses Haiku to score the model code against a task-specific rubric.
    Falls back to regex-based scoring if LLM fails.
    """
    details = {}
    model_py = run_dir / "model.py"

    if not model_py.exists():
        details["reason"] = "no model.py"
        return 0, details

    code = model_py.read_text()

    config = load_tasks()
    task = config["tasks"][task_id]
    rubric = task["judge_rubric"]
    task_name = task["name"]
    task_prompt = task["prompt"]

    judge_prompt = f"""Score this PyMC code for model appropriateness on a 0-5 scale.

Task: {task_name}
Task description: {task_prompt}

Rubric:
{rubric}

Score 0 if no code or code doesn't define a PyMC model.

Respond with ONLY a JSON object: {{"score": N, "reasoning": "brief explanation"}}

Code:
```python
{code}
```"""

    try:
        # Strip CLAUDECODE to allow nested claude --print calls.
        env = os.environ.copy()
        env.pop("CLAUDECODE", None)

        proc = subprocess.run(
            [
                "claude",
                "--print",
                "--model", JUDGE_MODEL,
                "--disable-slash-commands",
                "--no-session-persistence",
                "--max-budget-usd", JUDGE_BUDGET,
                "--output-format", "text",
            ],
            input=judge_prompt,
            capture_output=True,
            text=True,
            timeout=JUDGE_TIMEOUT,
            env=env,
        )

        response = proc.stdout.strip()
        details["llm_response"] = response[:1000]

        # Extract JSON from response — try progressively looser methods
        parsed_judge = _extract_judge_json(response)
        if parsed_judge is not None:
            score = int(parsed_judge.get("score", 0))
            score = max(0, min(5, score))
            details["reasoning"] = parsed_judge.get("reasoning", "")
            details["method"] = "llm"
            return score, details

        logger.warning(f"LLM judge returned non-JSON: {response[:200]}")

    except (subprocess.TimeoutExpired, Exception) as e:
        logger.warning(f"LLM judge failed: {e}, falling back to regex")
        details["llm_error"] = str(e)

    # Fallback: regex-based scoring
    return _score_appropriateness_regex(code, task_id, details)


def _score_appropriateness_regex(
    code: str, task_id: str, details: dict
) -> tuple[int, dict]:
    """Fallback regex scoring for model appropriateness."""
    details["method"] = "regex_fallback"
    score = 0

    # Base: any PyMC model
    if re.search(r'pm\.Model|pymc\.Model', code):
        score = 2

    # Task-specific patterns
    task_patterns = {
        "T1_hierarchical": [
            (r'(offset|noncentered|non_centered|mu\s*\+\s*sigma\s*\*)', "non-centered"),
            (r'Hyperprior|mu_mu|tau|sigma_group', "hyperpriors"),
            (r'Deterministic', "deterministic"),
        ],
        "T2_ordinal": [
            (r'OrderedLogistic|OrderedProbit', "ordinal_likelihood"),
            (r'ordered|Ordered', "ordered_transform"),
            (r'cutpoint|threshold', "cutpoints"),
        ],
        "T3_stochastic_volatility": [
            (r'GaussianRandomWalk', "gaussian_rw"),
            (r'StudentT|Student', "student_t"),
            (r'exp\s*\(', "exp_transform"),
        ],
        "T4_mixture": [
            (r'NormalMixture|Mixture', "mixture_dist"),
            (r'Dirichlet', "dirichlet"),
            (r'ordered|univariate_ordered', "ordered_transform"),
        ],
        "T5_horseshoe": [
            (r'horseshoe|Horseshoe', "horseshoe"),
            (r'target_accept|nuts_sampler_kwargs', "sampler_tuning"),
            (r'R2D2|regularized', "regularized"),
        ],
    }

    for pattern, name in task_patterns.get(task_id, []):
        if re.search(pattern, code, re.IGNORECASE):
            score += 1
            details[f"regex_{name}"] = True

    return min(score, 5), details


def score_workflow(run_dir: Path) -> tuple[int, dict]:
    """Score criterion 5: Bayesian workflow evidence (0-5).

    Checks model.py for evidence of robust Bayesian workflow practices.
    Each practice detected earns 1 point. No penalty for iteration count.

    +1 = prior predictive check (sample_prior_predictive)
    +1 = convergence diagnostics examined (az.summary, az.rhat, diverging check)
    +1 = posterior predictive check (sample_posterior_predictive)
    +1 = model comparison or sensitivity analysis (az.compare, az.loo, or 2+ models)
    +1 = early save (to_netcdf before post-processing / final print statements)
    """
    details: dict = {}
    model_py = run_dir / "model.py"

    if not model_py.exists():
        details["reason"] = "no model.py"
        return 0, details

    code = model_py.read_text()
    score = 0

    # 1. Prior predictive check
    has_prior_pred = bool(re.search(r'sample_prior_predictive', code))
    details["prior_predictive"] = has_prior_pred
    if has_prior_pred:
        score += 1

    # 2. Convergence diagnostics examined
    diag_patterns = [
        r'az\.summary|arviz\.summary',
        r'az\.rhat|arviz\.rhat',
        r'az\.ess|arviz\.ess',
        r'divergi',  # catches "diverging", "divergences", "divergent"
        r'r_hat',
    ]
    diag_found = [p for p in diag_patterns if re.search(p, code)]
    has_diagnostics = len(diag_found) >= 2  # need at least 2 different checks
    details["diagnostics"] = {"found": diag_found, "sufficient": has_diagnostics}
    if has_diagnostics:
        score += 1

    # 3. Posterior predictive check
    has_post_pred = bool(re.search(r'sample_posterior_predictive', code))
    details["posterior_predictive"] = has_post_pred
    if has_post_pred:
        score += 1

    # 4. Model comparison or sensitivity analysis
    comparison_patterns = [
        r'az\.compare|arviz\.compare',
        r'az\.loo\b|arviz\.loo\b',
        r'az\.waic|arviz\.waic',
        r'compute_log_likelihood',
    ]
    # Also check for multiple pm.Model blocks (sensitivity analysis)
    model_blocks = len(re.findall(r'pm\.Model\(|pymc\.Model\(', code))
    comparison_found = [p for p in comparison_patterns if re.search(p, code)]
    has_comparison = len(comparison_found) > 0 or model_blocks >= 2
    details["model_comparison"] = {
        "patterns_found": comparison_found,
        "model_blocks": model_blocks,
        "sufficient": has_comparison,
    }
    if has_comparison:
        score += 1

    # 5. Early save — to_netcdf appears before the last 20% of the file
    save_matches = list(re.finditer(r'to_netcdf|to_json', code))
    if save_matches:
        first_save_pos = save_matches[0].start()
        total_len = len(code)
        save_fraction = first_save_pos / total_len if total_len > 0 else 1.0
        early_save = save_fraction < 0.80
        details["early_save"] = {
            "save_position_frac": round(save_fraction, 2),
            "is_early": early_save,
        }
        if early_save:
            score += 1
    else:
        details["early_save"] = {"save_position_frac": None, "is_early": False}

    return min(score, 5), details


def score_parameter_recovery(run_dir: Path, task_id: str, idata=None) -> tuple[int, dict]:
    """Score criterion 6: Parameter recovery (0-5).

    For tasks with known ground truth, checks whether posterior estimates
    cover the true values. For tasks without exact ground truth, checks
    plausibility of estimates.
    """
    details: dict = {}
    nc_path = run_dir / "results.nc"

    if not nc_path.exists():
        details["reason"] = "no results.nc"
        return 0, details

    try:
        if idata is None:
            idata = az.from_netcdf(str(nc_path))
        if not hasattr(idata, "posterior") or idata.posterior is None:
            details["reason"] = "no posterior"
            return 0, details

        scorer = _RECOVERY_SCORERS.get(task_id)
        if scorer is None:
            details["reason"] = f"no recovery scorer for {task_id}"
            return 0, details

        return scorer(idata, details)

    except Exception as e:
        details["error"] = str(e)
        return 0, details


def _posterior_all_finite(posterior) -> bool:
    """Check that all posterior variable means are finite."""
    return all(
        np.all(np.isfinite(np.mean(posterior[v].values, axis=(0, 1))))
        for v in posterior.data_vars
    )


def _recovery_T1_hierarchical(idata, details: dict) -> tuple[int, dict]:
    """T1: Check school effects are in plausible range."""
    score = 0
    posterior = idata.posterior

    # Look for group-level mean (various naming conventions)
    mu_names = [v for v in posterior.data_vars
                if any(k in v.lower() for k in ["mu", "mean", "overall", "group"])]
    details["mu_candidates"] = mu_names

    if mu_names:
        mu_var = mu_names[0]
        mu_samples = posterior[mu_var].values.flatten()
        mu_mean = float(np.mean(mu_samples))
        details["group_mean"] = round(mu_mean, 2)

        # The grand mean should be roughly 3-12 (weighted avg of school effects)
        if -5 < mu_mean < 20:
            score += 2
            details["group_mean_plausible"] = True
        else:
            details["group_mean_plausible"] = False

    # Check that individual school effects exist and vary
    # Look for array-valued variables (school-level parameters)
    array_vars = [v for v in posterior.data_vars
                  if posterior[v].values.ndim > 2]  # chain x draw x schools
    details["array_vars"] = array_vars[:5]

    if array_vars:
        var = array_vars[0]
        means = np.mean(posterior[var].values, axis=(0, 1))
        details["school_effect_range"] = [round(float(means.min()), 2),
                                          round(float(means.max()), 2)]
        # Effects should span a reasonable range
        if means.max() - means.min() > 1.0:
            score += 2
            details["effects_vary"] = True
        else:
            score += 1
            details["effects_vary"] = False

    # Check finite and non-degenerate
    if _posterior_all_finite(posterior):
        score += 1

    return min(score, 5), details


def _recovery_T2_ordinal(idata, details: dict) -> tuple[int, dict]:
    """T2: Check cutpoints are ordered and coefficients have expected signs."""
    score = 0
    posterior = idata.posterior

    # Look for cutpoints
    cut_names = [v for v in posterior.data_vars
                 if any(k in v.lower() for k in ["cutpoint", "threshold", "cut"])]
    details["cutpoint_vars"] = cut_names

    if cut_names:
        cut_var = cut_names[0]
        cut_means = np.mean(posterior[cut_var].values, axis=(0, 1))
        details["cutpoint_means"] = [round(float(c), 2) for c in cut_means]

        # Cutpoints should be ordered
        is_ordered = all(cut_means[i] <= cut_means[i + 1]
                         for i in range(len(cut_means) - 1))
        details["cutpoints_ordered"] = is_ordered
        if is_ordered:
            score += 2
        else:
            score += 1

    # Check that depression coefficient is negative (depression -> less satisfaction)
    dep_names = [v for v in posterior.data_vars
                 if any(k in v.lower() for k in ["dep", "hlthdep", "depression"])]
    if dep_names:
        dep_mean = float(np.mean(posterior[dep_names[0]].values))
        details["depression_coeff"] = round(dep_mean, 3)
        if dep_mean < 0:
            score += 2
            details["depression_sign_correct"] = True
        else:
            score += 1
            details["depression_sign_correct"] = False

    # Finite check
    if _posterior_all_finite(posterior):
        score += 1

    return min(score, 5), details


def _recovery_T3_stochastic_volatility(idata, details: dict) -> tuple[int, dict]:
    """T3: Check latent volatility process is reasonable."""
    score = 0
    posterior = idata.posterior

    # Look for volatility-related variables.
    # Use word-boundary regex for short names like "h" to avoid matching
    # unrelated variables (e.g., "theta", "alpha").
    _vol_pat = re.compile(
        r'vol|log.?vol|sigma_h|(?:^|_)h(?:$|_|\d)|latent|innovation',
        re.IGNORECASE,
    )
    vol_names = [v for v in posterior.data_vars if _vol_pat.search(v)]
    details["volatility_vars"] = vol_names[:5]

    # Check for nu (degrees of freedom) — should be > 2 and < 30
    _nu_pat = re.compile(r'(?:^|_)nu(?:$|_|\d)|(?:^|_)df(?:$|_|\d)', re.IGNORECASE)
    nu_names = [v for v in posterior.data_vars if _nu_pat.search(v)]
    if nu_names:
        nu_mean = float(np.mean(posterior[nu_names[0]].values))
        details["nu_mean"] = round(nu_mean, 2)
        if 2 < nu_mean < 50:
            score += 2
            details["nu_plausible"] = True
        else:
            score += 1
            details["nu_plausible"] = False

    # Check for volatility step size (should be small, < 1)
    step_names = [v for v in posterior.data_vars
                  if any(k in v.lower() for k in ["sigma", "step_size", "sigma_h"])]
    if step_names:
        for name in step_names:
            vals = posterior[name].values.flatten()
            if vals.ndim == 1 and len(vals) > 0:
                step_mean = float(np.mean(vals))
                if 0 < step_mean < 2:
                    score += 1
                    details["step_size_plausible"] = True
                    break

    # Finite and non-degenerate
    if _posterior_all_finite(posterior):
        score += 1

    # Has time-varying component
    time_vars = [v for v in posterior.data_vars if posterior[v].values.ndim > 2]
    if time_vars:
        score += 1
        details["has_time_varying"] = True

    return min(score, 5), details


def _recovery_T4_mixture(idata, details: dict) -> tuple[int, dict]:
    """T4: Check mixture component recovery.

    Ground truth: 3 components at [-5.0, 0.0, 5.0], SDs [0.5, 2.0, 0.75],
    equal weights (1/3 each).
    """
    true_centers = np.array([-5.0, 0.0, 5.0])
    score = 0
    posterior = idata.posterior

    # Find component means (various naming conventions)
    mean_names = [v for v in posterior.data_vars
                  if any(k in v.lower() for k in ["mu", "mean", "center", "loc"])]
    details["mean_candidates"] = mean_names

    if mean_names:
        for name in mean_names:
            vals = posterior[name].values
            means = np.mean(vals, axis=(0, 1))  # average over chains and draws
            if means.ndim == 0:
                continue  # scalar, not component means
            means_sorted = np.sort(means.flatten())
            details["component_means"] = [round(float(m), 2) for m in means_sorted]

            if len(means_sorted) >= 3:
                # Check how many true centers are recovered (within 1.5)
                recovered = 0
                for tc in true_centers:
                    if any(abs(m - tc) < 1.5 for m in means_sorted):
                        recovered += 1
                details["centers_recovered"] = recovered

                if recovered == 3:
                    score += 3
                elif recovered == 2:
                    score += 2
                elif recovered >= 1:
                    score += 1
                break
            elif len(means_sorted) >= 2:
                score += 1
                details["note"] = f"only {len(means_sorted)} components found"
                break

    # Check weights sum to ~1 and are not degenerate.
    # Use word-boundary regex for short names like "w" and "pi" to avoid
    # matching unrelated variables (e.g., "mu_raw", "spike").
    _wt_pat = re.compile(
        r'weight|(?:^|_)w(?:$|_|\d)|(?:^|_)pi(?:$|_|\d)|mix.?prob',
        re.IGNORECASE,
    )
    weight_names = [v for v in posterior.data_vars if _wt_pat.search(v)]
    if weight_names:
        for name in weight_names:
            w_vals = np.mean(posterior[name].values, axis=(0, 1))
            if w_vals.ndim > 0 and len(w_vals) >= 2:
                details["weights"] = [round(float(w), 3) for w in w_vals]
                weight_sum = float(np.sum(w_vals))
                if 0.95 < weight_sum < 1.05:
                    score += 1
                    details["weights_valid"] = True
                break

    # Finite posteriors
    if _posterior_all_finite(posterior):
        score += 1

    return min(score, 5), details


def _recovery_T5_horseshoe(idata, details: dict) -> tuple[int, dict]:
    """T5: Check shrinkage — at least some coefficients near zero."""
    score = 0
    posterior = idata.posterior

    # Find coefficient variables (beta, coeff, etc.)
    coeff_names = [v for v in posterior.data_vars
                   if any(k in v.lower() for k in ["beta", "coeff", "b_"])]
    details["coeff_candidates"] = coeff_names

    if coeff_names:
        # Collect all coefficient means
        all_means = []
        for name in coeff_names:
            vals = posterior[name].values
            means = np.mean(vals, axis=(0, 1)).flatten()
            all_means.extend(means.tolist())

        if all_means:
            all_means = np.array(all_means)
            details["coeff_means"] = [round(float(m), 3) for m in all_means]

            # Count how many are shrunk near zero (|mean| < 0.1)
            near_zero = int(np.sum(np.abs(all_means) < 0.1))
            not_zero = int(np.sum(np.abs(all_means) >= 0.1))
            details["near_zero_count"] = near_zero
            details["not_zero_count"] = not_zero

            # Good shrinkage: some near zero, some not
            if near_zero >= 2 and not_zero >= 1:
                score += 3
                details["shrinkage_pattern"] = "good"
            elif near_zero >= 1:
                score += 2
                details["shrinkage_pattern"] = "partial"
            elif not_zero > 0:
                score += 1
                details["shrinkage_pattern"] = "weak"

    # Check for tau (global shrinkage) — should be small
    tau_names = [v for v in posterior.data_vars
                 if any(k in v.lower() for k in ["tau", "global_shrinkage"])]
    if tau_names:
        tau_mean = float(np.mean(posterior[tau_names[0]].values))
        details["tau_mean"] = round(tau_mean, 4)
        if 0 < tau_mean < 5:
            score += 1

    # Finite posteriors
    if _posterior_all_finite(posterior):
        score += 1

    return min(score, 5), details


# Registry of task-specific recovery scorers
_RECOVERY_SCORERS = {
    "T1_hierarchical": _recovery_T1_hierarchical,
    "T2_ordinal": _recovery_T2_ordinal,
    "T3_stochastic_volatility": _recovery_T3_stochastic_volatility,
    "T4_mixture": _recovery_T4_mixture,
    "T5_horseshoe": _recovery_T5_horseshoe,
}


def _count_rewrites_from_turns(turns_path: Path) -> tuple[int, int]:
    """Count model.py writes and bash executions from turns.jsonl.

    Returns (model_writes, bash_runs) where:
    - model_writes: Write tool calls targeting model.py
    - bash_runs: Bash tool calls that execute model.py
    """
    model_writes = 0
    bash_runs = 0

    with open(turns_path) as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            # Assistant messages have a "message" key with "content" blocks
            content = obj.get("message", {}).get("content", [])
            if not isinstance(content, list):
                continue

            for block in content:
                if block.get("type") != "tool_use":
                    continue
                tool_name = block.get("name", "")
                tool_input = block.get("input", {})

                if tool_name == "Write":
                    file_path = tool_input.get("file_path", "")
                    if "model.py" in file_path:
                        model_writes += 1

                elif tool_name == "Bash":
                    command = tool_input.get("command", "")
                    if re.search(r"python\s+.*model\.py", command):
                        bash_runs += 1

    return model_writes, bash_runs


def evaluate_pass_fail(
    run_dir: Path,
    model_produced_score: int,
    convergence_score: int,
    idata=None,
) -> tuple[bool, dict]:
    """Evaluate hard pass/fail gate for a benchmark run.

    A run passes if all three conditions hold:
    1. Sampling completed: model_produced >= 4 (>100 draws)
    2. Convergence acceptable: convergence >= 3
    3. Non-degenerate estimates: all posterior means finite, at least one var has std > 0
    """
    details: dict = {}

    # Check 1: sampling completed
    if model_produced_score < 4:
        details["sampling_completed"] = False
        details["reason"] = f"model_produced={model_produced_score} < 4"
        return False, details
    details["sampling_completed"] = True

    # Check 2: convergence acceptable
    if convergence_score < 3:
        details["convergence_acceptable"] = False
        details["reason"] = f"convergence={convergence_score} < 3"
        return False, details
    details["convergence_acceptable"] = True

    # Check 3: non-degenerate posterior estimates
    nc_path = run_dir / "results.nc"
    if not nc_path.exists():
        details["non_degenerate"] = False
        details["reason"] = "no results.nc"
        return False, details

    try:
        if idata is None:
            idata = az.from_netcdf(str(nc_path))
        if not hasattr(idata, "posterior") or idata.posterior is None:
            details["non_degenerate"] = False
            details["reason"] = "no posterior group"
            return False, details

        all_finite = True
        any_nonzero_std = False

        for var_name in idata.posterior.data_vars:
            values = idata.posterior[var_name].values
            mean_val = np.mean(values)
            std_val = np.std(values)

            if not np.isfinite(mean_val):
                all_finite = False
                details["non_finite_var"] = var_name
                break

            if std_val > 0:
                any_nonzero_std = True

        if not all_finite:
            details["non_degenerate"] = False
            details["reason"] = f"non-finite mean in variable '{details.get('non_finite_var')}'"
            return False, details

        if not any_nonzero_std:
            details["non_degenerate"] = False
            details["reason"] = "all variables have zero std (degenerate)"
            return False, details

        details["non_degenerate"] = True

    except Exception as e:
        details["non_degenerate"] = False
        details["reason"] = f"error checking posterior: {e}"
        return False, details

    return True, details


def count_retries(run_dir: Path) -> tuple[int, dict]:
    """Count raw error-fix cycles (model.py rewrites beyond the first).

    Uses turns.jsonl if available, otherwise falls back to metadata heuristic.
    Reported as informational metadata, not scored.
    """
    details: dict = {}
    turns_path = run_dir / "turns.jsonl"

    if turns_path.exists():
        model_writes, bash_runs = _count_rewrites_from_turns(turns_path)
        details["method"] = "turns_analysis"
        details["model_writes"] = model_writes
        details["bash_runs"] = bash_runs
        retries = max(0, model_writes - 1)
        details["retries"] = retries
        return retries, details

    # Fallback: estimate from metadata num_turns
    meta_path = run_dir / "metadata.json"
    if not meta_path.exists():
        details["method"] = "none"
        details["reason"] = "no turns.jsonl or metadata"
        return 0, details

    meta = json.loads(meta_path.read_text())
    num_turns = meta.get("num_turns", 0)
    details["method"] = "fallback_num_turns"
    details["num_turns"] = num_turns
    retries = max(0, (num_turns - 6) // 6)
    details["retries"] = retries
    return retries, details


def _get_wall_time(run_dir: Path) -> float:
    """Read wall_time from metadata.json, defaulting to 0."""
    meta_path = run_dir / "metadata.json"
    if not meta_path.exists():
        return 0.0
    meta = json.loads(meta_path.read_text())
    return float(meta.get("wall_time", 0.0))


def score_run(run_dir: Path, task_id: str, condition: str, rep: int) -> ScoreResult:
    """Score all criteria for a single benchmark run."""
    result = ScoreResult(task_id=task_id, condition=condition, rep=rep)
    
    # Load InferenceData once if it exists
    nc_path = run_dir / "results.nc"
    idata = None
    if nc_path.exists():
        try:
            idata = az.from_netcdf(str(nc_path))
        except Exception as e:
            logger.warning(f"Failed to load {nc_path}: {e}")

    mp_score, mp_details = score_model_produced(run_dir, idata=idata)
    result.model_produced = mp_score
    result.details["model_produced"] = mp_details

    conv_score, conv_details = score_convergence(run_dir, idata=idata)
    result.convergence = conv_score
    result.details["convergence"] = conv_details

    # No usable posterior — convergence is 0
    if mp_score < 3:
        result.convergence = 0

    ma_score, ma_details = score_model_appropriateness_llm(run_dir, task_id)
    result.model_appropriateness = ma_score
    result.details["model_appropriateness"] = ma_details

    bp_score, bp_details = score_best_practices(run_dir, task_id)
    result.best_practices = bp_score
    result.details["best_practices"] = bp_details

    wf_score, wf_details = score_workflow(run_dir)
    result.workflow = wf_score
    result.details["workflow"] = wf_details

    pr_score, pr_details = score_parameter_recovery(run_dir, task_id, idata=idata)
    result.parameter_recovery = pr_score
    result.details["parameter_recovery"] = pr_details

    result.compute_total()

    passed, pf_details = evaluate_pass_fail(
        run_dir, result.model_produced, result.convergence, idata=idata
    )
    result.passed = passed
    result.details["pass_fail"] = pf_details

    # Override for runs that exceeded the timeout cap — treat as failures
    wall_time = _get_wall_time(run_dir)
    if wall_time > DEFAULT_TIMEOUT:
        result.passed = False
        result.details["timeout_override"] = {
            "wall_time": wall_time,
            "cap": DEFAULT_TIMEOUT,
            "reason": f"wall_time {wall_time:.0f}s exceeds {DEFAULT_TIMEOUT}s cap",
        }
        logger.warning(
            f"Timeout override: {task_id} {condition} rep{rep} "
            f"wall_time={wall_time:.0f}s > {DEFAULT_TIMEOUT}s"
        )

    retries, retry_details = count_retries(run_dir)
    result.retries = retries
    result.details["retries"] = retry_details

    logger.info(
        f"Score {task_id} {condition} rep{rep}: "
        f"produced={result.model_produced} conv={result.convergence} "
        f"approp={result.model_appropriateness} bp={result.best_practices} "
        f"workflow={result.workflow} recovery={result.parameter_recovery} "
        f"total={result.total} passed={result.passed} retries={result.retries}"
    )

    return result


def score_all(runs_dir: Path | None = None) -> list[ScoreResult]:
    """Score all completed runs."""
    if runs_dir is None:
        runs_dir = RUNS_DIR

    # Load valid task IDs from tasks.yaml
    valid_tasks = set(load_tasks()["tasks"].keys())

    results = []
    for run_dir in sorted(runs_dir.iterdir()):
        if not run_dir.is_dir():
            continue

        meta_path = run_dir / "metadata.json"
        if not meta_path.exists():
            continue

        meta = json.loads(meta_path.read_text())
        task_id = meta["task_id"]
        condition = meta["condition"]
        rep = meta["rep"]

        if task_id not in valid_tasks:
            logger.warning(f"Skipping unknown task_id '{task_id}' in {run_dir.name}")
            continue

        # Skip infrastructure failures that produced no work product
        model_py = run_dir / "model.py"
        if not meta.get("success", False) and not model_py.exists():
            logger.info(f"Skipping failed run with no model.py: {run_dir.name}")
            continue

        score = score_run(run_dir, task_id, condition, rep)
        results.append(score)

        # Save score
        scores_dir = run_dir.parent.parent / "scores"
        scores_dir.mkdir(parents=True, exist_ok=True)
        score_file = scores_dir / f"{task_id}_{condition}_rep{rep}.json"
        score_file.write_text(json.dumps({
            "task_id": task_id,
            "condition": condition,
            "rep": rep,
            "model_produced": score.model_produced,
            "convergence": score.convergence,
            "model_appropriateness": score.model_appropriateness,
            "best_practices": score.best_practices,
            "workflow": score.workflow,
            "parameter_recovery": score.parameter_recovery,
            "total": score.total,
            "passed": score.passed,
            "retries": score.retries,
            "details": score.details,
        }, indent=2))

    return results
