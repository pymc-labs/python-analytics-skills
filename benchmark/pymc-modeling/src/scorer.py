"""Benchmark scorer — 6-criterion rubric (30 max).

Criteria:
1. Model produced (0-5): automated, checks results.nc
2. Convergence (0-5): automated, reads r_hat/ESS/divergences from .nc
3. Model appropriateness (0-5): LLM judge (Haiku) with task-specific rubric
4. Best practices (0-5): regex on model.py for coords/dims, nutpie, etc.
5. Thrashing (0-5): how many rewrite cycles before success (fewer = better)
6. Efficiency (0-5): how many turns Claude needed (fewer = better)
"""

import json
import logging
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

BENCHMARK_DIR = Path(__file__).parent.parent
RESULTS_DIR = BENCHMARK_DIR / "results"
RUNS_DIR = RESULTS_DIR / "runs"
TASKS_PATH = Path(__file__).parent.parent / "tasks.yaml"

# LLM judge settings
JUDGE_MODEL = "haiku"
JUDGE_BUDGET = "0.05"
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
    thrashing: int = 0
    efficiency: int = 0
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
            + self.thrashing
            + self.efficiency
        )


def _sampling_completed(run_dir: Path) -> bool:
    """Check if MCMC sampling completed (results.nc has posterior)."""
    nc_path = run_dir / "results.nc"
    if not nc_path.exists():
        return False
    try:
        import arviz as az
        idata = az.from_netcdf(str(nc_path))
        return hasattr(idata, "posterior") and idata.posterior is not None
    except Exception:
        return False


def score_model_produced(run_dir: Path) -> tuple[int, dict]:
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
        import arviz as az
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


def score_convergence(run_dir: Path) -> tuple[int, dict]:
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
        import arviz as az
        import numpy as np

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

    with open(TASKS_PATH) as f:
        config = yaml.safe_load(f)

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

    with open(TASKS_PATH) as f:
        config = yaml.safe_load(f)

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


def score_thrashing(run_dir: Path) -> tuple[int, dict]:
    """Score criterion 5: Answer thrashing (0-5, 5 = best).

    Measures how many rewrite cycles Claude went through before producing
    a working model. Fewer rewrites = skill provided correct patterns.

    5 = 0 rewrites (single write, single run, success)
    4 = 1 rewrite (one error-fix cycle)
    3 = 2 rewrites
    2 = 3 rewrites
    1 = 4+ rewrites or recurring same error
    0 = no model.py produced, or timeout with no progress
    """
    details = {}
    turns_path = run_dir / "turns.jsonl"

    if not turns_path.exists():
        # Fallback: use num_turns from metadata as a rough proxy
        meta_path = run_dir / "metadata.json"
        if not meta_path.exists():
            return 0, {"reason": "no metadata"}

        meta = json.loads(meta_path.read_text())
        num_turns = meta.get("num_turns", 0)
        details["method"] = "fallback_num_turns"
        details["num_turns"] = num_turns

        if num_turns == 0:
            return 0, details

        # Relaxed thresholds when we don't have detailed turn data
        # Assume ~1 rewrite per 6 extra turns beyond the initial ~6
        estimated_rewrites = max(0, (num_turns - 6) // 6)
        details["estimated_rewrites"] = estimated_rewrites

        if estimated_rewrites == 0:
            return 5, details
        elif estimated_rewrites == 1:
            return 4, details
        elif estimated_rewrites == 2:
            return 3, details
        elif estimated_rewrites == 3:
            return 2, details
        else:
            return 1, details

    model_writes, bash_runs = _count_rewrites_from_turns(turns_path)
    details["method"] = "turns_analysis"
    details["model_writes"] = model_writes
    details["bash_runs"] = bash_runs

    if model_writes == 0:
        details["reason"] = "no model.py writes"
        return 0, details

    # Rewrites = writes beyond the first one
    rewrites = max(0, model_writes - 1)
    details["rewrites"] = rewrites

    if rewrites == 0:
        return 5, details
    elif rewrites == 1:
        return 4, details
    elif rewrites == 2:
        return 3, details
    elif rewrites == 3:
        return 2, details
    else:
        return 1, details


def score_efficiency(run_dir: Path) -> tuple[int, dict]:
    """Score criterion 6: Efficiency (0-5, 5 = fastest).

    Measures how many turns Claude needed. Fewer turns = more efficient.

    5 = 1-6 turns
    4 = 7-12 turns
    3 = 13-18 turns
    2 = 19-25 turns
    1 = 26-35 turns
    0 = >35 turns or timeout (0 turns)
    """
    details = {}
    meta_path = run_dir / "metadata.json"

    if not meta_path.exists():
        return 0, {"reason": "no metadata"}

    meta = json.loads(meta_path.read_text())
    num_turns = meta.get("num_turns", 0)
    details["num_turns"] = num_turns

    if num_turns == 0:
        return 0, details
    elif num_turns <= 6:
        return 5, details
    elif num_turns <= 12:
        return 4, details
    elif num_turns <= 18:
        return 3, details
    elif num_turns <= 25:
        return 2, details
    elif num_turns <= 35:
        return 1, details
    else:
        return 0, details


def evaluate_pass_fail(
    run_dir: Path,
    model_produced_score: int,
    convergence_score: int,
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
        import arviz as az
        import numpy as np

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


def score_run(run_dir: Path, task_id: str, condition: str, rep: int) -> ScoreResult:
    """Score all criteria for a single benchmark run."""
    result = ScoreResult(task_id=task_id, condition=condition, rep=rep)

    mp_score, mp_details = score_model_produced(run_dir)
    result.model_produced = mp_score
    result.details["model_produced"] = mp_details

    conv_score, conv_details = score_convergence(run_dir)
    result.convergence = conv_score
    result.details["convergence"] = conv_details

    # Partial credit: if sampling completed but something crashed after,
    # still score convergence
    if mp_score >= 3 and _sampling_completed(run_dir):
        # Already scored above
        pass
    elif mp_score < 3:
        # No usable posterior — convergence is 0
        result.convergence = 0

    ma_score, ma_details = score_model_appropriateness_llm(run_dir, task_id)
    result.model_appropriateness = ma_score
    result.details["model_appropriateness"] = ma_details

    bp_score, bp_details = score_best_practices(run_dir, task_id)
    result.best_practices = bp_score
    result.details["best_practices"] = bp_details

    thr_score, thr_details = score_thrashing(run_dir)
    result.thrashing = thr_score
    result.details["thrashing"] = thr_details

    eff_score, eff_details = score_efficiency(run_dir)
    result.efficiency = eff_score
    result.details["efficiency"] = eff_details

    result.compute_total()

    passed, pf_details = evaluate_pass_fail(
        run_dir, result.model_produced, result.convergence
    )
    result.passed = passed
    result.details["pass_fail"] = pf_details

    retries, retry_details = count_retries(run_dir)
    result.retries = retries
    result.details["retries"] = retry_details

    logger.info(
        f"Score {task_id} {condition} rep{rep}: "
        f"produced={result.model_produced} conv={result.convergence} "
        f"approp={result.model_appropriateness} bp={result.best_practices} "
        f"thrash={result.thrashing} eff={result.efficiency} "
        f"total={result.total} passed={result.passed} retries={result.retries}"
    )

    return result


def score_all(runs_dir: Path | None = None) -> list[ScoreResult]:
    """Score all completed runs."""
    if runs_dir is None:
        runs_dir = RUNS_DIR

    # Load valid task IDs from tasks.yaml
    with open(TASKS_PATH) as f:
        valid_tasks = set(yaml.safe_load(f)["tasks"].keys())

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
            "thrashing": score.thrashing,
            "efficiency": score.efficiency,
            "total": score.total,
            "passed": score.passed,
            "retries": score.retries,
            "details": score.details,
        }, indent=2))

    return results
