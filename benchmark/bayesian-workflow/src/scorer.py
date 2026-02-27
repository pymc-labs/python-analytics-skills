"""Benchmark scorer — 6 workflow-focused criteria (30 max).

Criteria:
1. Iterative approach (0-5): Did Claude build simple → complex?
2. Prior predictive checking (0-5): Did Claude check prior implications?
3. Model criticism (0-5): Did Claude do posterior predictive checks?
4. Convergence & diagnostics (0-5): Did Claude diagnose properly?
5. Reporting quality (0-5): Did Claude report the workflow?
6. Efficiency (0-5): How many turns Claude needed (fewer = better)
"""

import json
import logging
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from src.extractor import extract_workflow_trace

logger = logging.getLogger(__name__)

BENCHMARK_DIR = Path(__file__).parent.parent
RESULTS_DIR = BENCHMARK_DIR / "results"
RUNS_DIR = RESULTS_DIR / "runs"
TASKS_PATH = BENCHMARK_DIR / "tasks.yaml"

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
    iterative_approach: int = 0
    prior_predictive: int = 0
    model_criticism: int = 0
    diagnostics: int = 0
    reporting: int = 0
    efficiency: int = 0
    total: int = 0
    passed: bool = False
    retries: int = 0
    details: dict = field(default_factory=dict)

    def compute_total(self):
        self.total = (
            self.iterative_approach
            + self.prior_predictive
            + self.model_criticism
            + self.diagnostics
            + self.reporting
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


def _extract_judge_json(response: str) -> dict | None:
    """Extract {"score": N, "reasoning": "..."} from LLM judge response."""
    try:
        data = json.loads(response)
        if "score" in data:
            return data
    except (json.JSONDecodeError, TypeError):
        pass

    fenced = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response, re.DOTALL)
    if fenced:
        try:
            data = json.loads(fenced.group(1))
            if "score" in data:
                return data
        except (json.JSONDecodeError, TypeError):
            pass

    brace = re.search(r'\{.*\}', response, re.DOTALL)
    if brace:
        try:
            data = json.loads(brace.group())
            if "score" in data:
                return data
        except (json.JSONDecodeError, TypeError):
            pass

    score_match = re.search(r'"score"\s*:\s*(\d)', response)
    if score_match:
        return {"score": int(score_match.group(1)), "reasoning": "extracted from partial JSON"}

    return None


def _call_judge(prompt: str) -> dict | None:
    """Call LLM judge and return parsed JSON response."""
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
            input=prompt,
            capture_output=True,
            text=True,
            timeout=JUDGE_TIMEOUT,
        )
        response = proc.stdout.strip()
        return _extract_judge_json(response), response
    except (subprocess.TimeoutExpired, Exception) as e:
        logger.warning(f"LLM judge failed: {e}")
        return None, str(e)


def score_iterative_approach(
    run_dir: Path, task_id: str, trace: dict
) -> tuple[int, dict]:
    """Score criterion 1: Iterative approach (0-5).

    Auto floor: total_models_fitted >= 2 → floor = 2
    LLM judge evaluates quality of iteration.
    """
    details = {"method": "hybrid"}

    # Auto floor from trace
    auto_floor = 0
    if trace["total_models_fitted"] >= 2:
        auto_floor = 2
        details["auto_floor_reason"] = f"fitted {trace['total_models_fitted']} models"
    if trace["loo_comparisons"] > 0:
        auto_floor = max(auto_floor, 2)
        details["loo_bonus"] = True

    details["auto_floor"] = auto_floor
    details["total_models_fitted"] = trace["total_models_fitted"]
    details["total_models_written"] = trace["total_models_written"]

    # LLM judge
    with open(TASKS_PATH) as f:
        config = yaml.safe_load(f)
    task = config["tasks"][task_id]

    judge_prompt = f"""Score this Bayesian modeling workflow for ITERATIVE APPROACH on a 0-5 scale.

Task: {task['name']}
Task description: {task['prompt'][:500]}
Reference workflow: {task['reference_workflow'][:500]}

Workflow trace:
- Models written: {trace['total_models_written']}
- Models fitted: {trace['total_models_fitted']}
- LOO comparisons: {trace['loo_comparisons']}
- Model files: {[m['file'] for m in trace['models']]}

Reasoning excerpts:
{chr(10).join(trace['reasoning_excerpts'][:10])}

Rubric:
0 = No model produced or only one model with no iteration
1 = Single model, no comparison to alternatives
2 = Two models fitted but no systematic building (e.g., just fixed errors)
3 = Clear simple→complex progression with 2-3 models
4 = Well-structured iteration with model comparisons (LOO or similar)
5 = Exemplary: systematic build-up, each expansion justified, formal comparison

Respond with ONLY a JSON object: {{"score": N, "reasoning": "brief explanation"}}"""

    parsed_judge, raw_response = _call_judge(judge_prompt)
    details["llm_response"] = raw_response[:500] if raw_response else ""

    if parsed_judge is not None:
        judge_score = max(0, min(5, int(parsed_judge.get("score", 0))))
        details["judge_score"] = judge_score
        details["reasoning"] = parsed_judge.get("reasoning", "")
        return max(auto_floor, judge_score), details

    # Fallback: use auto floor
    details["method"] = "auto_only"
    return auto_floor, details


def score_prior_predictive(
    run_dir: Path, task_id: str, trace: dict
) -> tuple[int, dict]:
    """Score criterion 2: Prior predictive checking (0-5).

    Auto floor: prior_predictive_count > 0 → floor = 2
    LLM judge evaluates quality.
    """
    details = {"method": "hybrid"}

    auto_floor = 0
    if trace["prior_predictive_count"] > 0:
        auto_floor = 2
        details["auto_floor_reason"] = f"prior predictive count: {trace['prior_predictive_count']}"
    details["auto_floor"] = auto_floor
    details["prior_predictive_count"] = trace["prior_predictive_count"]

    with open(TASKS_PATH) as f:
        config = yaml.safe_load(f)
    task = config["tasks"][task_id]

    judge_prompt = f"""Score this Bayesian modeling workflow for PRIOR PREDICTIVE CHECKING on a 0-5 scale.

Task: {task['name']}
Task description: {task['prompt'][:500]}

Workflow trace:
- Prior predictive checks: {trace['prior_predictive_count']}
- Models written: {trace['total_models_written']}

Reasoning excerpts:
{chr(10).join(trace['reasoning_excerpts'][:10])}

Rubric:
0 = No prior predictive checking at all
1 = Mentions priors but doesn't check their implications
2 = Runs sample_prior_predictive at least once
3 = Checks prior predictive AND discusses whether priors are reasonable
4 = Adjusts priors based on prior predictive results
5 = Systematic prior checking: initial check, adjustment, re-check, with clear justification

Respond with ONLY a JSON object: {{"score": N, "reasoning": "brief explanation"}}"""

    parsed_judge, raw_response = _call_judge(judge_prompt)
    details["llm_response"] = raw_response[:500] if raw_response else ""

    if parsed_judge is not None:
        judge_score = max(0, min(5, int(parsed_judge.get("score", 0))))
        details["judge_score"] = judge_score
        details["reasoning"] = parsed_judge.get("reasoning", "")
        return max(auto_floor, judge_score), details

    details["method"] = "auto_only"
    return auto_floor, details


def score_model_criticism(
    run_dir: Path, task_id: str, trace: dict
) -> tuple[int, dict]:
    """Score criterion 3: Model criticism (0-5).

    Auto floor: posterior_predictive_count > 0 → floor = 2
    LLM judge evaluates quality.
    """
    details = {"method": "hybrid"}

    auto_floor = 0
    if trace["posterior_predictive_count"] > 0:
        auto_floor = 2
        details["auto_floor_reason"] = f"posterior predictive count: {trace['posterior_predictive_count']}"
    details["auto_floor"] = auto_floor
    details["posterior_predictive_count"] = trace["posterior_predictive_count"]

    with open(TASKS_PATH) as f:
        config = yaml.safe_load(f)
    task = config["tasks"][task_id]

    judge_prompt = f"""Score this Bayesian modeling workflow for MODEL CRITICISM on a 0-5 scale.

Task: {task['name']}
Task description: {task['prompt'][:500]}

Workflow trace:
- Posterior predictive checks: {trace['posterior_predictive_count']}
- LOO comparisons: {trace['loo_comparisons']}
- Diagnostics performed: {trace['diagnostics_count']}

Reasoning excerpts:
{chr(10).join(trace['reasoning_excerpts'][:10])}

Rubric:
0 = No model criticism at all
1 = Reports parameter estimates but no checks on model fit
2 = Runs posterior predictive check at least once
3 = Posterior predictive check AND discusses fit quality
4 = Systematic criticism: PPC + identified specific misfits that motivated model changes
5 = Exemplary: PPC at each model stage, clear link between criticism and model improvements

Respond with ONLY a JSON object: {{"score": N, "reasoning": "brief explanation"}}"""

    parsed_judge, raw_response = _call_judge(judge_prompt)
    details["llm_response"] = raw_response[:500] if raw_response else ""

    if parsed_judge is not None:
        judge_score = max(0, min(5, int(parsed_judge.get("score", 0))))
        details["judge_score"] = judge_score
        details["reasoning"] = parsed_judge.get("reasoning", "")
        return max(auto_floor, judge_score), details

    details["method"] = "auto_only"
    return auto_floor, details


def score_diagnostics(run_dir: Path, trace: dict) -> tuple[int, dict]:
    """Score criterion 4: Convergence & diagnostics (0-5).

    Hybrid: automated checks on results.nc + trace evidence.
    """
    details = {}
    nc_path = run_dir / "results.nc"

    if not nc_path.exists():
        details["reason"] = "no results.nc"
        return 0, details

    score = 0

    try:
        import arviz as az
        import numpy as np

        idata = az.from_netcdf(str(nc_path))

        if not hasattr(idata, "posterior") or idata.posterior is None:
            details["reason"] = "no posterior group"
            return 0, details

        n_chains = idata.posterior.sizes.get("chain", 0)
        n_draws = idata.posterior.sizes.get("draw", 0)
        details["n_chains"] = n_chains
        details["n_draws"] = n_draws

        # Base: sampling ran
        score = 1

        # Divergences
        n_divergent = 0
        if hasattr(idata, "sample_stats"):
            try:
                div = idata.sample_stats.get("diverging")
                if div is not None:
                    n_divergent = int(div.values.sum())
            except Exception:
                pass
        details["n_divergent"] = n_divergent

        # R-hat (if multiple chains)
        if n_chains >= 2:
            try:
                rhat = az.rhat(idata)
                rhat_values = []
                for var in rhat.data_vars:
                    vals = rhat[var].values.flatten()
                    rhat_values.extend(vals[np.isfinite(vals)])
                rhat_values = np.array(rhat_values)
                max_rhat = float(np.max(rhat_values)) if len(rhat_values) > 0 else 999
                details["rhat_max"] = max_rhat
            except Exception:
                max_rhat = 999
        else:
            max_rhat = None
            details["rhat_max"] = "N/A (single chain)"

        # Score based on diagnostics quality
        if n_divergent == 0 and (max_rhat is None or max_rhat < 1.05):
            score = 3
        elif n_divergent < 10 and (max_rhat is None or max_rhat < 1.1):
            score = 2

        # Bonus for trace evidence of diagnostic checking
        if trace["diagnostics_count"] > 0:
            score = min(5, score + 1)
            details["diagnostics_in_trace"] = trace["diagnostics_count"]

        # Extra bonus for comprehensive checking
        if trace["diagnostics_count"] >= 3:
            score = min(5, score + 1)

    except Exception as e:
        details["error"] = str(e)
        return 0, details

    return score, details


def score_reporting(
    run_dir: Path, task_id: str, trace: dict
) -> tuple[int, dict]:
    """Score criterion 5: Reporting quality (0-5).

    LLM judge only — evaluates whether Claude reported the full workflow.
    """
    details = {"method": "llm"}

    with open(TASKS_PATH) as f:
        config = yaml.safe_load(f)
    task = config["tasks"][task_id]

    judge_prompt = f"""Score this Bayesian modeling workflow for REPORTING QUALITY on a 0-5 scale.

Task: {task['name']}
Task description: {task['prompt'][:500]}

Workflow trace:
- Models written: {trace['total_models_written']}
- Models fitted: {trace['total_models_fitted']}
- Prior predictive checks: {trace['prior_predictive_count']}
- Posterior predictive checks: {trace['posterior_predictive_count']}
- LOO comparisons: {trace['loo_comparisons']}
- Diagnostics: {trace['diagnostics_count']}

Reasoning excerpts (samples of what Claude reported):
{chr(10).join(trace['reasoning_excerpts'][:15])}

Rubric:
0 = No reporting — just code with no explanation
1 = Minimal reporting — states what model was fit but no workflow narrative
2 = Reports final results but doesn't describe the modeling journey
3 = Describes the sequence of models tried and why
4 = Clear narrative: what was tried, what was learned at each step, why changes were made
5 = Exemplary: structured report with model progression, diagnostic results, comparison outcomes, and conclusions

Respond with ONLY a JSON object: {{"score": N, "reasoning": "brief explanation"}}"""

    parsed_judge, raw_response = _call_judge(judge_prompt)
    details["llm_response"] = raw_response[:500] if raw_response else ""

    if parsed_judge is not None:
        judge_score = max(0, min(5, int(parsed_judge.get("score", 0))))
        details["judge_score"] = judge_score
        details["reasoning"] = parsed_judge.get("reasoning", "")
        return judge_score, details

    # Fallback: simple heuristic based on reasoning excerpts
    details["method"] = "fallback"
    n_excerpts = len(trace["reasoning_excerpts"])
    if n_excerpts >= 10:
        return 3, details
    elif n_excerpts >= 5:
        return 2, details
    elif n_excerpts >= 1:
        return 1, details
    return 0, details


def score_efficiency(run_dir: Path) -> tuple[int, dict]:
    """Score criterion 6: Efficiency (0-5, 5 = fastest).

    Workflow tasks are expected to take more turns than implementation tasks.
    Adjusted thresholds:

    5 = 1-10 turns
    4 = 11-18 turns
    3 = 19-26 turns
    2 = 27-35 turns
    1 = 36-45 turns
    0 = >45 turns or timeout (0 turns)
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
    elif num_turns <= 10:
        return 5, details
    elif num_turns <= 18:
        return 4, details
    elif num_turns <= 26:
        return 3, details
    elif num_turns <= 35:
        return 2, details
    elif num_turns <= 45:
        return 1, details
    else:
        return 0, details


def evaluate_pass_fail(
    run_dir: Path,
    diagnostics_score: int,
) -> tuple[bool, dict]:
    """Evaluate hard pass/fail gate for a benchmark run.

    A run passes if:
    1. Sampling completed (results.nc has posterior with >100 draws)
    2. Diagnostics acceptable (diagnostics >= 2)
    3. Non-degenerate estimates
    """
    details: dict = {}

    nc_path = run_dir / "results.nc"
    if not nc_path.exists():
        details["sampling_completed"] = False
        details["reason"] = "no results.nc"
        return False, details

    try:
        import arviz as az
        import numpy as np

        idata = az.from_netcdf(str(nc_path))

        if not hasattr(idata, "posterior") or idata.posterior is None:
            details["sampling_completed"] = False
            details["reason"] = "no posterior group"
            return False, details

        n_draws = idata.posterior.sizes.get("draw", 0)
        if n_draws < 100:
            details["sampling_completed"] = False
            details["reason"] = f"only {n_draws} draws"
            return False, details

        details["sampling_completed"] = True

        if diagnostics_score < 2:
            details["diagnostics_acceptable"] = False
            details["reason"] = f"diagnostics={diagnostics_score} < 2"
            return False, details
        details["diagnostics_acceptable"] = True

        # Non-degenerate check
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
            details["reason"] = f"non-finite mean in '{details.get('non_finite_var')}'"
            return False, details

        if not any_nonzero_std:
            details["non_degenerate"] = False
            details["reason"] = "all variables have zero std (degenerate)"
            return False, details

        details["non_degenerate"] = True

    except Exception as e:
        details["error"] = str(e)
        return False, details

    return True, details


def count_retries(run_dir: Path) -> tuple[int, dict]:
    """Count raw error-fix cycles (model.py rewrites beyond the first)."""
    details: dict = {}
    trace = extract_workflow_trace(run_dir)

    if trace["total_models_written"] > 0:
        details["method"] = "trace"
        details["total_models_written"] = trace["total_models_written"]
        # Count rewrites of the SAME file (not new model versions)
        rewrites = sum(1 for m in trace["models"] if m.get("is_rewrite", False))
        details["rewrites"] = rewrites
        return rewrites, details

    # Fallback: estimate from metadata num_turns
    meta_path = run_dir / "metadata.json"
    if not meta_path.exists():
        details["method"] = "none"
        return 0, details

    meta = json.loads(meta_path.read_text())
    num_turns = meta.get("num_turns", 0)
    details["method"] = "fallback_num_turns"
    details["num_turns"] = num_turns
    retries = max(0, (num_turns - 8) // 8)
    details["retries"] = retries
    return retries, details


def score_run(run_dir: Path, task_id: str, condition: str, rep: int) -> ScoreResult:
    """Score all criteria for a single benchmark run."""
    result = ScoreResult(task_id=task_id, condition=condition, rep=rep)

    # Extract workflow trace once, reuse across criteria
    trace = extract_workflow_trace(run_dir)
    result.details["trace_summary"] = {
        "total_models_written": trace["total_models_written"],
        "total_models_fitted": trace["total_models_fitted"],
        "prior_predictive_count": trace["prior_predictive_count"],
        "posterior_predictive_count": trace["posterior_predictive_count"],
        "loo_comparisons": trace["loo_comparisons"],
        "diagnostics_count": trace["diagnostics_count"],
    }

    # Criterion 1: Iterative approach
    iter_score, iter_details = score_iterative_approach(run_dir, task_id, trace)
    result.iterative_approach = iter_score
    result.details["iterative_approach"] = iter_details

    # Criterion 2: Prior predictive
    pp_score, pp_details = score_prior_predictive(run_dir, task_id, trace)
    result.prior_predictive = pp_score
    result.details["prior_predictive"] = pp_details

    # Criterion 3: Model criticism
    mc_score, mc_details = score_model_criticism(run_dir, task_id, trace)
    result.model_criticism = mc_score
    result.details["model_criticism"] = mc_details

    # Criterion 4: Diagnostics
    diag_score, diag_details = score_diagnostics(run_dir, trace)
    result.diagnostics = diag_score
    result.details["diagnostics"] = diag_details

    # Criterion 5: Reporting
    rep_score, rep_details = score_reporting(run_dir, task_id, trace)
    result.reporting = rep_score
    result.details["reporting"] = rep_details

    # Criterion 6: Efficiency
    eff_score, eff_details = score_efficiency(run_dir)
    result.efficiency = eff_score
    result.details["efficiency"] = eff_details

    result.compute_total()

    # Pass/fail gate
    passed, pf_details = evaluate_pass_fail(run_dir, result.diagnostics)
    result.passed = passed
    result.details["pass_fail"] = pf_details

    # Retries
    retries, retry_details = count_retries(run_dir)
    result.retries = retries
    result.details["retries"] = retry_details

    logger.info(
        f"Score {task_id} {condition} rep{rep}: "
        f"iter={result.iterative_approach} prior={result.prior_predictive} "
        f"crit={result.model_criticism} diag={result.diagnostics} "
        f"report={result.reporting} eff={result.efficiency} "
        f"total={result.total} passed={result.passed} retries={result.retries}"
    )

    return result


def score_all(runs_dir: Path | None = None) -> list[ScoreResult]:
    """Score all completed runs."""
    if runs_dir is None:
        runs_dir = RUNS_DIR

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

        model_py = run_dir / "model.py"
        if not meta.get("success", False) and not model_py.exists():
            logger.info(f"Skipping failed run with no model.py: {run_dir.name}")
            continue

        score = score_run(run_dir, task_id, condition, rep)
        results.append(score)

        scores_dir = run_dir.parent.parent / "scores"
        scores_dir.mkdir(parents=True, exist_ok=True)
        score_file = scores_dir / f"{task_id}_{condition}_rep{rep}.json"
        score_file.write_text(json.dumps({
            "task_id": task_id,
            "condition": condition,
            "rep": rep,
            "iterative_approach": score.iterative_approach,
            "prior_predictive": score.prior_predictive,
            "model_criticism": score.model_criticism,
            "diagnostics": score.diagnostics,
            "reporting": score.reporting,
            "efficiency": score.efficiency,
            "total": score.total,
            "passed": score.passed,
            "retries": score.retries,
            "details": score.details,
        }, indent=2))

    return results
