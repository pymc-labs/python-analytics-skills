"""Combined scoring: automated metrics + LLM-as-judge for code quality.

Scores each benchmark result on 5 criteria (0-5 each, total 25):
1. Execution — did the code run and produce InferenceData?
2. Convergence — r_hat, ESS, divergences from real diagnostics
3. Diagnostic completeness — which ArviZ functions are in the code?
4. Statistical quality — model structure, predictive checks
5. Code quality — LLM-as-judge (Haiku) for best practices
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

BENCHMARK_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BENCHMARK_DIR / "results" / "raw"
CODE_DIR = BENCHMARK_DIR / "results" / "code"
EXEC_DIR = BENCHMARK_DIR / "results" / "execution"
DIAG_DIR = BENCHMARK_DIR / "results" / "diagnostics"
SCORE_DIR = BENCHMARK_DIR / "results" / "scores"


def score_execution(exec_result: dict | None) -> int:
    """Score execution success (0-5)."""
    if exec_result is None:
        return 0
    if exec_result.get("error_type") == "missing_script":
        return 0
    if exec_result.get("error_type") == "syntax_error":
        return 1
    if exec_result.get("error_type") == "import_error":
        return 1
    if exec_result.get("error_type") in ("runtime_error", "memory_error"):
        return 2
    if exec_result.get("error_type") == "sampling_error":
        return 3
    if exec_result.get("error_type") == "timeout":
        return 2
    if exec_result.get("error_type") == "no_output":
        return 3
    if exec_result.get("execution_success") and not exec_result.get("idata_exists"):
        return 3
    if exec_result.get("execution_success") and exec_result.get("idata_exists"):
        return 5
    return 0


def score_convergence(diag_result: dict | None) -> int:
    """Score convergence from real diagnostics (0-5)."""
    if diag_result is None or not diag_result.get("load_success"):
        return 0

    conv = diag_result.get("convergence", {})
    samp = diag_result.get("sampling", {})

    if conv.get("error"):
        return 0

    r_hat_max = conv.get("r_hat_max")
    ess_bulk_min = conv.get("ess_bulk_min")
    n_div = samp.get("n_divergences", 0)

    if r_hat_max is None:
        return 1

    # Severe failure
    if r_hat_max > 1.1:
        return 1

    # Moderate issues
    if r_hat_max > 1.05 or n_div > 100:
        return 2

    # Minor issues
    if r_hat_max > 1.01 or (ess_bulk_min is not None and ess_bulk_min < 400):
        return 3

    # Good
    if conv.get("converged") and n_div == 0:
        ess_bulk_min = ess_bulk_min or 0
        if ess_bulk_min > 1000:
            return 5
        return 4

    # Good with minor divergences
    if conv.get("r_hat_all_below_1_01") and n_div < 10:
        return 4

    return 3


def score_diagnostic_completeness(code: str | None) -> int:
    """Score diagnostic completeness from code inspection (0-5).

    +1 for each diagnostic function found in the code.
    """
    if not code:
        return 0

    checks = [
        r"az\.summary\s*\(",
        r"az\.plot_trace\s*\(|az\.plot_rank\s*\(",
        r"sample_posterior_predictive|az\.plot_ppc\s*\(",
        r"az\.plot_loo_pit\s*\(|az\.loo\s*\(",
        r"az\.plot_energy\s*\(|az\.bfmi\s*\(",
    ]

    score = 0
    for pattern in checks:
        if re.search(pattern, code):
            score += 1

    return min(score, 5)


def score_statistical_quality(diag_result: dict | None) -> int:
    """Score statistical quality from diagnostics (0-5)."""
    if diag_result is None or not diag_result.get("load_success"):
        return 0

    structure = diag_result.get("model_structure", {})
    conv = diag_result.get("convergence", {})

    score = 0

    # Has posterior
    if structure.get("has_posterior"):
        score += 1

    # Model converged
    if conv.get("converged"):
        score += 1

    # Has prior predictive
    if structure.get("has_prior_predictive") or structure.get("has_prior"):
        score += 1

    # Has posterior predictive
    if structure.get("has_posterior_predictive"):
        score += 1

    # Has log likelihood (enables LOO-CV)
    if structure.get("has_log_likelihood"):
        score += 1

    return min(score, 5)


def score_code_quality_llm(code: str, task_name: str) -> dict:
    """Score code quality using LLM-as-judge (Haiku).

    Returns {"score": int, "reasoning": str}
    """
    if not code:
        return {"score": 0, "reasoning": "No code to evaluate"}

    prompt = f"""Score this PyMC code on a 0-5 scale for code quality.

Task: {task_name}

Criteria:
- 0: Non-functional or empty
- 1: Runs but very messy, no structure
- 2: Functional but poor practices (magic numbers, no comments)
- 3: Adequate structure and readability
- 4: Clean code with good practices (coords/dims, clear names)
- 5: Production-quality (coords/dims, nutpie, weakly informative priors, random seed, appropriate parameterization)

Specific things to check:
- Does it use coords and dims for interpretable InferenceData?
- Does it use nutpie sampler for efficiency?
- Are priors weakly informative (not flat sigma=100)?
- Does it set random_seed for reproducibility?
- Is parameterization appropriate (e.g., non-centered for hierarchical)?

Respond with ONLY a JSON object: {{"score": N, "reasoning": "brief explanation"}}

Code:
```python
{code[:8000]}
```"""

    try:
        proc = subprocess.run(
            [
                "claude",
                "--print",
                "--output-format",
                "json",
                "--model",
                "haiku",
                "--max-budget-usd",
                "0.05",
                "--no-session-persistence",
                "--disable-slash-commands",
            ],
            input=prompt,
            capture_output=True,
            text=True,
            timeout=60,
        )

        if proc.returncode != 0:
            return {"score": 0, "reasoning": f"LLM judge failed: {proc.stderr[:200]}"}

        # Parse the Claude JSON response
        try:
            output = json.loads(proc.stdout)
            response_text = output.get("result", "")
        except json.JSONDecodeError:
            response_text = proc.stdout

        # Extract JSON from response
        json_match = re.search(r'\{[^}]*"score"[^}]*\}', response_text)
        if json_match:
            return json.loads(json_match.group())

        return {
            "score": 0,
            "reasoning": f"Could not parse LLM response: {response_text[:200]}",
        }

    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        return {"score": 0, "reasoning": f"LLM judge error: {e}"}


def score_code_quality_automated(code: str) -> int:
    """Fallback automated code quality scoring (no LLM needed).

    Used when LLM-as-judge is unavailable (e.g., credit issues).
    """
    if not code:
        return 0
    if len(code) < 100:
        return 1

    score = 2  # Base: functional code present

    has_coords = bool(re.search(r"coords\s*=", code))
    has_dims = bool(re.search(r'dims\s*=\s*["\']', code))
    has_nutpie = bool(re.search(r'nuts_sampler\s*=\s*["\']nutpie["\']', code))
    has_seed = bool(re.search(r"random_seed\s*=", code))
    has_comments = bool(re.search(r"#\s*\w+", code))

    if has_coords or has_dims:
        score += 1
    if has_comments:
        score += 1
    if has_nutpie or has_seed:
        score += 1

    return min(score, 5)


def score_result(
    task_id: str,
    condition: str,
    replication: int,
    use_llm_judge: bool = True,
) -> dict:
    """Score a single benchmark result across all criteria."""
    SCORE_DIR.mkdir(parents=True, exist_ok=True)

    # Load raw result for metadata
    raw_files = list(RAW_DIR.glob(f"{task_id}_{condition}_rep{replication}_*.json"))
    raw_result = None
    if raw_files:
        with open(raw_files[-1]) as f:
            raw_result = json.load(f)

    # Load code
    code_path = CODE_DIR / f"{task_id}_{condition}_rep{replication}.py"
    code = code_path.read_text() if code_path.exists() else None

    # Load execution result
    exec_path = EXEC_DIR / f"{task_id}_{condition}_rep{replication}.json"
    exec_result = None
    if exec_path.exists():
        with open(exec_path) as f:
            exec_result = json.load(f)

    # Load diagnostics
    diag_path = DIAG_DIR / f"{task_id}_{condition}_rep{replication}.json"
    diag_result = None
    if diag_path.exists():
        with open(diag_path) as f:
            diag_result = json.load(f)

    # Compute scores
    task_name = raw_result.get("task_name", task_id) if raw_result else task_id

    execution = score_execution(exec_result)
    convergence = score_convergence(diag_result)
    diagnostic_completeness = score_diagnostic_completeness(code)
    statistical_quality = score_statistical_quality(diag_result)

    if use_llm_judge:
        llm_result = score_code_quality_llm(code or "", task_name)
        code_quality = llm_result.get("score", 0)
        llm_reasoning = llm_result.get("reasoning", "")
    else:
        code_quality = score_code_quality_automated(code or "")
        llm_reasoning = "automated fallback"

    scores = {
        "execution": execution,
        "convergence": convergence,
        "diagnostic_completeness": diagnostic_completeness,
        "statistical_quality": statistical_quality,
        "code_quality": code_quality,
    }

    result = {
        "task_id": task_id,
        "condition": condition,
        "replication": replication,
        "task_name": task_name,
        "tier": raw_result.get("tier", "") if raw_result else "",
        "scores": scores,
        "total_score": sum(scores.values()),
        "max_possible": 25,
        "code_quality_reasoning": llm_reasoning,
        "metrics": {
            "generation_time_seconds": (
                raw_result.get("execution_time_seconds") if raw_result else None
            ),
            "execution_time_seconds": (
                exec_result.get("wall_time_seconds") if exec_result else None
            ),
            "input_tokens": raw_result.get("input_tokens") if raw_result else None,
            "output_tokens": raw_result.get("output_tokens") if raw_result else None,
            "cost_usd": raw_result.get("cost_usd") if raw_result else None,
            "idata_exists": exec_result.get("idata_exists") if exec_result else False,
            "execution_success": (
                exec_result.get("execution_success") if exec_result else False
            ),
            "error_type": exec_result.get("error_type") if exec_result else None,
        },
    }

    # Save
    score_name = f"{task_id}_{condition}_rep{replication}.json"
    score_path = SCORE_DIR / score_name
    with open(score_path, "w") as f:
        json.dump(result, f, indent=2)

    return result


def score_all(use_llm_judge: bool = True) -> list[dict]:
    """Score all results."""
    # Find all unique (task_id, condition, replication) combos from raw results
    if not RAW_DIR.exists():
        print("No raw results found.")
        return []

    combos = set()
    for path in RAW_DIR.glob("*.json"):
        with open(path) as f:
            data = json.load(f)
        key = (data.get("task_id"), data.get("condition"), data.get("replication"))
        if all(key):
            combos.add(key)

    results = []
    for task_id, condition, replication in sorted(combos):
        print(f"  Scoring {task_id} {condition} rep{replication}...", end=" ")
        result = score_result(
            task_id, condition, replication, use_llm_judge=use_llm_judge
        )
        total = result["total_score"]
        print(f"{total}/25")
        results.append(result)

    if results:
        avg = sum(r["total_score"] for r in results) / len(results)
        print(f"\nScored {len(results)} results, average: {avg:.1f}/25")

    return results
