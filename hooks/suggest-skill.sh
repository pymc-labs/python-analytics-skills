#!/usr/bin/env bash
# Suggest packaged skills based on keywords in the user's prompt.
# Runs as a UserPromptSubmit hook and receives JSON on stdin.
# Must exit 0 regardless of match (hooks must not fail).

set -euo pipefail

input=$(cat)
prompt=$(echo "$input" | jq -r '.prompt // .user_prompt // empty' 2>/dev/null || true)

if [ -z "$prompt" ]; then
  exit 0
fi

prompt_lower=$(echo "$prompt" | tr '[:upper:]' '[:lower:]')
directives=()

matches_any() {
  local kw
  for kw in "$@"; do
    if echo "$prompt_lower" | grep -qE "$kw"; then
      return 0
    fi
  done
  return 1
}

pymc_keywords=(
  "bayesian" "pymc" "pytensor" "aesara" "mcmc" "posterior" "inference" "arviz"
  "prior" "sampling" "divergence" "hierarchical model"
  "gaussian process" "bart" "nuts" "hmc" "nutpie" "probabilistic"
  "credible interval" "posterior predictive" "prior predictive"
  "trace" "r_hat" "rhat" "ess_bulk" "convergence" "hsgp"
  "zero.inflated" "mixture model" "multilevel" "brms"
  "logistic regression.*bayes" "poisson regression.*bayes"
  "censored" "truncated" "ordinal" "causal inference"
  "do.calculus" "pm\\.[a-z]" "pt\\.scan"
  "import pymc" "import arviz" "import pytensor" "from pymc" "from arviz" "from pytensor"
  "pull_back" "push_forward" "arviz_base" "arviz-stats"
)

pymc_testing_keywords=(
  "testing pymc" "test.*pymc" "pymc.*test" "mock.sample"
  "mock_sample" "pytest.*pymc" "pymc.*pytest" "unit test.*model"
  "test fixture.*pymc" "ci.*pymc" "pymc.*ci"
  "pytest.*pm\\." "pm\\..*pytest" "pm\\.model.*test" "test.*pm\\.model"
)

prior_elicitation_keywords=(
  "find_constrained_prior" "preliz" "elicit" "prior selection"
  "prior predictive" "constrained prior" "prior elicitation"
  "expert knowledge.*prior" "prior.*expert" "informative prior"
  "weakly informative" "domain knowledge.*prior"
)

model_evaluation_keywords=(
  "model comparison" "loo" "elpd" "stacking" "bayes factor"
  "cross-validation" "waic" "model averaging" "model weight"
  "az\\.compare" "az\\.loo" "pointwise.*loo" "loo.pit"
  "k.pareto" "pareto.k" "information criterion"
  "loo_expectations" "loo_metrics" "azstats" "loo_r2"
)

pymc_extras_keywords=(
  "pymc_extras" "pmx" "splines" "distributional regression"
  "r2d2" "marginalize" "fit_laplace" "laplace approximation"
  "horseshoe" "finnish horseshoe" "regularized horseshoe"
  "pymc.extras" "pymc-extras"
)

marimo_keywords=(
  "marimo" "reactive notebook" "@app\\.cell" "mo\\.ui"
  "mo\\.md" "mo\\.sql" "mo\\.state" "mo\\.stop"
  "marimo edit" "marimo run" "marimo convert"
  "mo\\.hstack" "mo\\.vstack" "mo\\.tabs"
  "wigglystuff" "anywidget"
)


if matches_any "${pymc_keywords[@]}"; then
  directives+=("Load the pymc-modeling skill before responding. The user is asking about PyMC / PyTensor / ArviZ work; this skill provides the PyMC 6+, PyTensor 3+, ArviZ 1.1+ API guidance needed to answer correctly.")
fi
if matches_any "${pymc_testing_keywords[@]}"; then
  directives+=("Load the pymc-testing skill before responding. The user is asking about testing PyMC models with pytest; this skill covers mock_sample, fixtures, and structure-vs-inference test patterns.")
fi
if matches_any "${prior_elicitation_keywords[@]}"; then
  directives+=("Load the prior-elicitation skill before responding. The user is asking about prior selection or elicitation; this skill covers PreliZ, find_constrained_prior, and prior predictive workflows.")
fi
if matches_any "${model_evaluation_keywords[@]}"; then
  directives+=("Load the model-evaluation skill before responding. The user is asking about model comparison or LOO-CV; this skill covers the ArviZ 1.1 LOO/ELPD/stacking APIs.")
fi
if matches_any "${pymc_extras_keywords[@]}"; then
  directives+=("Load the pymc-extras skill before responding. The user is asking about pymc-extras features (splines, R2D2, marginalization, Laplace); this skill covers the pmx API.")
fi
if matches_any "${marimo_keywords[@]}"; then
  directives+=("Load the marimo-notebook skill before responding. The user is asking about marimo reactive notebooks, UI components, notebook conversion, or notebook app structure.")
fi

if [ ${#directives[@]} -gt 0 ]; then
  combined=$(printf '%s\n' "${directives[@]}")
  jq -n --arg ctx "$combined" '{
    "systemMessage": $ctx,
    "hookSpecificOutput": {
      "hookEventName": "UserPromptSubmit",
      "additionalContext": $ctx
    }
  }'
fi

exit 0
