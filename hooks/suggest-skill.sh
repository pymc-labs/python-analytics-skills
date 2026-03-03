#!/usr/bin/env bash
# Suggest analytics skills based on keywords in the user's prompt.
# Runs as a UserPromptSubmit hook — receives JSON on stdin with "user_prompt" field.
# Must exit 0 regardless of match (hooks must not fail).

set -euo pipefail

input=$(cat)
prompt=$(echo "$input" | jq -r '.user_prompt // empty' 2>/dev/null || true)

if [ -z "$prompt" ]; then
  exit 0
fi

# Convert to lowercase for matching
prompt_lower=$(echo "$prompt" | tr '[:upper:]' '[:lower:]')

suggest_pymc=false
suggest_marimo=false
suggest_pymc_testing=false

# PyMC keywords
pymc_keywords=(
  "bayesian" "pymc" "mcmc" "posterior" "inference" "arviz"
  "prior" "sampling" "divergence" "waic" "loo" "hierarchical model"
  "gaussian process" "bart" "nuts" "hmc" "nutpie" "probabilistic"
  "credible interval" "posterior predictive" "prior predictive"
  "trace" "r_hat" "rhat" "ess_bulk" "convergence" "hsgp"
  "zero.inflated" "mixture model" "multilevel" "brms"
  "logistic regression.*bayes" "poisson regression.*bayes"
  "censored" "truncated" "ordinal" "causal inference"
  "do.calculus" "pm\\.model" "pm\\.sample" "pm\\.normal"
)

for kw in "${pymc_keywords[@]}"; do
  if echo "$prompt_lower" | grep -qE "$kw"; then
    suggest_pymc=true
    break
  fi
done

# PyMC testing keywords
pymc_testing_keywords=(
  "testing pymc" "test.*pymc" "pymc.*test" "mock.sample"
  "mock_sample" "pytest.*pymc" "pymc.*pytest" "unit test.*model"
  "test fixture.*pymc" "ci.*pymc" "pymc.*ci"
)

for kw in "${pymc_testing_keywords[@]}"; do
  if echo "$prompt_lower" | grep -qE "$kw"; then
    suggest_pymc_testing=true
    break
  fi
done

# Marimo keywords
marimo_keywords=(
  "marimo" "reactive notebook" "@app\\.cell" "mo\\.ui"
  "mo\\.md" "mo\\.sql" "mo\\.state" "mo\\.stop"
  "marimo edit" "marimo run" "marimo convert"
  "mo\\.hstack" "mo\\.vstack" "mo\\.tabs"
  "wigglystuff" "anywidget"
)

for kw in "${marimo_keywords[@]}"; do
  if echo "$prompt_lower" | grep -qE "$kw"; then
    suggest_marimo=true
    break
  fi
done

# Build suggestion message
messages=()
if [ "$suggest_pymc" = true ]; then
  messages+=("Consider using the **pymc-modeling** skill for Bayesian modeling guidance.")
fi
if [ "$suggest_marimo" = true ]; then
  messages+=("Consider using the **marimo-notebook** skill for reactive notebook guidance.")
fi
if [ "$suggest_pymc_testing" = true ]; then
  messages+=("Consider using the **pymc-testing** skill for PyMC model testing guidance.")
fi

if [ ${#messages[@]} -gt 0 ]; then
  combined=$(printf '%s ' "${messages[@]}")
  # Output as JSON systemMessage for Claude
  jq -n --arg msg "$combined" '{
    "systemMessage": $msg
  }'
fi

exit 0
