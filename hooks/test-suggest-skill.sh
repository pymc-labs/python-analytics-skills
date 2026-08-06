#!/usr/bin/env bash
# Tests for the skill-suggestion hook's keyword matching.
#
# The hook greps the user's prompt for keywords. Single common English words like
# "prior", "loo", "bart" and "nuts" must be matched as whole words — unanchored they
# also match inside "priority", "look", "bartender" and "donuts", which fires a skill
# load on prompts that have nothing to do with Bayesian modeling.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOOK="$HERE/suggest-skill.sh"
fails=0

fired() {  # prompt -> the hook's directives, empty when it stays silent
  printf '{"prompt":%s}' "$(jq -Rn --arg p "$1" '$p')" \
    | bash "$HOOK" 2>/dev/null | jq -r '.systemMessage // empty'
}

expect_silent() {
  local out; out="$(fired "$1")"
  if [ -n "$out" ]; then
    echo "FAIL: should stay silent on \"$1\""
    echo "      got: $(echo "$out" | head -1 | cut -c1-60)..."
    fails=$((fails + 1))
  fi
}

expect_loads() {  # prompt, skill name
  local out; out="$(fired "$1")"
  if ! grep -q -- "$2" <<<"$out"; then
    echo "FAIL: \"$1\" should load $2"
    fails=$((fails + 1))
  fi
}

# --- Ordinary English that merely contains a keyword as a substring -------------
expect_silent "sort by priority"
expect_silent "sort by prioritize then by score"
expect_silent "show me where stage = prioritized"
expect_silent "take a look at the board"
expect_silent "bartender staffing costs"
expect_silent "peanuts and donuts inventory"

# Out of scope, deliberately not asserted: a prompt using one of these keywords as a
# genuine standalone English word ("the prior record was wrong", "trace the bug back")
# still fires. That is semantic ambiguity, not a substring collision — a keyword matcher
# cannot separate "the prior record" from "the prior on beta" without surrounding context.
# Narrowing it would mean requiring a co-occurring modeling term, a larger design change.

# --- Genuine usage must still be caught ----------------------------------------
expect_loads "how do I set a prior on the intercept"     "pymc-modeling"
expect_loads "run a prior predictive check"              "prior-elicitation"
expect_loads "compare the models with loo"               "model-evaluation"
expect_loads "import arviz and check r_hat"              "pymc-modeling"
expect_loads "use a BART model here"                     "pymc-modeling"
expect_loads "convert this notebook to marimo"           "marimo-notebook"
expect_loads "fit_laplace from pymc_extras"              "pymc-extras"
expect_loads "mock_sample in a pytest fixture for pymc"  "pymc-testing"

if [ "$fails" -eq 0 ]; then echo "PASS"; exit 0; fi
echo "$fails check(s) failed"
exit 1
