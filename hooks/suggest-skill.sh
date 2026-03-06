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

suggest_marimo=false

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

if [ "$suggest_marimo" = true ]; then
  jq -n '{
    "systemMessage": "Consider using the **marimo-notebook** skill for reactive notebook guidance."
  }'
fi

exit 0
