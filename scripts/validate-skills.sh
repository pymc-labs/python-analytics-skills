#!/usr/bin/env bash
#
# Validate skill structure for CI.
# Checks SKILL.md frontmatter, directory naming, hooks.json, and hook scripts.
# Exit 0 = all pass, exit 1 = errors found.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
SKILLS_DIR="$ROOT_DIR/skills"

errors=0
warnings=0

pass()  { echo "  PASS: $*"; }
fail()  { echo "  FAIL: $*" >&2; errors=$((errors + 1)); }
warn_() { echo "  WARN: $*"; warnings=$((warnings + 1)); }

echo "=== Validating skills ==="
echo ""

for skill_dir in "$SKILLS_DIR"/*/; do
    skill=$(basename "$skill_dir")
    skill_file="$skill_dir/SKILL.md"

    echo "Checking $skill..."

    # SKILL.md must exist
    if [[ ! -f "$skill_file" ]]; then
        fail "Missing SKILL.md in $skill"
        continue
    fi

    # Must have YAML frontmatter
    first_line=$(head -1 "$skill_file")
    if [[ "$first_line" != "---" ]]; then
        fail "SKILL.md missing YAML frontmatter (no opening ---): $skill"
        continue
    fi

    # Extract name field
    name=$(awk '/^---$/ { if (++n == 2) exit } n == 1 && /^name:/ { sub(/^name:[[:space:]]*/, ""); print }' "$skill_file")
    if [[ -z "$name" ]]; then
        fail "Missing 'name' in frontmatter: $skill"
    elif [[ "$name" != "$skill" ]]; then
        fail "Name mismatch: frontmatter says '$name', directory is '$skill'"
    else
        pass "name field matches directory"
    fi

    # Extract description
    desc=$(awk '
        /^---$/ { if (++n == 2) exit }
        n == 1 && /^description:/ {
            sub(/^description:[[:space:]]*[>|]?[[:space:]]*/, "")
            if (length($0) > 0) print
            capture = 1
            next
        }
        capture && /^[[:space:]]/ {
            sub(/^[[:space:]]+/, "")
            printf " %s", $0
        }
        capture && /^[a-z]/ { exit }
    ' "$skill_file" 2>/dev/null | tr -s ' ' | sed 's/^ //')

    if [[ -z "$desc" ]]; then
        fail "Missing 'description' in frontmatter: $skill"
    elif [[ ${#desc} -gt 1024 ]]; then
        warn_ "Description exceeds 1024 chars: $skill (${#desc} chars)"
    else
        pass "description present (${#desc} chars)"
    fi

    echo ""
done

echo "=== Validating hooks ==="
echo ""

hooks_file="$ROOT_DIR/hooks/hooks.json"

if [[ -f "$hooks_file" ]]; then
    # Valid JSON
    if jq empty "$hooks_file" 2>/dev/null; then
        pass "hooks/hooks.json is valid JSON"
    else
        fail "hooks/hooks.json is invalid JSON"
    fi

    # Check structure: must have "hooks" key
    if jq -e '.hooks' "$hooks_file" >/dev/null 2>&1; then
        pass "hooks/hooks.json has 'hooks' key"
    else
        fail "hooks/hooks.json missing 'hooks' key"
    fi

    # Check referenced scripts exist and are executable
    scripts=$(jq -r '.. | .command? // empty' "$hooks_file" 2>/dev/null | grep -oP '\$\{CLAUDE_PLUGIN_ROOT\}/\K.*' || true)
    for script in $scripts; do
        script_path="$ROOT_DIR/$script"
        if [[ ! -f "$script_path" ]]; then
            fail "Hook references missing script: $script"
        elif [[ ! -x "$script_path" ]]; then
            fail "Hook script not executable: $script"
        else
            pass "Hook script OK: $script"
        fi
    done
else
    warn_ "No hooks/hooks.json found"
fi

echo ""

echo "=== Validating marketplace.json ==="
echo ""

marketplace="$ROOT_DIR/.claude-plugin/marketplace.json"
if [[ -f "$marketplace" ]]; then
    if jq empty "$marketplace" 2>/dev/null; then
        pass "marketplace.json is valid JSON"
    else
        fail "marketplace.json is invalid JSON"
    fi

    # Check that all skills listed in marketplace.json exist
    skill_paths=$(jq -r '.plugins[].skills[]' "$marketplace" 2>/dev/null || true)
    for sp in $skill_paths; do
        # Resolve relative path
        resolved="$ROOT_DIR/${sp#./}"
        if [[ ! -d "$resolved" ]]; then
            fail "marketplace.json references missing skill directory: $sp"
        elif [[ ! -f "$resolved/SKILL.md" ]]; then
            fail "marketplace.json skill missing SKILL.md: $sp"
        else
            pass "marketplace.json skill exists: $sp"
        fi
    done
else
    fail "Missing .claude-plugin/marketplace.json"
fi

echo ""
echo "=== Results ==="
echo "Errors:   $errors"
echo "Warnings: $warnings"

if [[ $errors -gt 0 ]]; then
    echo "FAILED"
    exit 1
else
    echo "PASSED"
    exit 0
fi
