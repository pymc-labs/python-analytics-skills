#!/usr/bin/env bash
#
# Install python-analytics-skills for AI coding assistants
#
# Usage:
#   ./install.sh <platform...> [-- skill...]
#   ./install.sh --list
#   ./install.sh --validate
#
# Platforms:
#   claude   - Install to ~/.claude/skills/
#   opencode - Install to ~/.config/opencode/skills/
#   gemini   - Install to ~/.gemini/skills/
#   cursor   - Install to ~/.cursor/skills/
#   copilot  - Install to ~/.copilot/skills/
#   all      - Install to all platform directories
#
# Flags:
#   --list      - List available skills with descriptions
#   --validate  - Validate skill structure (SKILL.md frontmatter)
#   --dry-run   - Show what would be installed without doing it
#
# Examples:
#   ./install.sh claude                       # All skills to Claude Code
#   ./install.sh gemini -- pymc-modeling      # Specific skill to Gemini CLI
#   ./install.sh claude cursor opencode       # All skills to multiple platforms
#   ./install.sh all                          # All skills to all platforms
#   ./install.sh --list                       # Show available skills
#   ./install.sh --validate                   # Check skill structure

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SKILLS_DIR="$SCRIPT_DIR/skills"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m'

info()    { echo -e "${BLUE}[INFO]${NC} $*"; }
success() { echo -e "${GREEN}[OK]${NC} $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC} $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*" >&2; }

DRY_RUN=false

usage() {
    cat << EOF
Usage: ./install.sh <platform...> [-- skill...]
       ./install.sh --list | --validate

Platforms:
  claude    ~/.claude/skills/
  opencode  ~/.config/opencode/skills/
  gemini    ~/.gemini/skills/
  cursor    ~/.cursor/skills/
  copilot   ~/.copilot/skills/
  all       All of the above

Flags:
  --list      List available skills with descriptions
  --validate  Validate skill structure
  --dry-run   Show what would be installed without doing it

Examples:
  ./install.sh claude                    # All skills to Claude Code
  ./install.sh gemini -- pymc-modeling   # Specific skill to Gemini CLI
  ./install.sh all                       # All skills to all platforms
  ./install.sh --list                    # Show available skills
  ./install.sh --validate                # Validate skill structure
EOF
    exit 1
}

get_target_dir() {
    local platform="$1"
    case "$platform" in
        claude)   echo "$HOME/.claude/skills" ;;
        opencode) echo "$HOME/.config/opencode/skills" ;;
        gemini)   echo "$HOME/.gemini/skills" ;;
        cursor)   echo "$HOME/.cursor/skills" ;;
        copilot)  echo "$HOME/.copilot/skills" ;;
        *) error "Unknown platform: $platform"; exit 1 ;;
    esac
}

is_platform() {
    case "$1" in
        claude|opencode|gemini|cursor|copilot|all) return 0 ;;
        *) return 1 ;;
    esac
}

# Extract name from SKILL.md frontmatter
extract_name() {
    local skill_file="$1"
    grep -m1 '^name:' "$skill_file" 2>/dev/null | sed 's/^name:[[:space:]]*//'
}

# Extract description from SKILL.md frontmatter
extract_description() {
    local skill_file="$1"
    awk '
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
    ' "$skill_file" 2>/dev/null | tr -s ' ' | sed 's/^ //' | cut -c1-120
}

# List available skills with descriptions
list_skills() {
    echo -e "${BOLD}Available skills:${NC}"
    echo ""
    for skill_dir in "$SKILLS_DIR"/*/; do
        local skill
        skill=$(basename "$skill_dir")
        local skill_file="$skill_dir/SKILL.md"
        if [[ -f "$skill_file" ]]; then
            local desc
            desc=$(extract_description "$skill_file")
            echo -e "  ${GREEN}${skill}${NC}"
            echo "    $desc"
            echo ""
        fi
    done
}

# Validate skill structure
validate_skills() {
    local errors=0

    echo -e "${BOLD}Validating skills...${NC}"
    echo ""

    for skill_dir in "$SKILLS_DIR"/*/; do
        local skill
        skill=$(basename "$skill_dir")
        local skill_file="$skill_dir/SKILL.md"

        echo -e "  Checking ${BOLD}$skill${NC}..."

        # Check SKILL.md exists
        if [[ ! -f "$skill_file" ]]; then
            error "  Missing SKILL.md in $skill"
            errors=$((errors + 1))
            continue
        fi

        # Check name field
        local name
        name=$(extract_name "$skill_file")
        if [[ -z "$name" ]]; then
            error "  Missing 'name' in frontmatter: $skill"
            errors=$((errors + 1))
        elif [[ "$name" != "$skill" ]]; then
            warn "  Name mismatch: frontmatter says '$name', directory is '$skill'"
        fi

        # Check description field
        local desc
        desc=$(extract_description "$skill_file")
        if [[ -z "$desc" ]]; then
            error "  Missing 'description' in frontmatter: $skill"
            errors=$((errors + 1))
        elif [[ ${#desc} -gt 1024 ]]; then
            warn "  Description exceeds 1024 chars: $skill (${#desc} chars)"
        fi

        success "  $skill OK"
    done

    # Validate hooks
    local hooks_file="$SCRIPT_DIR/hooks/hooks.json"
    echo ""
    echo -e "  Checking ${BOLD}hooks${NC}..."

    if [[ -f "$hooks_file" ]]; then
        if jq empty "$hooks_file" 2>/dev/null; then
            success "  hooks/hooks.json is valid JSON"
        else
            error "  hooks/hooks.json is invalid JSON"
            errors=$((errors + 1))
        fi

        # Check referenced scripts exist and are executable
        local scripts
        scripts=$(jq -r '.. | .command? // empty' "$hooks_file" 2>/dev/null | grep -oP '\$\{CLAUDE_PLUGIN_ROOT\}/\K.*' || true)
        for script in $scripts; do
            local script_path="$SCRIPT_DIR/$script"
            if [[ ! -f "$script_path" ]]; then
                error "  Hook references missing script: $script"
                errors=$((errors + 1))
            elif [[ ! -x "$script_path" ]]; then
                warn "  Hook script not executable: $script"
            else
                success "  Hook script OK: $script"
            fi
        done
    else
        warn "  No hooks/hooks.json found"
    fi

    echo ""
    if [[ $errors -eq 0 ]]; then
        success "All validations passed!"
        return 0
    else
        error "$errors error(s) found"
        return 1
    fi
}

install_skill() {
    local skill="$1"
    local target_dir="$2"
    local skill_src="$SKILLS_DIR/$skill"
    local skill_dst="$target_dir/$skill"

    if [[ ! -d "$skill_src" ]]; then
        error "Skill not found: $skill"
        return 1
    fi

    if $DRY_RUN; then
        info "[dry-run] Would install $skill -> $skill_dst"
        return 0
    fi

    mkdir -p "$target_dir"

    if [[ -d "$skill_dst" ]]; then
        warn "Overwriting existing skill: $skill_dst"
        rm -rf "$skill_dst"
    fi

    cp -r "$skill_src" "$skill_dst"
    success "Installed $skill -> $skill_dst"
}

get_all_skills() {
    find "$SKILLS_DIR" -maxdepth 1 -mindepth 1 -type d -exec basename {} \; 2>/dev/null | sort
}

main() {
    if [[ $# -eq 0 ]]; then
        usage
    fi

    # Handle flags
    case "${1:-}" in
        --list)
            list_skills
            exit 0
            ;;
        --validate)
            validate_skills
            exit $?
            ;;
        --help|-h)
            usage
            ;;
    esac

    local platforms=()
    local skills=()
    local parsing_skills=false

    for arg in "$@"; do
        case "$arg" in
            --dry-run) DRY_RUN=true; continue ;;
            --) parsing_skills=true; continue ;;
        esac

        if $parsing_skills; then
            skills+=("$arg")
        elif is_platform "$arg"; then
            if [[ "$arg" == "all" ]]; then
                platforms=(claude opencode gemini cursor copilot)
            else
                platforms+=("$arg")
            fi
        else
            skills+=("$arg")
        fi
    done

    if [[ ${#platforms[@]} -eq 0 ]]; then
        error "No valid platform specified"
        echo ""
        usage
    fi

    # Deduplicate platforms
    local unique_platforms=()
    for p in "${platforms[@]}"; do
        local found=false
        for up in "${unique_platforms[@]:-}"; do
            if [[ "$p" == "$up" ]]; then
                found=true
                break
            fi
        done
        if ! $found; then
            unique_platforms+=("$p")
        fi
    done
    platforms=("${unique_platforms[@]}")

    if [[ ${#skills[@]} -eq 0 ]]; then
        mapfile -t skills < <(get_all_skills)
    fi

    if [[ ${#skills[@]} -eq 0 ]]; then
        error "No skills found in $SKILLS_DIR"
        exit 1
    fi

    if $DRY_RUN; then
        info "[dry-run] No changes will be made"
    fi

    info "Skills to install: ${skills[*]}"
    info "Target platforms: ${platforms[*]}"
    echo ""

    for p in "${platforms[@]}"; do
        local target_dir
        target_dir=$(get_target_dir "$p")
        info "Installing to $p ($target_dir)..."

        for skill in "${skills[@]}"; do
            install_skill "$skill" "$target_dir"
        done
        echo ""
    done

    if $DRY_RUN; then
        info "[dry-run] Complete — no changes were made"
    else
        success "Installation complete!"
    fi
}

main "$@"
