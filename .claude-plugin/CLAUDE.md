# Python Analytics Skills - Plugin Development

## Adding a New Skill

1. Create a directory under `skills/`:
   ```
   skills/your-skill-name/
   ├── SKILL.md          # Required: main instructions
   └── references/       # Optional: detailed reference docs
   ```

2. `SKILL.md` must have YAML frontmatter with `name` and `description`:
   ```yaml
   ---
   name: your-skill-name
   description: >
     When to trigger this skill. Include keywords and task descriptions
     that help the assistant decide when to load it.
   ---
   ```

3. The `name` field must match the directory name exactly.

4. Add the skill path to `.claude-plugin/marketplace.json` in the `skills` array:
   ```json
   "skills": [
     "./skills/pymc-modeling",
     "./skills/marimo-notebook",
     "./skills/your-skill-name"
   ]
   ```

5. Add an entry to `skills.json`.

6. Run `./install.sh --validate` to verify.

## Skill Structure Requirements

- `SKILL.md` is the entry point -- keep it focused and actionable
- Put detailed reference material in `references/*.md` and link from SKILL.md
- Description should be under 1024 characters
- Include trigger phrases in the description (keywords that indicate when the skill should activate)

## Testing Locally

```bash
# Validate structure
./install.sh --validate

# Test as a plugin
claude --plugin-dir .

# Test hooks
echo '{"user_prompt": "test prompt"}' | bash hooks/suggest-skill.sh
```

## Hook Development

Hooks live in `hooks/hooks.json`. The plugin uses `UserPromptSubmit` to suggest skills.

To add a new hook:

1. Add the hook configuration to `hooks/hooks.json`
2. If using a command hook, create the script and make it executable
3. Use `${CLAUDE_PLUGIN_ROOT}` for all file paths (never hardcode)
4. Hooks must exit 0 -- failures should be handled gracefully
5. Set appropriate timeouts (default: 60s for commands, 30s for prompts)
6. Restart Claude Code to pick up hook changes (hooks load at session start)

See the [hook-development skill](https://github.com/anthropics/claude-plugins-official) for full reference.
