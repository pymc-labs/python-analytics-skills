# Contributing

## Adding a New Skill

1. Create the skill directory:
   ```bash
   mkdir -p skills/your-skill-name/references
   ```

2. Create `skills/your-skill-name/SKILL.md` with YAML frontmatter:
   ```yaml
   ---
   name: your-skill-name
   description: >
     When to trigger this skill. Include specific keywords and task
     descriptions so the assistant knows when to load it.
   ---

   # Your Skill Name

   Main instructions go here...
   ```

3. Add reference documents in `references/` for detailed content that SKILL.md links to.

4. Register the skill in `.claude-plugin/marketplace.json`:
   ```json
   "skills": [
     "./skills/pymc-modeling",
     "./skills/marimo-notebook",
     "./skills/your-skill-name"
   ]
   ```

5. Add an entry to `skills.json`.

6. Validate:
   ```bash
   ./scripts/validate-skills.sh
   ```

## Naming Conventions

- Skill directory names: lowercase, hyphen-separated (e.g., `pymc-modeling`)
- The `name` field in SKILL.md frontmatter must match the directory name exactly
- Reference files: lowercase, underscores OK (e.g., `references/custom_models.md`)

## Skill Writing Guidelines

- Keep SKILL.md focused and actionable — it's what the assistant loads first
- Put detailed reference material in `references/` and link from SKILL.md
- Include code examples with correct, tested patterns
- Description should be under 1024 characters
- Include trigger phrases that help the assistant decide when to load the skill

## Running Validation Locally

```bash
# Full validation (skills, hooks, marketplace.json)
./scripts/validate-skills.sh

# Quick check with install script
./install.sh --validate

# Test hooks
echo '{"user_prompt": "your test prompt"}' | bash hooks/suggest-skill.sh
```

## Pull Request Checklist

- [ ] `./scripts/validate-skills.sh` passes
- [ ] Skill registered in `marketplace.json` and `skills.json`
- [ ] SKILL.md has valid frontmatter with `name` and `description`
- [ ] Name field matches directory name
- [ ] Code examples are correct and tested
