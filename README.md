# Python Analytics Skills

A plugin for Claude Code and other AI coding platforms providing [Agent Skills](https://agentskills.io) for Bayesian modeling and reactive notebooks. Packages specialized knowledge for PyMC and marimo into skills that Claude loads on-demand.

## Skills

| Skill | Description |
|-------|-------------|
| [pymc-modeling](skills/pymc-modeling/) | Bayesian statistical modeling with PyMC v5+. Covers model specification, MCMC inference (nutpie, NumPyro), ArviZ diagnostics, hierarchical models, GLMs, GPs, BART, time series, and more. |
| [marimo-notebook](skills/marimo-notebook/) | Reactive Python notebooks with marimo. Covers CLI, UI components, layout, SQL integration, caching, state management, and wigglystuff widgets. |

## Installation

### As a Claude Code Plugin (Recommended)

```bash
claude plugin marketplace add /path/to/python-analytics-skills
```

Or install from GitHub:

```bash
claude plugin marketplace add https://github.com/pymc-labs/python-analytics-skills
```

### Via npx

```bash
npx skills add @pymc-labs/python-analytics-skills
```

### Manual Installation to Platform Skills Directories

Use the install script to copy skills to any supported platform:

```bash
git clone https://github.com/pymc-labs/python-analytics-skills.git
cd python-analytics-skills

# Install to a specific platform
./install.sh claude              # Claude Code (~/.claude/skills/)
./install.sh opencode            # OpenCode (~/.config/opencode/skills/)
./install.sh gemini              # Gemini CLI (~/.gemini/skills/)
./install.sh cursor              # Cursor (~/.cursor/skills/)
./install.sh copilot             # VS Code Copilot (~/.copilot/skills/)
./install.sh all                 # All platforms

# Install a specific skill
./install.sh claude -- pymc-modeling

# Preview without changes
./install.sh --dry-run claude
```

### Utility Commands

```bash
# List available skills with descriptions
./install.sh --list

# Validate skill structure
./install.sh --validate
```

## Platform Support

| Platform | Install Location | Auto-Discovered |
|----------|-----------------|-----------------|
| Claude Code | `~/.claude/skills/` | Yes |
| OpenCode | `~/.config/opencode/skills/` | Yes |
| Gemini CLI | `~/.gemini/skills/` | Yes |
| Cursor | `~/.cursor/skills/` | Yes |
| VS Code Copilot | `~/.copilot/skills/` | Yes |

## Plugin Structure

```
python-analytics-skills/
├── .claude-plugin/
│   ├── marketplace.json    # Plugin registry metadata
│   └── plugin.json         # Plugin configuration
├── skills/
│   ├── pymc-modeling/
│   │   ├── SKILL.md        # Main skill instructions
│   │   └── references/     # 12 detailed reference docs
│   └── marimo-notebook/
│       ├── SKILL.md        # Main skill instructions
│       ├── references/     # 4 reference docs
│       ├── assets/         # Notebook templates
│       └── scripts/        # Conversion utilities
├── hooks/
│   ├── hooks.json          # Hook configuration
│   └── suggest-skill.sh    # Keyword-based skill suggestion
├── install.sh              # Multi-platform installer
├── package.json            # npm package metadata
└── skills.json             # Skills registry
```

## Hooks

The plugin includes a `UserPromptSubmit` hook that suggests relevant skills when it detects keywords in your prompt:

- **PyMC keywords**: bayesian, pymc, mcmc, posterior, inference, arviz, prior, sampling, divergence, hierarchical model, gaussian process, bart, etc.
- **Marimo keywords**: marimo, reactive notebook, @app.cell, mo.ui, etc.

## Troubleshooting

**Skill not loading:**

1. Verify the skill directory exists with a valid `SKILL.md`
2. Run `./install.sh --validate` to check structure
3. For Claude Code plugins, check `claude --debug` for hook/skill loading errors

**Hook not firing:**

1. Hooks load at session start -- restart Claude Code after changes
2. Use `/hooks` in Claude Code to see loaded hooks
3. Test the hook script directly: `echo '{"user_prompt": "bayesian model"}' | bash hooks/suggest-skill.sh`

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on adding new skills.

## License

MIT License. See [LICENSE](LICENSE) for details.
