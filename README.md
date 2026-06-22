# Python Analytics Skills

A plugin for Claude Code and other AI coding platforms providing [Agent Skills](https://agentskills.io) for Bayesian modeling and reactive Python notebooks. Packages specialized knowledge for PyMC and marimo into skills that Claude loads on demand.

## Skills

| Skill | Description |
|-------|-------------|
| [pymc-modeling](skills/pymc-modeling/) | Bayesian statistical modeling with PyMC 6+, PyTensor 3+, and ArviZ 1.1+. Covers model specification, inference, diagnostics, hierarchical models, GLMs, GPs/HSGPs, BART, time series, mixtures, causal models, priors, and custom likelihoods. |
| [prior-elicitation](skills/prior-elicitation/) | Prior selection, prior predictive checks, constrained priors, PreliZ workflows, expert priors, weakly informative priors, and prior sensitivity analysis. |
| [pymc-extras](skills/pymc-extras/) | pymc-extras guidance for splines, distributional regression, R2D2/horseshoe priors, marginalization, and Laplace approximation. |
| [pymc-testing](skills/pymc-testing/) | Testing PyMC models with pytest. Covers `pymc.testing.mock_sample`, fixtures, structure-only tests, and slow posterior inference tests. |
| [model-evaluation](skills/model-evaluation/) | Bayesian model comparison and predictive evaluation with ArviZ 1.1 LOO/ELPD, stacking, model averaging, Bayes factors, and Pareto-k diagnostics. |
| [marimo-notebook](skills/marimo-notebook/) | Reactive Python notebooks with marimo. Covers CLI usage, `@app.cell` notebooks, UI components, layout, SQL integration, caching, state management, notebook conversion, templates, and wigglystuff widgets. |

## Installation

### Via npx (Recommended — works across agents)

```bash
npx skills add pymc-labs/python-analytics-skills
```

One command, works with Claude Code, Cursor, Gemini CLI, and 15+ other agents.

### As a Claude Code Plugin

Two-step process using Claude Code slash commands:

```bash
/plugin marketplace add pymc-labs/python-analytics-skills
/plugin install analytics@pymc-labs-python-analytics-skills
```

Installs all skills plus the keyword-suggestion hook. Supports `/plugin update` for future updates.

### For Pi / Oh My Pi

Oh My Pi can install this repository as a marketplace plugin using the Claude-compatible `.claude-plugin/marketplace.json` catalog:

```text
/marketplace add pymc-labs/python-analytics-skills
/marketplace install analytics@python-analytics-skills
```

CLI equivalent:

```bash
omp plugin marketplace add pymc-labs/python-analytics-skills
omp plugin install analytics@python-analytics-skills
```

For a project-local install, add `--scope project` to the install command.

### Manual Installation

```bash
git clone https://github.com/pymc-labs/python-analytics-skills.git
cd python-analytics-skills
./install.sh claude                    # Claude Code
./install.sh all                       # All platforms
./install.sh claude -- pymc-modeling   # Specific skill only
```

### Utility Commands

```bash
# List available skills with descriptions
./install.sh --list

# Validate skill structure
./install.sh --validate
./scripts/validate-skills.sh
```

## Benchmark

The repo includes a PyMC skill benchmark in [`benchmark/pymc-modeling/`](benchmark/pymc-modeling/). It compares Claude outputs with and without `skills/pymc-modeling/SKILL.md` injected via `--append-system-prompt`.

Benchmark contents:

- `tasks.yaml` — five Bayesian modeling tasks
- `src/` — runner, scorer, analysis, and CLI modules
- `tests/` — pytest coverage for the benchmark harness
- `data/` — small input datasets used by tasks
- `scripts/prepare_data.py` — deterministic data preparation
- `pixi.toml` / `pixi.lock` — benchmark environment

Run benchmark commands from the benchmark directory:

```bash
cd benchmark/pymc-modeling
pixi run validate
pixi run test
pixi run run-all
pixi run score-all
pixi run analyze
```

Generated benchmark outputs such as `results/`, `.nc` files, figures, and archived runs are not package source.

## Platform Support

| Platform | Install Location | Auto-Discovered |
|----------|-----------------|-----------------|
| Claude Code | `~/.claude/skills/` | Yes |
| OpenCode | `~/.config/opencode/skills/` | Yes |
| Gemini CLI | `~/.gemini/skills/` | Yes |
| Cursor | `~/.cursor/skills/` | Yes |
| VS Code Copilot | `~/.copilot/skills/` | Yes |

## Project Structure

```text
python-analytics-skills/
├── .claude-plugin/
│   ├── marketplace.json    # Plugin registry metadata
│   ├── plugin.json         # Plugin configuration
│   └── CLAUDE.md           # Claude plugin guidance
├── benchmark/
│   └── pymc-modeling/      # PyMC skill benchmark harness
├── hooks/
│   ├── hooks.json          # Hook configuration
│   └── suggest-skill.sh    # Keyword-based skill suggestion
├── scripts/
│   └── validate-skills.sh  # Repository validation
├── skills/
│   ├── pymc-modeling/      # Main PyMC skill plus 13 reference docs
│   ├── prior-elicitation/  # Prior workflow references
│   ├── pymc-extras/        # pymc-extras references
│   ├── pymc-testing/       # PyMC testing skill plus testing patterns
│   ├── model-evaluation/   # LOO/model comparison references
│   └── marimo-notebook/    # Marimo references, templates, and conversion utilities
├── install.sh              # Multi-platform installer
├── package.json            # npm package metadata
└── skills.json             # Skills registry
```

## Hooks

The plugin includes a `UserPromptSubmit` hook that suggests relevant skills when it detects keywords in your prompt:

- **PyMC keywords**: bayesian, pymc, mcmc, posterior, inference, arviz, prior, sampling, divergence, hierarchical model, gaussian process, bart, causal inference, etc.
- **PyMC companion keywords**: PreliZ, prior elicitation, model comparison, LOO, ELPD, pymc-extras, splines, testing PyMC, mock sampling, etc.
- **PyMC testing keywords**: testing PyMC, pytest, mock sampling, test fixtures, CI/CD for Bayesian models, etc.
- **Marimo keywords**: marimo, reactive notebook, `@app.cell`, `mo.ui`, notebook conversion, wigglystuff, etc.

Test the hook directly:

```bash
echo '{"user_prompt": "bayesian model"}' | bash hooks/suggest-skill.sh
```

## Troubleshooting

**Skill not loading:**

1. Verify the skill directory exists with a valid `SKILL.md`
2. Run `./install.sh --validate` to check structure
3. For Claude Code plugins, check `claude --debug` for hook/skill loading errors

**Hook not firing:**

1. Hooks load at session start -- restart Claude Code after changes
2. Use `/hooks` in Claude Code to see loaded hooks
3. Test the hook script directly with the command above

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on adding new skills.

## License

MIT License. See [LICENSE](LICENSE) for details.
