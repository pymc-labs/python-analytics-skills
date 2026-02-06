# PyMC Skill Benchmark Suite

Measures whether the `pymc-modeling` Claude Code skill improves Bayesian model building compared to baseline (no skill). Unlike regex-based approaches, this benchmark **actually executes** generated PyMC code and extracts real MCMC diagnostics from ArviZ InferenceData objects.

## Quick Start

```bash
cd benchmark/pymc-modeling

# Install the pixi environment
pixi install

# Run the full benchmark (12 tasks x 2 conditions x 3 reps = 72 runs)
pixi run benchmark-all

# Or with LLM-as-judge code quality scoring (uses Haiku, adds ~$1 cost)
pixi run benchmark-all-llm

# Generate the report
pixi run report
```

## How It Works

The benchmark runs each task under two conditions:

- **`with_skill`** — the `pymc-modeling` skill is present at `~/.claude/skills/pymc-modeling/` and auto-loads into Claude's context
- **`no_skill`** — the skill directory is temporarily moved to `pymc-modeling.disabled/` so Claude has no access to the skill's guidance

Each run goes through a 5-phase pipeline:

```
tasks.yaml
    |
    v
[Phase 1: runner.py]  Claude CLI generates PyMC code (--print mode)
    |
    v
[Phase 1.5: extractor.py]  Extract Python scripts from Claude output
    |
    v
[Phase 2: executor.py]  Execute scripts via pixi, capture InferenceData
    |
    v
[Phase 3: diagnostics.py]  Real ArviZ diagnostics from .nc files
    |
    v
[Phase 4: scorer.py]  Automated scoring (5 criteria, 0-5 each)
    |
    v
[Phase 5: analysis.py]  Polars aggregation, effect sizes, markdown report
```

### Phase 1: Code Generation

Invokes `claude --print --output-format json --model sonnet` with a task prompt prepended by a standard preamble requesting a self-contained `.py` script that saves `results.nc`. Skills auto-load from `~/.claude/skills/` — no `--append-system-prompt` needed.

### Phase 1.5: Code Extraction

Parses fenced Python code blocks from Claude's response. Handles edge cases like multiple code blocks, marimo notebook output, and missing code.

### Phase 2: Script Execution

Runs each extracted script in an isolated temp directory via `pixi run python script.py` with a 10-minute timeout. Environment variables (`MPLBACKEND=Agg`, `PYTENSOR_FLAGS=device=cpu`, `OMP_NUM_THREADS=4`) ensure deterministic headless execution. Checks for `results.nc` output.

### Phase 3: MCMC Diagnostics

Loads actual InferenceData from `.nc` files and computes real diagnostics via ArviZ:

| Metric | Source | Description |
|--------|--------|-------------|
| r_hat (max, mean) | `az.summary()` | Gelman-Rubin convergence statistic |
| ESS bulk (min, median) | `az.summary()` | Effective sample size for bulk of posterior |
| ESS tail (min, median) | `az.summary()` | Effective sample size for tail quantiles |
| Divergences (count, %) | `sample_stats` | NUTS divergent transitions |
| BFMI (min, mean) | `az.bfmi()` | Bayesian fraction of missing information |
| Max tree depth hits | `sample_stats` | Saturated NUTS tree depth |
| Model structure | InferenceData groups | Which groups present (posterior, prior, predictive, log_likelihood) |

### Phase 4: Scoring

Five automated criteria, each scored 0-5 (total 25):

| Criterion | Method | What it measures |
|-----------|--------|------------------|
| **Execution** | Automated | Did the code run? Produce InferenceData? (0=extraction failed, 5=clean run with .nc) |
| **Convergence** | Automated | r_hat < 1.01, ESS > 400, divergences = 0 → score 5 |
| **Diagnostic completeness** | Code regex | Presence of `az.summary`, trace/rank plots, PPC, LOO, energy plots |
| **Statistical quality** | Automated | Has posterior, converged, prior/posterior predictive, log_likelihood |
| **Code quality** | LLM-as-judge or regex | coords/dims, nutpie, weakly informative priors, random seed, parameterization |

Code quality scoring has two modes:
- **LLM-as-judge** (`benchmark-all-llm`): Uses `claude --print --model haiku` to evaluate code against PyMC best practices
- **Automated fallback** (`benchmark-all` / `--no-llm`): Regex checks for coords/dims, nutpie, seed, comments

### Phase 5: Analysis

Loads all score JSONs into a Polars DataFrame and computes:
- Summary statistics by condition (mean, std, n)
- Breakdown by task and by tier
- Cohen's d effect sizes (with_skill - no_skill)
- Failure mode categorization by condition
- Cost analysis (tokens, USD)

## Task Suite

12 tasks across 3 difficulty tiers. Tier 2-3 tasks using the GSS 2022 dataset are designed as key discriminators — the skill explicitly covers these patterns while a naive user would likely make common mistakes.

### Tier 1: Basic (T1-T4)

| Task | Name | Dataset | Challenge |
|------|------|---------|-----------|
| T1 | Linear Regression | mtcars.csv | Basic model structure, priors, diagnostics |
| T2 | Logistic Regression | titanic.csv | GLM, Bernoulli likelihood, sigmoid link |
| T3 | Poisson Regression | soccer_goals.csv | Count data, overdispersion check |
| T4 | Prior Predictive Check | (inline) | Critique bad priors, suggest improvements |

### Tier 2: Intermediate (T5-T8)

| Task | Name | Dataset | Challenge |
|------|------|---------|-----------|
| T5 | Hierarchical (Non-Centered) | 8-schools (inline) | Non-centered parameterization, divergence diagnosis |
| T6 | Ordinal Regression | gss_2022.csv | `pm.OrderedLogistic`, ordered cutpoints |
| T7 | Time Series AR(2) | airline_passengers.csv | Autoregressive model, forecasting with uncertainty |
| T8 | Missing Data Handling | gss_2022.csv | Bayesian imputation of ~30K missing values |

### Tier 3: Advanced (T9-T12)

| Task | Name | Dataset | Challenge |
|------|------|---------|-----------|
| T9 | Gaussian Process (HSGP) | mauna_loa_co2.csv | HSGP approximation, periodic + trend kernels |
| T10 | Horseshoe Variable Selection | gss_2022.csv | Regularized horseshoe prior, 13 predictors |
| T11 | Monotonic Ordinal Predictors | gss_2022.csv | Dirichlet simplex for ordinal predictor effects |
| T12 | Model Comparison Pipeline | regression_comparison.csv | LOO-CV, Pareto k diagnostics, 3-model comparison |

### Why the GSS Tasks Are Good Discriminators

The `pymc-modeling` skill explicitly covers `OrderedLogistic` (specialized likelihoods reference), horseshoe/shrinkage priors (priors reference), and missing data patterns (troubleshooting reference). Without the skill, Claude is likely to:

- **T6**: Treat the ordinal outcome as continuous (incorrect model)
- **T8**: Drop rows with missing data (losing most of the dataset)
- **T10**: Use flat priors with 13 predictors (overfitting)
- **T11**: Treat ordinal predictors as continuous (violating monotonicity)

## Pixi Tasks

| Task | Description |
|------|-------------|
| `pixi run benchmark-all` | Full pipeline, all tasks, 3 reps, automated scoring (~5h, ~$11 API) |
| `pixi run benchmark-all-llm` | Same with LLM-as-judge code quality scoring (~$12 API) |
| `pixi run pipeline` | Base pipeline command (pass extra args, e.g. `--task T1 --condition with_skill`) |
| `pixi run run-all` | Phase 1 only: Claude code generation |
| `pixi run extract-all` | Phase 1.5 only: code extraction |
| `pixi run execute-all` | Phase 2 only: script execution |
| `pixi run diagnose-all` | Phase 3 only: ArviZ diagnostics |
| `pixi run score-all` | Phase 4 only: scoring |
| `pixi run report` | Phase 5: generate markdown report |
| `pixi run summary` | Print summary table to stdout |
| `pixi run effects` | Print effect sizes to stdout |
| `pixi run status` | Show completion matrix (what's been run) |
| `pixi run cleanup` | Restore skill directory if left disabled |
| `pixi run list-tasks` | List all 12 tasks with metadata |

### Resuming Interrupted Runs

The benchmark automatically skips scenarios that already have raw results in `results/raw/`. This means you can safely re-run `pixi run benchmark-all` after an interruption — it will pick up where it left off. Use `--force` to re-run scenarios that already have results:

```bash
# Resume an interrupted benchmark (skips completed scenarios)
pixi run benchmark-all

# Force re-run of all scenarios, ignoring existing results
pixi run pipeline --all --reps 3 --no-llm --force
```

### Running Individual Tasks

```bash
# Single task, single condition, single rep
pixi run pipeline --task T1 --condition with_skill --reps 1

# All tasks for one condition
pixi run pipeline --all --condition no_skill --reps 3

# Dry run (print prompts without calling Claude)
pixi run pipeline --all --dry-run

# Custom model and budget
pixi run pipeline --task T9 --condition with_skill --model sonnet --max-budget 2.0

# Force re-run a single task even if results exist
pixi run pipeline --task T1 --condition no_skill --reps 1 --force
```

## Skill Control Mechanism

The `no_skill` condition works by temporarily moving the skill directory:

```
~/.claude/skills/pymc-modeling/  →  ~/.claude/skills/pymc-modeling.disabled/
```

Crash safety measures:
- `try...finally` in the context manager
- `atexit` handler for backup restoration
- `SIGINT`/`SIGTERM` signal handlers
- PID lock file (`.benchmark-lock`) to detect stale backups
- `pixi run cleanup` command for manual restoration

## Output Structure

```
results/
├── raw/              # Claude CLI JSON output (tokens, cost, response text)
├── code/             # Extracted .py scripts
├── execution/        # Execution results (.json) + InferenceData (.nc)
├── diagnostics/      # MCMC diagnostic JSONs (r_hat, ESS, divergences, etc.)
├── scores/           # Final scored JSONs (5 criteria + metrics)
└── reports/          # Markdown benchmark report
```

## Experimental Design

| Parameter | Value |
|-----------|-------|
| Conditions | `no_skill` vs `with_skill` |
| Tasks | 12 (4 basic, 4 intermediate, 4 advanced) |
| Replications | 3 per task per condition |
| Total runs | 72 |
| Claude model | Sonnet (configurable) |
| Budget cap | $1.00 per Claude call |
| Estimated API cost | ~$11 (without LLM judge) |
| Execution timeout | 10 minutes per script |
| Estimated wall time | ~5 hours sequential |

## Dependencies

Managed by pixi (conda-forge):

- Python >= 3.11
- PyMC >= 5.10
- ArviZ >= 0.18
- nutpie >= 0.12
- PyMC-BART >= 0.6
- Polars >= 1.0
- NumPy, SciPy, Matplotlib, PyYAML

Also requires `claude` CLI installed and authenticated.
