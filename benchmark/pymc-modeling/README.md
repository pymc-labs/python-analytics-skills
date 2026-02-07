# PyMC Skill Benchmark

Measures whether the `pymc-modeling` skill improves the quality of Bayesian models that Claude builds. The benchmark compares model outputs **with** and **without** `SKILL.md` injected via `--append-system-prompt` across five tasks chosen to exercise specific skill coverage areas.

## Design

### Conditions

| Condition | Description |
|-----------|-------------|
| `no_skill` | Claude with base knowledge only |
| `with_skill` | Claude with `SKILL.md` (~18KB) injected via `--append-system-prompt` |

Both conditions use identical Claude CLI flags — the only difference is the system prompt injection. The Skill tool is excluded from the tool list, slash commands are disabled, and each run uses an isolated working directory outside the repo to prevent plugin hooks or project-level settings from leaking in.

### Tasks

Every task targets something the skill explicitly teaches. No basic models — Claude handles linear/logistic regression fine without help.

| ID | Task | What the skill teaches | Data |
|----|------|------------------------|------|
| T1 | Hierarchical model (8 schools) | Non-centered parameterization, divergence diagnosis | Embedded in prompt |
| T2 | Ordinal regression | `OrderedLogistic`, ordered transform, variable/dim name conflicts | `gss_2022_clean.csv` |
| T3 | Model comparison (LOO-CV) | `pm.compute_log_likelihood()` after nutpie, correct `az.compare()` columns | `synthetic/regression_comparison.csv` |
| T4 | Gaussian process (Mauna Loa CO2) | HSGP + HSGPPeriodic, InverseGamma lengthscale priors, `m`/`c` selection | `mauna_loa_co2.csv` |
| T5 | Sparse variable selection | Regularized horseshoe, `target_accept=0.99`, R2D2 alternative | `gss_2022_clean.csv` |

Prompts describe the *statistical problem* only — they never mention PyMC-specific patterns. The skill's value is knowing which PyMC tools and patterns to use.

### Scoring Rubric (0-20)

Each run is scored on four criteria, each worth 0-5 points:

| Criterion | Method | What it measures |
|-----------|--------|------------------|
| **Model produced** | Automated | Does `results.nc` exist with a posterior group? How complete is the InferenceData? |
| **Convergence** | Automated | R-hat, ESS, divergence count — computed directly from the `.nc` file |
| **Model appropriateness** | LLM judge (Haiku) | Is the model structure correct for this problem? Task-specific rubric |
| **Best practices** | Automated (regex) | coords/dims, nutpie, random_seed, log_likelihood, task-specific patterns |

The LLM judge uses Haiku with a structured rubric per task. If the judge fails (timeout, API error), scoring falls back to regex-based pattern matching.

### Execution Model

Claude runs in `--print` mode with tools (`Bash`, `Read`, `Write`, `Glob`, `Grep`). Despite being `--print` mode, Claude uses tools iteratively — it writes `model.py`, runs it via Bash, sees errors, and fixes them. Each run is capped at $2.00 and 10 minutes wall-clock time.

### Analysis

Cohen's d effect sizes comparing conditions, broken down by task. Paired comparison (same task, different condition) gives maximum statistical power. Reports are generated in Markdown with per-task breakdowns.

## Setup

### Prerequisites

- [pixi](https://pixi.sh) package manager
- Claude CLI (`claude`) installed and authenticated
- Linux x86_64 (the pixi environment is pinned to `linux-64`)

### Install

```bash
cd benchmark/pymc-modeling
pixi install
```

### Prepare Data

Cleans and subsamples the raw datasets for fast MCMC sampling (~2 min per task with nutpie):

```bash
pixi run prepare-data
```

This produces:
- `data/gss_2022_clean.csv` — 487 rows (13 columns, nulls dropped)
- `data/mauna_loa_co2.csv` — 396 rows (every other month)
- `data/synthetic/regression_comparison.csv` — 150 rows (unchanged)

## Running the Benchmark

### Step 1: Validate

The validation gate runs T1 (simplest task) in both conditions and checks every assumption: tool execution, artifact production, scoring, and working directory isolation. **This must pass before running the full suite.**

```bash
pixi run validate
```

Checks performed:
- Both runs completed with multiple turns (tools were used)
- No `Skill` tool calls in either condition
- `model.py` produced (>100 bytes)
- `results.nc` loads with a posterior group
- Scorer returns scores for all 4 criteria
- Working directory has no `.claude-plugin/` or `hooks/` contamination

### Step 2: Full Suite

5 tasks x 2 conditions x 3 replications = 30 runs. Conditions are interleaved for fairness.

```bash
pixi run python -m src.cli run --all
```

Options:

```
--reps N       Replications per task/condition (default: 3)
--force        Overwrite cached results
--resume       Re-run only missing or failed runs
--task T1_hierarchical   Run a single task
```

Results are cached in `results/runs/`. Re-running without `--force` skips completed runs.

### Step 3: Score

```bash
pixi run python -m src.cli score --all
```

Reads `model.py` and `results.nc` from each run directory, computes all four scoring criteria, and saves score JSONs to `results/scores/`.

### Step 4: Analyze

```bash
pixi run python -m src.cli analyze
```

Generates a Markdown report with:
- Mean scores by task and condition
- Cohen's d effect sizes per task and criterion
- Overall effect size across all tasks

Output is saved to `results/analysis/report.md`.

## CLI Reference

All commands run from the `benchmark/pymc-modeling/` directory.

| Command | Description |
|---------|-------------|
| `pixi run validate` | Validation gate (T1 only, both conditions) |
| `pixi run python -m src.cli run --all` | Run all tasks |
| `pixi run python -m src.cli run --task T2_ordinal` | Run a single task |
| `pixi run python -m src.cli run --all --resume` | Resume interrupted suite |
| `pixi run python -m src.cli score --all` | Score all completed runs |
| `pixi run python -m src.cli analyze` | Generate analysis report |
| `pixi run list-tasks` | List available tasks |
| `pixi run status` | Show run/score completion status |
| `pixi run test` | Run unit tests |

## Project Structure

```
benchmark/pymc-modeling/
├── pixi.toml                 # Dependencies and task aliases
├── conftest.py               # pytest configuration (MPLBACKEND=Agg)
├── tasks.yaml                # 5 task definitions with prompts and rubrics
├── scripts/
│   └── prepare_data.py       # Dataset cleaning and subsampling
├── src/
│   ├── runner.py             # Launches Claude, manages working dirs, caching
│   ├── scorer.py             # 4-criterion rubric with LLM judge
│   ├── analysis.py           # Cohen's d, Polars reports
│   └── cli.py                # CLI entry point (run, score, analyze, validate)
├── data/
│   ├── gss_2022.csv          # Raw GSS 2022 survey data
│   ├── gss_2022_clean.csv    # Cleaned subset (487 rows)
│   ├── mauna_loa_co2.csv     # CO2 measurements (396 rows)
│   └── synthetic/
│       └── regression_comparison.csv  # Synthetic regression data (150 rows)
├── results/                  # Created at runtime
│   ├── runs/                 # Per-run directories (model.py, results.nc, metadata)
│   ├── scores/               # Individual score JSONs
│   └── analysis/             # Aggregate report and CSVs
└── tests/
    ├── test_runner.py        # Command construction, caching, isolation checks
    ├── test_scorer.py        # Each criterion with synthetic InferenceData
    └── test_analysis.py      # Cohen's d, effect sizes, report generation
```

## Isolation Measures

The benchmark must ensure that the `no_skill` condition has zero access to skill content. These measures prevent contamination:

| Risk | Mitigation |
|------|------------|
| Skill tool loads `~/.claude/skills/` | Excluded from `--tools` list |
| Slash commands invoke skills | `--disable-slash-commands` |
| Plugin hooks suggest the skill | Working dir is `/tmp/benchmark/`, not the repo |
| Session state leaks between runs | `--no-session-persistence` |
| Permission prompts block execution | `--dangerously-skip-permissions` |

## Cost Estimate

Each run is capped at $2.00. With 30 runs plus scoring (Haiku judge calls), expect roughly $30-60 for a full suite. The validation run (2 runs) costs ~$0.50.
