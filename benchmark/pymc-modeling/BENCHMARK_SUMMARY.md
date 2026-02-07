# PyMC Modeling Skill Benchmark Suite — Overview

## What This Benchmark Measures

This benchmarking suite evaluates whether the `pymc-modeling` AI skill improves the quality of Bayesian statistical models built by Claude (Anthropic's AI assistant). It compares model outputs **with** and **without** the skill on advanced modeling tasks that specifically require knowledge the skill provides.

### Core Goal

Measure the causal impact of skill injection on model quality, isolating the skill's contribution from Claude's baseline knowledge.

---

## The Five Tasks

The benchmark uses five tasks, each targeting a specific PyMC modeling pattern documented in the skill:

| Task | Domain | Skill Pattern Tested |
|------|--------|---------------------|
| **T1: Hierarchical Model** | 8 Schools (SAT coaching) | Non-centered parameterization, divergence diagnosis |
| **T2: Ordinal Regression** | GSS job satisfaction survey | OrderedLogistic, cutpoint priors, name/dimension conflicts |
| **T3: Model Comparison** | Synthetic regression data | LOO-CV workflow, `compute_log_likelihood()`, Pareto-k diagnostics |
| **T4: Gaussian Process** | Mauna Loa CO2 | HSGP + HSGPPeriodic, InverseGamma priors, kernel selection |
| **T5: Sparse Selection** | GSS financial satisfaction | Regularized horseshoe, `target_accept=0.99`, R2D2 alternative |

**Why these tasks matter:** Each tests something Claude commonly gets wrong without guidance. For example:
- Without the skill, Claude uses centered parameterization for hierarchical models and hits divergences
- Without the skill, Claude names a variable and dimension both "cutpoints", causing a `ValueError`
- Without the skill, Claude attempts full O(n³) GPs instead of scalable HSGP approximations

---

## Experimental Design

### Two Conditions

1. **no_skill**: Claude with base knowledge only — no skill content injected
2. **with_skill**: Claude with the SKILL.md (~4,500 tokens) injected via `--append-system-prompt`

### Isolation (Critical for Validity)

The skill is installed system-wide at `~/.claude/skills/pymc-modeling/`. If it leaks into the "no_skill" condition, the benchmark is invalid.

**Mitigation measures:**
- Run from isolated temp directories (`/tmp/benchmark/run_N/`) outside the repo
- Exclude the Skill tool from available tools
- Disable slash commands and session persistence
- No `--plugin-dir` flag (prevents plugin hooks from triggering)
- Verify token counts: `with_skill` should have ~4,500 more input tokens
- Parse responses to confirm zero `Skill` tool calls in `no_skill` condition

### Prompt Design

**Preamble** (infrastructure-only):
```
You are working in a directory that contains data files in data/.
Write a Python script called model.py that builds a Bayesian model using PyMC.
Run the script yourself. If it fails, fix it and try once more.
Save the InferenceData to results.nc after sampling.
Available packages: PyMC, ArviZ, nutpie, NumPy, SciPy, matplotlib, polars.
```

**Task prompts** describe the *statistical problem only* — never mention PyMC-specific patterns (non-centered, HSGP, OrderedLogistic, etc.). This ensures we're testing whether the skill provides the right knowledge, not whether Claude can follow explicit instructions.

### Execution Model

Claude runs in `--print` mode with tool access (Bash, Read, Write, Glob, Grep). This is functionally agentic:
1. Claude writes `model.py`
2. Runs it via Bash tool
3. Sees errors
4. Fixes and retries (up to 2 attempts)
5. Saves `results.nc`

**Hard limits per task:**
- **Wall-clock timeout**: 10 minutes (kills process)
- **Retry cap**: 2 tries (enforced by prompt + `--max-budget-usd 2.0`)
- **No web search**: Skill tool excluded

### Replications

3 replications per task per condition = **30 total runs** (5 tasks × 2 conditions × 3 reps). Conditions are interleaved to avoid time-of-day effects.

---

## Scoring System (0-20 Points)

Each run is scored on four criteria:

| Criterion | Points | Method |
|-----------|--------|--------|
| **Model Produced** | 0-5 | Automated: Checks `results.nc` exists with posterior group |
| **Convergence** | 0-5 | Automated: r_hat, ESS, divergences from `.nc` file |
| **Model Appropriateness** | 0-5 | **LLM Judge** (Haiku) with task-specific rubric |
| **Best Practices** | 0-5 | Automated: Regex checks for coords/dims, nutpie, random_seed, etc. |

### LLM Judge Details

The "Model Appropriateness" criterion requires subjective judgment (Is this the right likelihood? Are priors reasonable?). An LLM (Haiku, cheap and fast) scores the code against task-specific rubrics:

**Example rubric (T1 Hierarchical):**
- 1 = any hierarchical structure
- 2 = + hyperpriors
- 3 = + non-centered parameterization
- 4 = + reasonable priors
- 5 = + Deterministic for derived effects

**Judge prompt template:**
```
Score this PyMC code for model appropriateness on a 0-5 scale.

Task: {task_name}
Rubric: {task_specific_rubric}

Respond with ONLY a JSON object: {"score": N, "reasoning": "..."}

Code:
```python
{code}
```
```

**Fallback:** If the LLM judge fails (timeout, API error), the system falls back to regex-based scoring using task-specific patterns.

---

## Analysis

After scoring all 30 runs, the system computes:

### Cohen's d Effect Sizes

For each task and criterion, comparing `with_skill` vs `no_skill`:

```python
d = (mean_with_skill - mean_no_skill) / pooled_sd
```

**Interpretation:**
- d > 0: skill helps
- |d| < 0.2: negligible
- 0.2 ≤ |d| < 0.5: small
- 0.5 ≤ |d| < 0.8: medium
- |d| ≥ 0.8: large

### Output Reports

- **Markdown report** (`results/analysis/report.md`): Summary tables and effect sizes
- **CSV exports** (`summary.csv`, `effects.csv`): For further analysis

---

## Validation Gate

Before running the full suite (~$60, 30 runs), a **validation run** must pass:

```bash
pixi run python -m src.cli validate
```

This runs T1 in both conditions and checks:
1. Both runs completed (token counts reasonable)
2. No Skill tool calls in either condition
3. Tools were actually used (multiple turns)
4. `model.py` was produced
5. `results.nc` has posterior group
6. Scorer runs successfully
7. Working directory isolation (no `.claude-plugin/`, `hooks/` contamination)

If validation fails, the full suite refuses to start. This prevents wasting money on invalid runs.

---

## Why This Benchmark Is Valid and Useful

### 1. **Tests Real Skill Value**
The tasks are deliberately chosen to be hard enough that the skill makes a visible difference. No basic linear regression — every task targets specific skill coverage areas.

### 2. **Controlled Experiment**
The only difference between conditions is `--append-system-prompt`. Same model (Sonnet), same tools, same timeout, same prompts. Any performance difference is attributable to the skill.

### 3. **Robust Isolation**
Multiple safeguards ensure the skill doesn't leak into the "no_skill" condition: excluded tools, disabled slash commands, temp directories, token verification, and response parsing.

### 4. **Comprehensive Scoring**
The 4-criterion rubric evaluates:
- **Did it work?** (Model produced)
- **Did it converge?** (Statistical validity)
- **Was it appropriate?** (Subjective quality via LLM judge)
- **Did it follow best practices?** (Code quality)

### 5. **Statistical Power**
With 3 replications per condition and paired comparison (same task, different condition), the design maximizes statistical power to detect skill effects.

### 6. **Practical Relevance**
The tasks mirror real-world Bayesian modeling challenges: hierarchical data, ordinal outcomes, model comparison, time series with GPs, and high-dimensional variable selection.

---

## CLI Commands

```bash
# Validation (required before full suite)
pixi run python -m src.cli validate

# Run all tasks (30 runs)
pixi run python -m src.cli run --all --reps 3

# Run single task
pixi run python -m src.cli run --task T1_hierarchical

# Score all completed runs
pixi run python -m src.cli score --all

# Generate analysis report
pixi run python -m src.cli analyze

# Check status
pixi run python -m src.cli status

# List available tasks
pixi run python -m src.cli list-tasks
```

---

## Data Preparation

All datasets are sized so well-specified models sample in under 2 minutes:

| Dataset | Source | Preparation |
|---------|--------|-------------|
| GSS 2022 | NORC | Cleaned (missing values dropped), subsampled to 500 rows |
| Mauna Loa CO2 | NOAA | Subsampled to ~400 rows (every other month) |
| Synthetic regression | Generated | 150 rows, includes nonlinear pattern + outliers |

The rationale: 3,544 rows with horseshoe priors can take 10+ minutes. 500 rows produces meaningful posteriors while keeping sampling fast.

---

## Key Files

```
benchmark/pymc-modeling/
├── src/
│   ├── cli.py           # CLI entry point
│   ├── runner.py        # Launches Claude, manages working dirs
│   ├── scorer.py        # 4-criterion scoring + LLM judge
│   └── analysis.py      # Effect sizes and reports
├── tasks.yaml           # 5 task definitions + rubrics
├── data/                # Datasets (cleaned, subsampled)
├── results/
│   ├── runs/            # Per-run working dirs (model.py, results.nc)
│   ├── scores/          # Individual score JSONs
│   └── analysis/        # Aggregate reports
└── tests/               # Unit tests for scorer, runner, analysis
```

---

## Lessons Embedded

These constraints reflect hard-won experience:

- **Save early**: Claude should save `results.nc` immediately after sampling. A late crash in post-processing shouldn't destroy a valid posterior. The skill teaches this; the benchmark measures whether Claude learns it.

- **Pre-clean data**: PyMC API errors from missing data handling dominated prior failure modes. Pre-cleaning isolates modeling skill.

- **nutpie quirk**: nutpie ignores `idata_kwargs`, so `log_likelihood` can be silently dropped. The skill documents this.

- **ArviZ API drift**: `az.compare()` returns `d_loo`/`dse` columns, not the older names. The skill has current API guidance.

- **Name conflicts**: PyMC forbids using the same string for both variable and dimension names. The skill warns about this; without it, ordinal models commonly crash.

---

## Summary

This benchmark suite provides a rigorous, statistically sound method for evaluating whether the `pymc-modeling` skill actually improves Claude's Bayesian modeling capabilities. It uses controlled experimentation, robust isolation, comprehensive scoring, and effect size analysis to answer the question: **Does this skill help Claude build better PyMC models?**
