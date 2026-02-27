# PyMC Skill Review: Gaps and Recommendations

**Date**: 2026-02-09
**Based on**: Benchmark report (2026-02-07), analysis of 30 run artifacts

## What the Benchmark Reveals

The skill improves scores across all 5 tasks (d=0.55 overall), with the largest gains on **best practices** (+1.7 points average). But there are specific gaps where stronger guidance in SKILL.md would produce measurably better results.

## 1. HSGP Threshold Is Wrong (T4 — Highest Impact)

**Current guidance** (SKILL.md line 413):

> Default to HSGP for most GP problems (n > 500, 1-3D inputs)... For small datasets (n < 500), use pm.gp.Marginal or pm.gp.Latent.

**Problem**: The Mauna Loa dataset has n=396. Claude reads "n < 500 → use Marginal" and produces O(n³) code that times out at 15 minutes. This happened in **5 of 6 T4 runs** (including 2 with_skill runs). The sole success (with_skill rep1, 15/20) used HSGP + HSGPPeriodic.

Even at n=300, Marginal GP requires computing a 300×300 covariance matrix and its Cholesky decomposition on every gradient evaluation during NUTS — thousands of times. HSGP replaces this with O(nm) where m~20.

**Fix**: Lower the threshold dramatically. HSGP is nearly always faster and accurate enough. Something like: "**Always prefer HSGP** unless n < ~50 and exact inference is specifically needed. Even at n=200, Marginal GP is prohibitively slow for MCMC."

## 2. No nutpie log_likelihood Warning (T3 — High Impact)

**Current guidance** (SKILL.md lines 73-81): Shows nutpie as the default sampler, with no caveats.

**Problem**: nutpie silently ignores `idata_kwargs={"log_likelihood": True}`. This means model comparison with LOO-CV fails if Claude uses nutpie (the recommended sampler) and doesn't know to call `pm.compute_log_likelihood()` afterwards. The no_skill T5 rep0 file shows exactly this: `nuts_sampler="nutpie", idata_kwargs={"log_likelihood": True}` — the log_likelihood is silently dropped.

This is documented in `references/inference.md` and `references/diagnostics.md`, but those files are **not available in --print mode** where only SKILL.md is injected.

**Fix**: Add a warning directly in the nutpie sampling section:

```
**Important**: nutpie does not store log_likelihood automatically.
If you need LOO-CV or model comparison, compute it after sampling:
    pm.compute_log_likelihood(idata, model=model)
```

## 3. No Horseshoe/Shrinkage Pattern in SKILL.md (T5 — High Impact)

**Current**: The "Common Patterns" section has hierarchical, GLMs, GPs, time series, BART, mixtures, specialized likelihoods — but **no shrinkage/variable selection pattern**. The regularized horseshoe is only in `references/priors.md`, invisible in --print mode.

**What the benchmark shows**:

- **no_skill**: Uses simple horseshoe (`beta = Normal(0, tau * lambda_)`) — no regularization, no `target_accept` tuning. 1 of 3 runs timed out.
- **with_skill**: Uses regularized horseshoe (`c2` slab parameter) — stable, all 3 succeed. But still misses `target_accept=0.95+` (a rubric criterion).

**Fix**: Add a "Sparse Regression / Horseshoe" pattern to Common Patterns:

```python
# Regularized horseshoe prior (Finnish horseshoe)
tau = pm.HalfCauchy("tau", beta=1)
lambda_tilde = pm.HalfCauchy("lambda_tilde", beta=1, dims="features")
c2 = pm.InverseGamma("c2", alpha=1, beta=1)  # Slab regularization
lambda_sq = c2 * lambda_tilde**2 / (c2 + tau**2 * lambda_tilde**2)

beta_raw = pm.Normal("beta_raw", 0, 1, dims="features")
beta = pm.Deterministic("beta", tau * pm.math.sqrt(lambda_sq) * beta_raw, dims="features")
```

Include the note: "Horseshoe priors create double-funnel geometry. Use `target_accept=0.95` or higher."

## 4. Ordinal Regression Dimension Name Conflict (T2 — Medium Impact)

**Current** (SKILL.md lines 522-528): The ordinal example uses `cutpoints` as a variable name. The benchmark rubric specifically checks: "variable names and dimension names don't conflict (no 'cutpoints' for both)."

**Problem**: Claude sometimes creates a `cutpoints` dimension AND a `cutpoints` variable, causing PyMC shape errors. The SKILL.md example doesn't warn about this.

**Fix**: Add a note after the ordinal example:

```
**Note**: Don't use the same name for a variable and a dimension.
If you use dims="categories", don't also name a variable "categories".
```

## 5. Save-Early-After-Sampling Pattern (All Tasks — Medium Impact)

The report notes that late crashes (post-MCMC) destroy valid results. The preamble tells Claude to save, but the SKILL.md's workflow pattern (lines 350-362) shows saving *after* posterior predictive, not immediately after sampling.

**Fix**: In the workflow pattern, emphasize saving immediately:

```python
idata = pm.sample(...)
idata.to_netcdf("results.nc")  # Save immediately after sampling!

# Then do posterior predictive, diagnostics, etc.
pm.sample_posterior_predictive(idata, extend_inferencedata=True)
idata.to_netcdf("results.nc")  # Update with posterior predictive
```

## 6. Minor: az.compare() Column Names (T3)

The SKILL.md already uses the correct column names (`d_loo`, `dse`) at line 299. This is good and should stay.

## Summary of Expected Impact

| Change | Tasks Affected | Expected Score Gain | Tokens Added |
|--------|---------------|-------------------|-------------|
| Lower HSGP threshold | T4 | +4-8 points (timeout → success) | ~50 words |
| nutpie log_likelihood warning | T3, T5 | +0.5-1 point (model comparison) | ~30 words |
| Horseshoe pattern in SKILL.md | T5 | +1-2 points (regularized + target_accept) | ~150 words |
| Ordinal name conflict warning | T2 | +0.3-0.5 points | ~20 words |
| Save-early pattern | All | Prevents ~10% of late-crash failures | ~30 words |

The HSGP threshold fix is by far the highest-value change — it's the difference between 0/6 and potentially 4-6/6 on T4. The horseshoe pattern addition is second highest, moving T5 from "sometimes works" to "reliably works with all best practices."
