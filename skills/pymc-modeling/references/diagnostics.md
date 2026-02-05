# Diagnostics and Troubleshooting

Quick reference for post-sampling diagnostics and common issues. For comprehensive ArviZ usage, see [arviz.md](arviz.md). For model-building problems (shape errors, initialization failures), see [troubleshooting.md](troubleshooting.md).

## Table of Contents
- [Quick Symptom Reference](#quick-symptom-reference)
- [Quick Diagnostic Checklist](#quick-diagnostic-checklist)
- [Convergence Thresholds](#convergence-thresholds)
- [Divergence Troubleshooting](#divergence-troubleshooting)
- [Common Problems and Fixes](#common-problems-and-fixes)
- [Model Comparison Quick Reference](#model-comparison-quick-reference)

---

## Quick Symptom Reference

| Symptom | Primary Cause | Solution |
|---------|---------------|----------|
| `ValueError: Shape mismatch` | Parameter vs observation dimensions | Index into group params: `alpha[group_idx]` |
| `Initial evaluation failed: -inf` | Data outside distribution support | Check bounds; reduce jitter; use `init="adapt_diag"` |
| `Mass matrix contains zeros` | Unscaled features or flat priors | Standardize predictors; use weakly informative priors |
| High divergence count | Funnel geometry or hard boundaries | Non-centered parameterization; increase `target_accept` |
| Poor GP convergence | Inappropriate lengthscale prior | InverseGamma prior based on data scale |
| `TypeError` in model logic | Python `if/else` inside model | Use `pt.switch()` or `pytensor.ifelse` |
| Slow sampling with discrete vars | NUTS incompatible with discrete | Marginalize discrete variables |
| Inconsistent predictions | Different group factorization | Use `pd.Categorical` or `sort=True` in factorize |

---

## Quick Diagnostic Checklist

Run this immediately after every sampling run:

```python
import arviz as az

def check_sampling(idata, var_names=None):
    """Quick post-sampling diagnostic check."""
    # Exclude auxiliary parameters by default
    if var_names is None:
        var_names = ["~offset", "~raw"]

    # 1. Divergences
    n_div = idata.sample_stats["diverging"].sum().item()
    n_samples = idata.sample_stats["diverging"].size
    print(f"Divergences: {n_div} ({100*n_div/n_samples:.2f}%)")

    # 2. Summary with key diagnostics
    summary = az.summary(idata, var_names=var_names)
    display_cols = ["mean", "sd", "hdi_3%", "hdi_97%", "ess_bulk", "ess_tail", "r_hat"]
    print(summary[[c for c in display_cols if c in summary.columns]])

    # 3. Flag issues
    if n_div > 0:
        print(f"\n⚠️  {n_div} divergences - see 'Divergence Troubleshooting'")
    bad_rhat = summary[summary["r_hat"] > 1.01].index.tolist()
    if bad_rhat:
        print(f"⚠️  High R-hat: {bad_rhat}")
    low_ess = summary[(summary["ess_bulk"] < 400) | (summary["ess_tail"] < 400)].index.tolist()
    if low_ess:
        print(f"⚠️  Low ESS: {low_ess}")

    return summary

# Usage
summary = check_sampling(idata)
```

### Essential Visual Checks

```python
# 1. Trace plots (mixing and stationarity)
az.plot_trace(idata, compact=True)

# 2. Rank plots (more sensitive than traces)
az.plot_rank(idata, var_names=["beta", "sigma"])

# 3. Pair plot with divergences (if any divergences)
if idata.sample_stats["diverging"].sum() > 0:
    az.plot_pair(idata, divergences=True)
```

---

## Convergence Thresholds

| Diagnostic | ✅ Good | ⚠️ Acceptable | ❌ Investigate |
|------------|---------|---------------|----------------|
| R-hat | < 1.01 | < 1.05 | > 1.05 |
| ESS bulk | > 400 | > 100 | < 100 |
| ESS tail | > 400 | > 100 | < 100 |
| Divergences | 0 | < 0.1% (random) | > 0.1% or clustered |
| MCSE/SD | < 5% | < 10% | > 10% |

### What Each Diagnostic Tells You

**R-hat (Potential Scale Reduction Factor)**
- Compares between-chain and within-chain variance
- R-hat ≈ 1.0 means chains have converged to same distribution
- High R-hat = chains disagree, don't trust results

**ESS (Effective Sample Size)**
- Accounts for autocorrelation in MCMC samples
- ESS_bulk: accuracy for posterior mean/median
- ESS_tail: accuracy for credible intervals (often lower)
- Low ESS = estimates unreliable, need more samples or better mixing

**Divergences**
- HMC/NUTS diagnostic for numerical issues
- Occur when sampler encounters difficult geometry
- Even a few divergences can indicate biased results

---

## Divergence Troubleshooting

### Step 1: Locate Divergent Regions

```python
# Where do divergences occur in parameter space?
az.plot_pair(
    idata,
    var_names=["alpha", "beta", "sigma"],  # adjust to your params
    divergences=True,
    divergences_kwargs={"color": "red", "alpha": 0.8}
)
```

Look for: Divergences clustered in specific regions (often near boundaries or in funnels).

### Step 2: Identify the Cause

| Pattern | Likely Cause | Fix |
|---------|--------------|-----|
| Funnel at low σ | Centered hierarchical | Non-centered parameterization |
| Boundary clustering | Weak/flat prior on scale | Informative prior (HalfNormal) |
| Scattered randomly | Step size too large | Increase target_accept |
| Near constraint | Hard boundary | Reparameterize or soften |

### Step 3: Apply Fixes

**Fix 1: Non-centered parameterization** (most common fix)

```python
# BEFORE: Centered (causes funnel)
alpha = pm.Normal("alpha", mu_alpha, sigma_alpha, dims="group")

# AFTER: Non-centered
alpha_offset = pm.Normal("alpha_offset", 0, 1, dims="group")
alpha = pm.Deterministic("alpha", mu_alpha + sigma_alpha * alpha_offset, dims="group")
```

**Fix 2: Better priors on scale parameters**

```python
# BEFORE: Problematic
sigma = pm.HalfCauchy("sigma", beta=10)  # too diffuse
sigma = pm.Uniform("sigma", 0, 100)      # flat prior

# AFTER: Weakly informative
sigma = pm.HalfNormal("sigma", sigma=1)
sigma = pm.Exponential("sigma", lam=1)
```

**Fix 3: Increase target acceptance**

```python
# More careful sampling (slower but fewer divergences)
idata = pm.sample(target_accept=0.95)  # default is 0.8

# For nutpie
idata = nutpie.sample(compiled, target_accept=0.95)
```

**Fix 4: Increase adaptation** (rare)

```python
idata = pm.sample(tune=2000)  # default is 1000
```

### When Divergences Persist

If divergences persist after trying above fixes:
1. Check model specification for errors
2. Simplify model to isolate problem
3. Consider if model is appropriate for data
4. Use `az.plot_energy(idata)` to diagnose sampler health

---

## Common Problems and Fixes

### Problem: High R-hat

**Symptoms**: R-hat > 1.01 for some parameters

**Diagnostic**:
```python
summary = az.summary(idata)
print(summary[summary["r_hat"] > 1.01])

az.plot_trace(idata, var_names=["problem_param"], compact=False)
```

**Causes and fixes**:
1. **Insufficient warmup**: Increase `tune` (e.g., tune=2000)
2. **Multimodality**: Check for multiple modes in trace plot
3. **Label switching**: Add ordering constraint for mixtures
4. **Slow mixing**: Reparameterize or increase samples

### Problem: Low ESS

**Symptoms**: ESS < 400 (especially ESS_tail)

**Diagnostic**:
```python
az.plot_ess(idata, var_names=["beta"], kind="evolution")
az.plot_autocorr(idata, var_names=["beta"])
```

**Causes and fixes**:
1. **High autocorrelation**: Reparameterize, improve sampler settings
2. **Not enough samples**: Increase `draws`
3. **Difficult posterior**: Increase `target_accept`, use non-centered

### Problem: Poor Posterior Predictive Fit

**Symptoms**: Observed data outside posterior predictive distribution

**Diagnostic**:
```python
with model:
    pm.sample_posterior_predictive(idata, extend_inferencedata=True)

az.plot_ppc(idata, kind="cumulative")
az.plot_loo_pit(idata, y="y")
```

**Causes and fixes**:
1. **Missing predictor**: Add relevant covariates
2. **Wrong likelihood**: Try different distribution family
3. **Missing structure**: Add hierarchical levels, random effects
4. **Outliers**: Use robust likelihood (StudentT instead of Normal)

### Problem: High Pareto k Values

**Symptoms**: `pareto_k > 0.7` for many observations in LOO

**Diagnostic**:
```python
loo = az.loo(idata, pointwise=True)
az.plot_khat(idata)

# Which observations are problematic?
import numpy as np
bad_idx = np.where(loo.pareto_k.values > 0.7)[0]
print(f"Problematic observations: {bad_idx}")
```

**Causes and fixes**:
1. **Influential outliers**: Investigate those data points
2. **Model misspecification**: Improve model for those observations
3. **Use K-fold CV**: When LOO approximation fails
4. **Moment matching**: Can improve LOO estimates (advanced)

---

## Model Comparison Quick Reference

### LOO-CV (Preferred)

```python
# Single model
loo = az.loo(idata, pointwise=True)
print(f"ELPD: {loo.elpd_loo:.1f} ± {loo.se:.1f}")
print(f"p_loo: {loo.p_loo:.1f}")

# Check Pareto k
print(f"Bad k (>0.7): {(loo.pareto_k > 0.7).sum().item()}")
```

### Model Comparison

```python
comparison = az.compare({
    "model_a": idata_a,
    "model_b": idata_b,
    "model_c": idata_c,
}, ic="loo")

# Key columns
print(comparison[["rank", "elpd_loo", "p_loo", "d_loo", "weight", "dse"]])

# Visual comparison
az.plot_compare(comparison)
```

**Interpretation**:
- `elpd_loo`: Higher is better (log predictive density)
- `d_loo`: Difference from best model
- `dse`: Standard error of difference
- **Rule**: If `d_loo < 2*dse`, models are effectively equivalent

### When to Use WAIC vs LOO

- **LOO (default)**: More robust, handles outliers better
- **WAIC**: Faster for large datasets, but less robust

```python
# WAIC alternative
waic = az.waic(idata)
```

---

## See Also

- [troubleshooting.md](troubleshooting.md) - Comprehensive problem-solution guide
- [arviz.md](arviz.md) - Comprehensive ArviZ usage guide
- [gotchas.md](gotchas.md) - Common modeling pitfalls
- [inference.md](inference.md) - Sampler selection and configuration
- [priors.md](priors.md) - Prior selection guide
