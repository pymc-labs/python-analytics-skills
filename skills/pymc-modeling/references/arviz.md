# ArviZ: Expert Bayesian Analysis

This guide covers ArviZ like an expert Bayesian modeler uses it: not just what each function does, but *when* to use it, *what* to look for, and *how* to interpret results.

## Table of Contents
- [The Expert Workflow](#the-expert-workflow)
- [InferenceData Fundamentals](#inferencedata-fundamentals)
- [Phase 1: Immediate Post-Sampling Checks](#phase-1-immediate-post-sampling-checks)
- [Phase 2: Deep Convergence Assessment](#phase-2-deep-convergence-assessment)
- [Phase 3: Model Criticism](#phase-3-model-criticism)
- [Phase 4: Parameter Interpretation](#phase-4-parameter-interpretation)
- [Phase 5: Model Comparison](#phase-5-model-comparison)
- [Advanced Techniques](#advanced-techniques)
- [Common Interpretation Patterns](#common-interpretation-patterns)
- [Publication-Quality Figures](#publication-quality-figures)

---

## The Expert Workflow

An expert doesn't randomly try plots—they follow a systematic workflow:

```
┌──────────────────────────────────────────────────────────────────────┐
│  1. SAMPLE → 2. DIAGNOSE → 3. CRITICIZE → 4. INTERPRET → 5. COMPARE │
└──────────────────────────────────────────────────────────────────────┘
```

**Phase 1 (Immediate)**: Did sampling work? Check divergences, R-hat, ESS, trace plots.
**Phase 2 (Deep)**: Are chains healthy? Rank plots, energy, autocorrelation, MCSE.
**Phase 3 (Criticism)**: Does the model fit? PPC, LOO-PIT, residual analysis.
**Phase 4 (Interpretation)**: What did we learn? Posteriors, forest plots, pair plots.
**Phase 5 (Comparison)**: Which model is best? LOO-CV, WAIC, stacking.

**Critical rule**: Never interpret parameters (Phase 4) until Phases 1-3 pass.

---

## InferenceData Fundamentals

ArviZ uses `InferenceData`—an xarray-based container. Master this to unlock ArviZ's power.

### Structure

```python
import arviz as az

# InferenceData groups:
idata.posterior          # MCMC samples: (chain, draw, *dims)
idata.posterior_predictive  # Predictions at observed points
idata.prior              # Prior samples
idata.prior_predictive   # Prior predictions
idata.observed_data      # The actual data
idata.sample_stats       # Sampler diagnostics (divergences, energy, etc.)
idata.log_likelihood     # Pointwise log-likelihoods (for LOO/WAIC)
```

### Essential Operations

```python
# Access a parameter (returns xarray.DataArray)
beta = idata.posterior["beta"]

# Convert to numpy
beta_vals = idata.posterior["beta"].values  # shape: (chains, draws, *dims)

# Flatten across chains
beta_flat = idata.posterior["beta"].stack(sample=("chain", "draw")).values

# Select specific chains/draws
idata.posterior["beta"].sel(chain=0, draw=slice(500, None))

# Compute statistics
idata.posterior["beta"].mean(dim=["chain", "draw"])
idata.posterior["beta"].quantile([0.025, 0.975], dim=["chain", "draw"])

# Filter variables
idata.posterior[["alpha", "beta"]]
```

### Combining InferenceData Objects

```python
# Add posterior predictive to existing idata
with model:
    pm.sample_posterior_predictive(idata, extend_inferencedata=True)

# Or manually extend
idata.extend(pm.sample_posterior_predictive(idata))

# Merge separate idata objects
idata_combined = az.concat([idata1, idata2], dim="chain")
```

### Saving and Loading

```python
# Save (NetCDF format, portable)
idata.to_netcdf("results.nc")

# Load
idata = az.from_netcdf("results.nc")

# Save with compression (for large files)
idata.to_netcdf("results.nc", engine="h5netcdf",
                encoding={var: {"zlib": True} for var in idata.posterior.data_vars})
```

---

## Phase 1: Immediate Post-Sampling Checks

Run these checks immediately after `pm.sample()` returns. If any fail, do not proceed to interpretation.

### The 30-Second Check

```python
def quick_diagnostics(idata, var_names=None):
    """Run immediately after sampling."""

    # 1. Divergences (must be 0 or near 0)
    n_div = idata.sample_stats["diverging"].sum().item()
    n_samples = idata.sample_stats["diverging"].size
    div_pct = 100 * n_div / n_samples
    print(f"Divergences: {n_div} ({div_pct:.2f}%)")

    # 2. Summary with convergence stats
    summary = az.summary(idata, var_names=var_names)

    # 3. Flag problems
    bad_rhat = summary[summary["r_hat"] > 1.01]
    low_ess = summary[(summary["ess_bulk"] < 400) | (summary["ess_tail"] < 400)]

    if len(bad_rhat) > 0:
        print(f"\n⚠️  R-hat > 1.01 for: {list(bad_rhat.index)}")
    if len(low_ess) > 0:
        print(f"\n⚠️  Low ESS for: {list(low_ess.index)}")
    if n_div > 0:
        print(f"\n⚠️  {n_div} divergences detected - investigate before interpreting!")

    return summary

# Usage
summary = quick_diagnostics(idata, var_names=["~offset"])  # exclude auxiliary
```

### az.summary: The Diagnostic Swiss Army Knife

```python
# Full summary
summary = az.summary(idata)

# Key columns to check:
# - mean, sd: posterior mean and standard deviation
# - hdi_3%, hdi_97%: 94% highest density interval
# - mcse_mean, mcse_sd: Monte Carlo standard error
# - ess_bulk: effective sample size for the bulk of the distribution
# - ess_tail: effective sample size for the tails (crucial for credible intervals)
# - r_hat: potential scale reduction factor

# Exclude auxiliary parameters (e.g., non-centered offsets)
summary = az.summary(idata, var_names=["~offset", "~raw"])

# Include specific stats only
summary = az.summary(idata, stat_funcs={"median": np.median}, extend=True)

# Custom credible interval
summary = az.summary(idata, hdi_prob=0.90)
```

**Interpretation thresholds:**

| Metric | Good | Acceptable | Investigate |
|--------|------|------------|-------------|
| `r_hat` | < 1.01 | < 1.05 | > 1.05 |
| `ess_bulk` | > 400 | > 100 | < 100 |
| `ess_tail` | > 400 | > 100 | < 100 |
| `mcse_mean` | < 5% of SD | < 10% of SD | > 10% of SD |

### az.plot_trace: Visual Convergence Check

```python
# Basic trace plot
az.plot_trace(idata, var_names=["beta", "sigma"])

# Compact mode for many parameters
az.plot_trace(idata, compact=True, combined=True)

# Rank-normalized traces (more sensitive to problems)
az.plot_trace(idata, kind="rank_bars")
az.plot_trace(idata, kind="rank_vlines")
```

**What to look for:**

✅ **Good mixing** (left panel): Overlapping density curves for all chains
✅ **Stationarity** (right panel): Fuzzy caterpillar, no trends, no steps
❌ **Bad mixing**: Separated densities, one chain stuck in different region
❌ **Non-stationarity**: Visible trends, sudden jumps, periodic patterns

### Checking Divergences

```python
# Count divergences
n_div = idata.sample_stats["diverging"].sum().item()

# Percentage
div_pct = 100 * idata.sample_stats["diverging"].mean().item()

# When did they occur? (during warmup vs sampling)
# Divergences in sample_stats are from sampling phase only

# Visualize where divergences occur in parameter space
az.plot_pair(idata, var_names=["alpha", "sigma"], divergences=True)
```

**Divergence rules:**
- **0 divergences**: Ideal
- **< 0.1% and random**: Often acceptable, but investigate
- **> 0.1% or clustered**: Problem—do not trust results

---

## Phase 2: Deep Convergence Assessment

If Phase 1 passes, these deeper checks validate that the sampler fully explored the posterior.

### az.plot_rank: The Modern Convergence Test

Rank plots are more sensitive than trace plots for detecting convergence issues.

```python
az.plot_rank(idata, var_names=["beta", "sigma"])
```

**Interpretation**: Each chain should have a roughly uniform distribution of ranks. If one chain consistently has higher or lower ranks, it hasn't converged to the same distribution.

**What to look for:**
- ✅ Uniform histograms across all chains
- ❌ One chain with ranks concentrated at one end
- ❌ Systematic patterns (periodicity, gaps)

### az.plot_ess: Effective Sample Size Evolution

```python
# How ESS grows with more draws
az.plot_ess(idata, var_names=["beta"], kind="evolution")

# ESS across the distribution (quantile-specific)
az.plot_ess(idata, var_names=["beta"], kind="quantile")

# Local ESS (identifies problematic regions)
az.plot_ess(idata, var_names=["beta"], kind="local")
```

**Evolution plot interpretation:**
- ✅ Linear growth: Good mixing
- ❌ Plateauing: Poor mixing, more samples won't help
- ❌ Early plateau then growth: Slow adaptation

**Quantile plot interpretation:**
- ✅ Roughly constant ESS across quantiles
- ❌ Low ESS at tails: Unreliable credible intervals
- ❌ Very low ESS at specific quantiles: Possible multimodality

### az.plot_mcse: Monte Carlo Error Visualization

```python
# MCSE across quantiles
az.plot_mcse(idata, var_names=["beta"])

# Local MCSE (error varies across distribution)
az.plot_mcse(idata, var_names=["beta"], kind="local")
```

**Rule of thumb**: MCSE should be < 5% of posterior SD for reliable inference.

### az.plot_autocorr: Mixing Efficiency

```python
az.plot_autocorr(idata, var_names=["beta", "sigma"])
```

**Interpretation:**
- ✅ Rapid decay to zero (within ~20 lags)
- ❌ Slow decay: High autocorrelation → low ESS
- ❌ Negative autocorrelation: Can indicate sampler issues

### az.plot_energy: HMC/NUTS Health

```python
az.plot_energy(idata)
```

This plot shows the marginal energy distribution and the energy transition distribution.

**Interpretation:**
- ✅ Overlapping distributions: Good exploration
- ❌ Large gap between distributions: Poor exploration, potential bias
- ❌ Marginal energy has long tails: Difficult posterior geometry

**Expert tip**: Energy plots are especially useful for diagnosing problems that R-hat and ESS miss, like when the sampler explores only part of a multimodal posterior.

### az.rhat and az.ess: Programmatic Access

```python
# Get R-hat for all parameters
rhat_values = az.rhat(idata)

# Get ESS
ess_bulk = az.ess(idata, method="bulk")
ess_tail = az.ess(idata, method="tail")

# Find problematic parameters
for var in rhat_values.data_vars:
    rhat = rhat_values[var].values
    if np.any(rhat > 1.01):
        print(f"Warning: {var} has R-hat = {rhat.max():.3f}")
```

---

## Phase 3: Model Criticism

Convergence doesn't mean the model is good—only that MCMC worked. Now assess whether the model actually fits the data.

### az.plot_ppc: Posterior Predictive Checks

The most important model criticism tool. Does the model generate data that looks like the observed data?

```python
# First, generate posterior predictive samples
with model:
    pm.sample_posterior_predictive(idata, extend_inferencedata=True)

# Density overlay (default)
az.plot_ppc(idata, kind="kde")

# Cumulative distribution (better for systematic deviations)
az.plot_ppc(idata, kind="cumulative")

# Scatter plot (for continuous outcomes)
az.plot_ppc(idata, kind="scatter")

# Subsample for speed
az.plot_ppc(idata, num_pp_samples=100)
```

**What to look for (density/KDE):**
- ✅ Dark line (observed) within the cloud of light lines (predicted)
- ❌ Observed data outside predicted distribution
- ❌ Shape mismatch (e.g., model symmetric but data skewed)

**What to look for (cumulative):**
- ✅ Observed CDF within the posterior predictive band
- ❌ Systematic deviation (observed consistently above/below)
- ❌ Crossing pattern (model wrong in different directions at different values)

### Grouped/Stratified PPC

Check fit across subgroups:

```python
# PPC by group (if using coords)
az.plot_ppc(idata, kind="cumulative", flatten=[])

# For specific observed variable
az.plot_ppc(idata, var_names=["y_obs"], kind="kde")
```

### Custom Posterior Predictive Checks

Sometimes you need to check specific features:

```python
# Define test statistics
def tail_fraction(x):
    """Fraction of values > 95th percentile of observed data"""
    threshold = np.percentile(idata.observed_data["y"].values, 95)
    return (x > threshold).mean()

def zero_fraction(x):
    """Fraction of zeros (for zero-inflated data)"""
    return (x == 0).mean()

# Compute for observed data
obs_stat = tail_fraction(idata.observed_data["y"].values)

# Compute for each posterior predictive draw
pp_stats = []
for i in range(idata.posterior_predictive.dims["draw"]):
    pp_sample = idata.posterior_predictive["y"].isel(draw=i).values.flatten()
    pp_stats.append(tail_fraction(pp_sample))

# Compare
import matplotlib.pyplot as plt
plt.hist(pp_stats, bins=30, alpha=0.7, label="Posterior predictive")
plt.axvline(obs_stat, color="red", linewidth=2, label="Observed")
plt.xlabel("Tail fraction")
plt.legend()
```

### az.plot_loo_pit: Calibration Diagnostic

The LOO-PIT (Leave-One-Out Probability Integral Transform) checks calibration: are the posterior predictive quantiles uniformly distributed?

```python
az.plot_loo_pit(idata, y="y")
```

**Interpretation:**
- ✅ Uniform distribution (flat histogram, diagonal line): Well-calibrated
- ❌ U-shape: Underdispersed (model overconfident)
- ❌ Inverted U-shape: Overdispersed (model underconfident)
- ❌ S-curve: Systematic bias

**Expert insight**: LOO-PIT is more sensitive than PPC for detecting calibration issues because it evaluates each observation using a model fit without that observation.

### az.plot_bpv: Bayesian p-values

```python
# Histogram of Bayesian p-values
az.plot_bpv(idata, kind="p_value")

# Using a test statistic
az.plot_bpv(idata, kind="t_stat")
```

**Interpretation:**
- Values near 0.5: Good calibration
- Values near 0 or 1: Model systematically over/under-predicts

### Residual Analysis

For regression models, check residuals:

```python
import numpy as np

# Compute posterior mean predictions
y_pred = idata.posterior_predictive["y"].mean(dim=["chain", "draw"])
y_obs = idata.observed_data["y"]

# Residuals
residuals = y_obs - y_pred

# Plot residuals vs fitted
import matplotlib.pyplot as plt
plt.scatter(y_pred, residuals, alpha=0.5)
plt.axhline(0, color="red", linestyle="--")
plt.xlabel("Fitted values")
plt.ylabel("Residuals")
```

**What to look for:**
- ✅ Random scatter around zero
- ❌ Funnel shape: Heteroscedasticity
- ❌ Curved pattern: Missing nonlinear term
- ❌ Clusters: Missing grouping structure

---

## Phase 4: Parameter Interpretation

Only after Phases 1-3 pass should you interpret parameter estimates.

### az.plot_posterior: Marginal Summaries

```python
# Basic posterior summary
az.plot_posterior(idata, var_names=["beta", "sigma"])

# With reference value (null hypothesis)
az.plot_posterior(idata, var_names=["beta"], ref_val=0)

# With ROPE (Region of Practical Equivalence)
az.plot_posterior(idata, var_names=["beta"], rope=[-0.1, 0.1])

# Custom HDI probability
az.plot_posterior(idata, hdi_prob=0.90)

# Point estimate options
az.plot_posterior(idata, point_estimate="mode")  # or "mean", "median"
```

**ROPE interpretation:**
- Report % of posterior inside ROPE
- If > 95% inside ROPE: Practically equivalent to null
- If < 5% inside ROPE: Practically different from null

### az.plot_forest: Comparing Parameters

Essential for hierarchical models and comparing effects.

```python
# Basic forest plot
az.plot_forest(idata, var_names=["alpha"], combined=True)

# With R-hat coloring (spot convergence issues)
az.plot_forest(idata, var_names=["alpha"], r_hat=True)

# With ESS sizing
az.plot_forest(idata, var_names=["alpha"], ess=True)

# Compare multiple models
az.plot_forest(
    [idata_pooled, idata_hierarchical, idata_unpooled],
    model_names=["Pooled", "Hierarchical", "Unpooled"],
    var_names=["alpha"],
    combined=True
)

# Ridgeplot style (better for many groups)
az.plot_forest(idata, var_names=["alpha"], kind="ridgeplot")
```

### az.plot_pair: Parameter Relationships

Critical for understanding parameter correlations and identifying problems.

```python
# Basic pair plot
az.plot_pair(idata, var_names=["alpha", "beta", "sigma"])

# With divergences (essential for debugging)
az.plot_pair(idata, var_names=["alpha", "sigma"], divergences=True)

# Different plot types
az.plot_pair(idata, kind="kde")      # Kernel density
az.plot_pair(idata, kind="hexbin")   # Better for large samples
az.plot_pair(idata, kind="scatter")  # Default

# With marginal distributions
az.plot_pair(idata, marginals=True)

# Reference values (e.g., true values in simulation study)
az.plot_pair(idata, reference_values={"alpha": 1.0, "beta": 0.5})

# Point estimate overlay
az.plot_pair(idata, point_estimate="mean")
```

**What to look for:**
- **Strong correlations**: May indicate non-identifiability or need for reparameterization
- **Banana/funnel shapes**: Common in hierarchical models, may need non-centered parameterization
- **Divergences clustered**: Indicates problematic posterior regions
- **Multimodality**: Multiple clusters suggest mixture or label switching

### az.plot_parallel: High-Dimensional Relationships

```python
az.plot_parallel(idata, var_names=["alpha", "beta", "gamma", "sigma"])
```

Divergent samples shown in different color—look for patterns indicating where the sampler struggles.

### az.plot_violin: Distribution Comparison

```python
az.plot_violin(idata, var_names=["alpha"])
```

Good for comparing distributions across groups when you want quartile information.

### az.plot_ridge: Compact Multi-Parameter View

```python
az.plot_ridge(idata, var_names=["alpha"])
```

Useful when you have many group-level parameters to display compactly.

---

## Phase 5: Model Comparison

Compare models using out-of-sample predictive accuracy.

### az.loo: Leave-One-Out Cross-Validation

```python
# Compute LOO (requires log_likelihood in idata)
loo_result = az.loo(idata, pointwise=True)
print(loo_result)
```

**Key outputs:**
- `elpd_loo`: Expected log pointwise predictive density (higher is better)
- `se`: Standard error of elpd_loo
- `p_loo`: Effective number of parameters (model complexity)
- `pareto_k`: Diagnostic for approximation quality

**Pareto k interpretation:**
| k value | Interpretation | Action |
|---------|----------------|--------|
| < 0.5 | Good | LOO reliable |
| 0.5-0.7 | OK | Probably fine, be cautious |
| 0.7-1.0 | Bad | LOO unreliable for these points |
| > 1.0 | Very bad | Likely model misspecification |

```python
# Check Pareto k values
loo = az.loo(idata, pointwise=True)
print(f"Good k (< 0.5): {(loo.pareto_k < 0.5).sum().item()}")
print(f"OK k (0.5-0.7): {((loo.pareto_k >= 0.5) & (loo.pareto_k < 0.7)).sum().item()}")
print(f"Bad k (> 0.7): {(loo.pareto_k > 0.7).sum().item()}")
```

### az.plot_khat: Visualize Pareto k

```python
az.plot_khat(idata)
```

Points above the 0.7 line are influential observations where LOO approximation is unreliable. Consider:
1. Using moment matching (`az.loo(..., pointwise=True)` then refit)
2. K-fold CV instead
3. Investigating why these points are influential

### az.waic: Widely Applicable Information Criterion

```python
waic_result = az.waic(idata)
print(waic_result)
```

WAIC is an alternative to LOO. LOO is generally preferred (more robust), but WAIC is faster for large datasets.

### az.compare: Model Comparison Table

```python
comparison = az.compare({
    "linear": idata_linear,
    "quadratic": idata_quad,
    "spline": idata_spline,
}, ic="loo")

print(comparison)
```

**Key columns:**
- `rank`: Model ranking (0 is best)
- `elpd_loo`: Expected log pointwise predictive density
- `p_loo`: Effective number of parameters
- `d_loo`: Difference from best model
- `weight`: Stacking weight (for model averaging)
- `se`: Standard error
- `dse`: Standard error of difference

**Decision rules:**
- If `d_loo < 2`: Models practically indistinguishable
- If `d_loo < dse`: Difference not significant
- If `d_loo > 4` and `d_loo > 2*dse`: Meaningful difference

### az.plot_compare: Visual Model Comparison

```python
az.plot_compare(comparison)
```

Shows ELPD differences with error bars. Overlapping bars suggest models perform similarly.

### az.plot_elpd: Pointwise Comparison

```python
az.plot_elpd({"linear": idata_linear, "quadratic": idata_quad})
```

Identifies which observations drive model differences—useful for understanding where models disagree.

### Model Averaging with Stacking Weights

```python
# Stacking weights from comparison
weights = comparison["weight"]

# Use weights for prediction averaging
y_pred_avg = (
    weights["linear"] * idata_linear.posterior_predictive["y"].mean(dim=["chain", "draw"]) +
    weights["quadratic"] * idata_quad.posterior_predictive["y"].mean(dim=["chain", "draw"])
)
```

---

## Advanced Techniques

### Working with xarray Directly

```python
import xarray as xr

# Compute custom statistics
posterior = idata.posterior

# Probability of effect > 0
prob_positive = (posterior["beta"] > 0).mean(dim=["chain", "draw"])

# Probability of effect > threshold
threshold = 0.5
prob_gt_threshold = (posterior["beta"] > threshold).mean(dim=["chain", "draw"])

# Posterior contrasts
if "group" in posterior["alpha"].dims:
    contrast = posterior["alpha"].sel(group="treatment") - posterior["alpha"].sel(group="control")
    az.plot_posterior(contrast.to_dataset(name="treatment_effect"))
```

### Custom Summary Functions

```python
def prob_direction(x):
    """Probability parameter has same sign as mean"""
    return (np.sign(x) == np.sign(x.mean())).mean()

def hdi_width(x):
    """Width of 94% HDI"""
    hdi = az.hdi(x, hdi_prob=0.94)
    return hdi[1] - hdi[0]

# Add to summary
summary = az.summary(
    idata,
    stat_funcs={"P(same sign)": prob_direction, "HDI width": hdi_width},
    extend=True
)
```

### Subsampling for Large InferenceData

```python
# Thin samples (keep every nth)
idata_thin = idata.sel(draw=slice(None, None, 10))  # Keep every 10th

# Random subset
import numpy as np
n_draws = idata.posterior.dims["draw"]
keep_idx = np.random.choice(n_draws, size=500, replace=False)
idata_subset = idata.sel(draw=keep_idx)
```

### Extracting Samples for Custom Analysis

```python
# Get flattened samples for external tools
samples_dict = {
    var: idata.posterior[var].stack(sample=("chain", "draw")).values
    for var in ["alpha", "beta", "sigma"]
}

# Or use az.extract
samples = az.extract(idata, var_names=["alpha", "beta"])
```

### Combining Results from Multiple Runs

```python
# Concatenate chains from different runs
idata_combined = az.concat([idata1, idata2], dim="chain")

# Be careful: this assumes same model and convergence
```

---

## Common Interpretation Patterns

### Pattern: Multimodal Posterior

**Symptoms:**
- Bimodal density in `plot_trace`
- Separated clusters in `plot_pair`
- Very high R-hat

**Diagnosis:**
```python
az.plot_pair(idata, var_names=["theta"], marginals=True)
az.plot_trace(idata, var_names=["theta"], compact=False)
```

**Causes and fixes:**
1. **Label switching in mixtures**: Add ordering constraint
2. **Multiple solutions**: May be valid—interpret carefully or use additional constraints
3. **Insufficient warmup**: Increase `tune`

### Pattern: Funnel Geometry

**Symptoms:**
- Divergences clustered at low scale parameter values
- Funnel shape in pair plot of location vs scale
- Low ESS for scale parameters

**Diagnosis:**
```python
az.plot_pair(idata, var_names=["mu_group", "sigma_group"], divergences=True)
```

**Fix:** Use non-centered parameterization (see [gotchas.md](gotchas.md))

### Pattern: Poor Tail Sampling

**Symptoms:**
- `ess_tail` much lower than `ess_bulk`
- Credible intervals unreliable
- Autocorrelation high at extremes

**Diagnosis:**
```python
az.plot_ess(idata, kind="quantile")  # Check ESS at different quantiles
```

**Fixes:**
1. Increase samples
2. Increase `target_accept` (e.g., 0.95)
3. Reparameterize

### Pattern: Scale Parameter at Boundary

**Symptoms:**
- Scale parameter (σ) posterior piled up near zero
- May indicate overfitting/overparameterization

**Diagnosis:**
```python
az.plot_posterior(idata, var_names=["sigma"])
```

**Interpretation:**
- If σ → 0: Group effects explain almost no variance—simplify model
- Check if prior is appropriate

### Pattern: Strong Parameter Correlations

**Symptoms:**
- Nearly linear relationship in pair plot
- Can indicate non-identifiability

**Diagnosis:**
```python
az.plot_pair(idata, var_names=["alpha", "beta"])
```

**Fixes:**
1. Center predictors
2. Use sum-to-zero constraints
3. Consider if parameters are actually identifiable

---

## Publication-Quality Figures

### Style Configuration

```python
import arviz as az
import matplotlib.pyplot as plt

# Set ArviZ style
az.style.use("arviz-darkgrid")  # or "arviz-whitegrid", "arviz-white"

# Custom style
az.rcParams["plot.max_subplots"] = 40
az.rcParams["stats.hdi_prob"] = 0.94
az.rcParams["stats.ic_scale"] = "log"  # for LOO/WAIC
```

### Figure Sizing and Layout

```python
# Control figure size
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
az.plot_posterior(idata, var_names=["beta"], ax=axes.flatten())
plt.tight_layout()

# Or let ArviZ handle it
axes = az.plot_posterior(idata, var_names=["beta"], figsize=(12, 4))
```

### Combining Multiple Plots

```python
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

fig = plt.figure(figsize=(14, 10))
gs = GridSpec(2, 2, figure=fig)

# Trace plot
ax1 = fig.add_subplot(gs[0, :])
az.plot_trace(idata, var_names=["beta"], compact=True, combined=True, ax=ax1)

# Posterior
ax2 = fig.add_subplot(gs[1, 0])
az.plot_posterior(idata, var_names=["beta"], ax=ax2)

# PPC
ax3 = fig.add_subplot(gs[1, 1])
az.plot_ppc(idata, kind="cumulative", ax=ax3)

plt.tight_layout()
```

### Saving Figures

```python
# Save with high resolution
fig = az.plot_posterior(idata, var_names=["beta"])
plt.savefig("posterior.png", dpi=300, bbox_inches="tight")
plt.savefig("posterior.pdf", bbox_inches="tight")  # Vector format
```

### LaTeX Labels

```python
import matplotlib.pyplot as plt
plt.rcParams["text.usetex"] = True

# Use LaTeX in labels
az.plot_posterior(
    idata,
    var_names=["beta"],
    labeller=az.labels.MapLabeller(var_name_map={"beta": r"$\beta$"})
)
```

---

## Quick Reference: Which Plot When?

| Question | Plot | Function |
|----------|------|----------|
| Did MCMC converge? | Trace, Rank | `plot_trace`, `plot_rank` |
| Are there divergences? | Pair with divergences | `plot_pair(..., divergences=True)` |
| Is ESS adequate? | ESS evolution | `plot_ess(kind="evolution")` |
| Is mixing efficient? | Autocorrelation | `plot_autocorr` |
| Is HMC healthy? | Energy | `plot_energy` |
| Does model fit? | PPC | `plot_ppc` |
| Is model calibrated? | LOO-PIT | `plot_loo_pit` |
| What are the estimates? | Posterior | `plot_posterior` |
| Compare group effects? | Forest | `plot_forest` |
| Parameters correlated? | Pair | `plot_pair` |
| Which model is best? | Compare | `plot_compare` |
| Which points influential? | Pareto k | `plot_khat` |
| Prior sensible? | Prior predictive | `plot_ppc(..., group="prior")` |
