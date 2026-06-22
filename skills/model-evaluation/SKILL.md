---
name: model-evaluation
description: >
  Load when the user is comparing Bayesian models, computing LOO-CV / ELPD, calling
  arviz_stats.loo or arviz_stats.compare, doing model stacking/averaging, or computing
  Bayes factors. Covers the ArviZ 1.1 LOO/ELPD/stacking APIs exclusively (no waic).
  Triggers include: model comparison, LOO, ELPD, compare, loo_expectations,
  loo_metrics, loo_r2, Pareto k, stacking, Bayes factor, cross-validation,
  predictive accuracy, information criterion.
---

# Model Evaluation and Comparison (ArviZ 1.1)

CRITICAL: PyMC 6 returns xarray DataTree objects by default, and ArviZ 1.1 stats/plots are DataTree-first while still accepting idata-like inputs. `az.waic` is removed entirely — use PSIS-LOO-CV exclusively. Default credible intervals are 0.89 ETI, controlled via `ci_prob=` and `ci_kind=` for summaries/plots; low-level `hdi()` uses `prob=`.

For model building context, prior selection, and convergence diagnostics, see the [pymc-modeling skill](../pymc-modeling/SKILL.md).

## LOO-CV with ArviZ 1.1

Leave-one-out cross-validation via Pareto-smoothed importance sampling (PSIS).

```python
import arviz_stats as azs
import arviz_plots as azp

# dt is a DataTree from pm.sample()
loo_result = azs.loo(dt)
print(loo_result)
# Returns: ELPDData with elpd, se, p, n_data_points, pareto_k

# Equivalent via the xarray accessor when arviz_stats is imported:
loo_result = dt.azstats.loo()
```

### Pareto k Diagnostics

Pareto k values indicate reliability of PSIS approximation for each observation:

| k value | Interpretation | Action |
|---|---|---|
| k < 0.5 | Good | LOO estimate reliable |
| 0.5 < k < 0.7 | Marginal | Results usable but less accurate |
| 0.7 < k < 1.0 | Bad | Estimate unreliable — use moment matching or k-fold |
| k > 1.0 | Very bad | PSIS fails entirely — must use k-fold CV |

```python
# Check Pareto k values
print(loo_result.pareto_k)

# Plot Pareto k diagnostics
azp.plot_khat(loo_result)

# Count problematic observations
import numpy as np
k_values = loo_result.pareto_k.values
print(f"k > 0.7: {np.sum(k_values > 0.7)} observations")
```

### What to Do When k > 0.7

1. Try moment matching first (fast, automatic)
2. If still bad, use k-fold cross-validation
3. Check if problematic observations are outliers — consider robust likelihood
4. Re-examine the model — high k often signals model misspecification

## Moment Matching

Automatically refit problematic observations using moment matching:

```python
# Requires log_likelihood in the DataTree
loo_mm = azs.loo_moment_match(dt)
```

This importance-weights the posterior for each problematic observation, improving the PSIS approximation without refitting the model. Much faster than k-fold.

## K-Fold Cross-Validation

When LOO is unreliable for many observations, use exact k-fold CV:

```python
# Perform 10-fold cross-validation
kfold_result = azs.loo_kfold(dt, K=10)
print(kfold_result)
```

This refits the model K times, so it is K times slower than LOO. Use only when LOO diagnostics indicate problems.

## compare() — Full Workflow

Compare multiple models on predictive accuracy:

```python
# dt1, dt2, dt3 are DataTree objects from pm.sample()
comparison = azs.compare(
    {"linear": dt1, "quadratic": dt2, "spline": dt3},
)
print(comparison)
```

Note: `compare` in ArviZ 1.1 only supports LOO, so the old `ic=` and `scale=` arguments have been dropped.

### Interpreting the Comparison Table

| Column | Meaning |
|---|---|
| `rank` | Model rank (0 = best) |
| `elpd` | Expected log pointwise predictive density |
| `p` | Effective number of parameters |
| `elpd_diff` | Difference in ELPD from the reference model |
| `weight` | Stacking weight (sums to 1) |
| `se` | Standard error of ELPD |
| `dse` | Standard error of the ELPD difference |
| `diag_elpd` | Pareto-k diagnostic issues for each model's ELPD |
| `diag_diff` | Small-data or practically-equivalent-difference diagnostics |

### Decision Rules

- `elpd_diff` = 0: reference/best model
- `|elpd_diff| < 4`: models are practically indistinguishable — prefer simpler one
- `|elpd_diff| > 4` and `|elpd_diff / dse| > 2`: meaningful difference in predictive accuracy
- Non-empty `diag_elpd`: LOO unreliable for this model — investigate Pareto k values

```python
# Visualize comparison
azp.plot_compare(comparison)

# Detailed forest plot of ELPD differences
azp.plot_elpd({"linear": dt1, "quadratic": dt2, "spline": dt3})
```

See `references/model_comparison.md` for detailed usage.

## Model Averaging

### Stacking Weights (Default)

Stacking minimizes KL divergence from the true predictive distribution to the weighted mixture. This is the recommended default.

```python
comparison = azs.compare({"m1": dt1, "m2": dt2, "m3": dt3})
# Stacking weights are in the "weight" column by default
print(comparison["weight"])
```

### Pseudo-BMA+ Weights

Alternative weighting based on Bayesian bootstrap of ELPD:

```python
comparison = azs.compare(
    {"m1": dt1, "m2": dt2, "m3": dt3},
    method="BB-pseudo-BMA",
)
```

### When to Use Which

| Method | Use When |
|---|---|
| Stacking | Default. Best for prediction when true model is not in the set |
| Pseudo-BMA+ | Want Bayesian uncertainty over weights |
| Equal weights | Models represent different scientific hypotheses to average over |

### Generating Averaged Predictions

```python
weights = comparison["weight"].values
# Manually mix posterior predictive samples
# weighted by stacking weights
```

See `references/stacking.md` for detailed averaging workflows.

## Bayes Factors via Bridge Sampling

Bayes factors compare marginal likelihoods. Conceptually different from LOO (predictive accuracy vs. evidence).

```python
# Bayes factors are difficult to compute reliably
# Bridge sampling is the most reliable method but requires specialized setup
# For most applied work, LOO-CV is preferred

# Approximate Bayes factor from LOO (rough):
# BF ~ exp(elpd_m1 - elpd_m2)
# This is a very rough approximation — use with caution
```

### Limitations of Bayes Factors

- Highly sensitive to prior specification (unlike LOO)
- Numerically unstable for complex models
- Penalize model complexity differently than LOO
- Not recommended for routine model comparison — prefer LOO

## LOO-PIT Calibration

LOO probability integral transform checks if the model is calibrated:

```python
azp.plot_loo_pit(dt, var_names=["observed_data_name"])
```

### Interpretation

- **Uniform histogram**: model is well-calibrated
- **U-shaped**: underdispersed predictions (too narrow)
- **Inverted U**: overdispersed predictions (too wide)
- **Skewed**: systematic bias in predictions

This is a powerful diagnostic that LOO uniquely provides — it checks calibration without held-out data.

## New ArviZ 1.1 Functions

### loo_expectations()

Compute LOO-weighted posterior expectations (mean, variance, quantile) for each observation. Requires both `posterior_predictive` and `log_likelihood` groups on the DataTree:

```python
# LOO-weighted posterior predictive mean for each observation
loo_mean = azs.loo_expectations(dt, kind="mean")
loo_var = azs.loo_expectations(dt, kind="var")
loo_q = azs.loo_expectations(dt, kind="quantile", probs=[0.055, 0.945])
```

### loo_metrics()

Compute common LOO-based predictive metrics (RMSE, MAE, etc.) from `posterior_predictive` and `log_likelihood`:

```python
metrics = azs.loo_metrics(dt, kind="rmse")
```

### .azstats xarray accessor

`import arviz_stats as azs` registers an `.azstats` accessor on `DataArray`, `Dataset`, and `DataTree`. This gives a fluent xarray-native interface alongside the function API:

```python
import arviz_stats as azs

dt.azstats.loo()                       # same as azs.loo(dt)
dt["posterior"].azstats.rhat()         # on a Dataset
dt["posterior"].azstats.ess()
dt["posterior"].azstats.summary()
dt["posterior"].azstats.hdi()
dt["posterior"].azstats.eti()
```

### loo_r2()

Bayesian R-squared via LOO:

```python
r2 = azs.loo_r2(dt)
print(f"LOO-R2: {r2.mean():.3f} [{r2.quantile(0.055):.3f}, {r2.quantile(0.945):.3f}]")
```

### loo_score()

Compute LOO-based scoring rules (CRPS, log score):

```python
score = azs.loo_score(dt, score_func="crps")
```

### loo_subsample()

LOO with subsampling for large datasets:

```python
# When n > 10000, subsample for speed
loo_sub = azs.loo_subsample(dt, observations=1000)
```

### reloo()

Exact refit LOO for observations with high Pareto k:

```python
# Refits the model for problematic observations
loo_exact = azs.reloo(dt, loo_result, model=model)
```

## Standard Evaluation Workflow

```python
import arviz_stats as azs
import arviz_plots as azp

# 1. Compute LOO
loo = azs.loo(dt)
print(loo)

# 2. Check Pareto k
azp.plot_khat(loo)

# 3. If k > 0.7, try moment matching
if (loo.pareto_k > 0.7).any():
    loo = azs.loo_moment_match(dt)

# 4. LOO-PIT calibration
azp.plot_loo_pit(dt, var_names=["y"])

# 5. Compare models
comparison = azs.compare({"model_a": dt_a, "model_b": dt_b})
azp.plot_compare(comparison)
print(comparison)

# 6. Predictive R2
r2 = azs.loo_r2(dt)
```
