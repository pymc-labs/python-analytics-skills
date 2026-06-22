# LOO-CV Deep Technical Reference (ArviZ 1.1)

## Overview

Leave-one-out cross-validation (LOO-CV) estimates out-of-sample predictive accuracy without actually refitting the model. ArviZ 1.1 uses Pareto-smoothed importance sampling (PSIS) for efficient computation.

CRITICAL: PyMC 6 returns DataTree by default, and ArviZ 1.1 is DataTree-first while still accepting idata-like inputs. `az.waic` is removed — use LOO exclusively.

## az.loo()

```python
import arviz as az

# dt is a DataTree from pm.sample()
loo_result = az.loo(dt, pointwise=False)
```

### Parameters

- **dt**: xarray.DataTree containing `log_likelihood` group
- **pointwise**: If True, return pointwise LOO values (default: False)

### Return Value: ELPDData

```python
loo_result.elpd    # Expected log pointwise predictive density
loo_result.se          # Standard error of ELPD
loo_result.p       # Effective number of parameters
loo_result.pareto_k    # Pareto k diagnostic values (one per observation)
loo_result.n_data_points  # Number of observations
```

### Ensuring log_likelihood Exists

```python
with pm.Model() as model:
    ...
    idata = pm.sample()

# PyMC 6: compute log_likelihood explicitly after sampling
pm.compute_log_likelihood(idata, model=model)

# Verify
print("log_likelihood" in idata.children)
```

## Pareto k Diagnostics

Each observation gets a Pareto k value indicating how well PSIS approximates the LOO posterior for that observation.

### Thresholds

| k range | Quality | Reliability | Action |
|---|---|---|---|
| (-inf, 0.5] | Good | Reliable | No action needed |
| (0.5, 0.7] | OK | Usable | Monitor, acceptable for most purposes |
| (0.7, 1.0] | Bad | Unreliable | Use moment matching or k-fold |
| (1.0, inf) | Very bad | Fails | Must use k-fold or exact refit |

### Diagnosing High k Values

```python
import numpy as np

k_values = loo_result.pareto_k.values

# Find problematic observations
bad_idx = np.where(k_values > 0.7)[0]
print(f"Problematic observations: {bad_idx}")
print(f"Their k values: {k_values[bad_idx]}")

# Visualize
az.plot_khat(loo_result)
```

### Common Causes of High k

1. **Outliers**: Observations far from the model's predictions
2. **Influential observations**: Single points that strongly affect the posterior
3. **Model misspecification**: Likelihood doesn't match data-generating process
4. **Overfitting**: Too many parameters relative to data

## Moment Matching

Improves PSIS estimates for observations with high Pareto k by adjusting importance weights using moment matching.

```python
loo_mm = az.loo_moment_match(dt)
```

This is much faster than k-fold because it reweights existing posterior samples rather than refitting the model. Try this first when k values are marginal (0.7-1.0).

## K-Fold Cross-Validation

When PSIS fails (many k > 1.0), use exact k-fold CV:

```python
kfold_result = az.loo_kfold(dt, K=10)
```

### Choosing K

- K=10: Good default, balances accuracy and computation
- K=5: Faster, slightly less accurate
- K=N (leave-one-out): Exact LOO, very expensive

### When to Use K-Fold

- Many observations with k > 0.7
- Moment matching doesn't fix the k values
- Model has discrete parameters that make PSIS unreliable

## Subsampled LOO

For large datasets (n > 10000), subsample for speed:

```python
loo_sub = az.loo_subsample(dt, observations=1000)
```

Uses a subsample of observations to estimate ELPD with controlled approximation error. The standard error accounts for the subsampling.

## Exact Refit LOO (reloo)

Refit the model exactly for observations with high Pareto k:

```python
loo_exact = az.reloo(dt, loo_result, model=model)
```

This literally refits the model leaving out each problematic observation. Most expensive option but most reliable.

## Effective Number of Parameters (p)

`p` estimates the effective model complexity:

- `p < actual_params`: Model is well-regularized (priors doing work)
- `p ~ actual_params`: Priors are non-informative
- `p > actual_params`: Model misspecification or influential observations

A large p relative to the number of observations signals potential overfitting.

## LOO-PIT (Probability Integral Transform)

```python
az.plot_loo_pit(dt, var_names=["y"])
```

The LOO-PIT plot shows the distribution of leave-one-out probability integral transform values. Under a well-calibrated model, these should be uniform.

### Interpreting Patterns

- **Uniform**: Well-calibrated model
- **U-shaped**: Underdispersed (predictions too narrow)
- **Inverted U (hump)**: Overdispersed (predictions too wide)
- **Right-skewed**: Model systematically underpredicts
- **Left-skewed**: Model systematically overpredicts

## Scoring Rules

### Log Score (default)

The standard log predictive density. Sensitive to tail behavior — a single poorly predicted observation can dominate.

### CRPS (Continuous Ranked Probability Score)

```python
crps = az.loo_score(dt, score_func="crps")
```

Less sensitive to outliers than log score. Evaluates the full predictive distribution, not just the point prediction.

## LOO R-squared

```python
r2 = az.loo_r2(dt)
```

Bayesian R-squared computed via LOO predictions. Returns a distribution (not a point estimate), reflecting uncertainty in R-squared itself.

Interpretation:
- Distribution centered near 1: Model explains most variance
- Distribution centered near 0: Model explains little variance
- Wide distribution: Uncertain about explanatory power (small sample or complex model)
