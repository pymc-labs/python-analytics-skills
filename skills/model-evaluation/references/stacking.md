# Model Averaging and Stacking Reference (ArviZ 1.1)

## Overview

Model averaging combines predictions from multiple models rather than selecting a single "best" model. This accounts for model uncertainty and often produces better predictions.

## Stacking

### Theory

Stacking finds weights w_1, ..., w_K that minimize the KL divergence from the combined predictive distribution to the true data-generating process:

```
maximize sum_i log(sum_k w_k * p(y_i | y_{-i}, M_k))
```

where p(y_i | y_{-i}, M_k) is the LOO predictive density for observation i under model k.

### Usage with ArviZ 1.1

```python
import arviz as az

# Compare with stacking weights (default)
comparison = az.compare({
    "model_a": dt_a,
    "model_b": dt_b,
    "model_c": dt_c,
}, method="stacking")

# Extract weights
weights = comparison["weight"]
print(weights)
# model_a    0.65
# model_b    0.35
# model_c    0.00
```

### Properties of Stacking

- Weights sum to 1
- Weights can be exactly 0 (model is dominated)
- Optimal when the true model is NOT in the candidate set (the M-open view)
- Does not require models to be nested
- Invariant to the inclusion of duplicate or very similar models

## Pseudo-BMA+ (Bayesian Bootstrap)

### Theory

Pseudo-BMA uses the Bayesian bootstrap to estimate uncertainty in model weights:

1. Resample ELPD values with Dirichlet weights
2. Compute model selection for each bootstrap replicate
3. Average across replicates

### Usage

```python
comparison = az.compare({
    "model_a": dt_a,
    "model_b": dt_b,
}, method="BB-pseudo-BMA")

weights = comparison["weight"]
```

### Properties of Pseudo-BMA+

- Weights are always positive (no exact zeros)
- Provides implicit uncertainty over weights
- More robust to outlier ELPD values
- Still sums to 1

## When to Use Which Method

| Scenario | Recommended Method |
|---|---|
| Default / prediction focus | Stacking |
| Want uncertainty in weights | Pseudo-BMA+ |
| Some models are clearly dominated | Stacking (gives them 0 weight) |
| All models are comparably good | Either works |
| Comparing > 5 models | Stacking (more stable) |
| Want Bayesian interpretation of weights | Pseudo-BMA+ |

## Generating Averaged Predictions

ArviZ provides model weights but does not directly generate averaged predictions. You construct them manually:

### Posterior Predictive Averaging

```python
import numpy as np
import xarray as xr

# Get posterior predictive from each model
pp_a = dt_a["posterior_predictive"]["y"].values  # shape: (chain, draw, obs)
pp_b = dt_b["posterior_predictive"]["y"].values
pp_c = dt_c["posterior_predictive"]["y"].values

# Get stacking weights
w = comparison["weight"].values  # [0.65, 0.35, 0.0]

# Method 1: Weighted mixture of samples
# Subsample from each model proportional to weight
n_samples = pp_a.shape[0] * pp_a.shape[1]  # total chain * draw
n_a = int(np.round(w[0] * n_samples))
n_b = int(np.round(w[1] * n_samples))
n_c = n_samples - n_a - n_b

# Flatten chain/draw dimensions
pp_a_flat = pp_a.reshape(-1, pp_a.shape[-1])
pp_b_flat = pp_b.reshape(-1, pp_b.shape[-1])
pp_c_flat = pp_c.reshape(-1, pp_c.shape[-1])

# Sample from each
rng = np.random.default_rng(42)
idx_a = rng.choice(pp_a_flat.shape[0], size=n_a, replace=True)
idx_b = rng.choice(pp_b_flat.shape[0], size=n_b, replace=True)
idx_c = rng.choice(pp_c_flat.shape[0], size=n_c, replace=True)

pp_averaged = np.concatenate([
    pp_a_flat[idx_a],
    pp_b_flat[idx_b],
    pp_c_flat[idx_c],
], axis=0)
```

### Point Prediction Averaging

```python
# Simpler: weighted mean of posterior predictive means
mean_a = pp_a.mean(axis=(0, 1))
mean_b = pp_b.mean(axis=(0, 1))
mean_c = pp_c.mean(axis=(0, 1))

averaged_mean = w[0] * mean_a + w[1] * mean_b + w[2] * mean_c
```

## Equal-Weight Averaging

When you want to treat all models as equally plausible hypotheses (e.g., different scientific theories):

```python
n_models = 3
equal_weights = np.ones(n_models) / n_models
averaged_mean = (mean_a + mean_b + mean_c) / 3
```

This is rarely optimal for prediction but may be appropriate when models represent genuinely different scientific hypotheses.

## Stacking for New Data

Stacking weights are optimized for the training data. For genuine out-of-sample prediction:

1. Compute stacking weights on training data
2. Generate posterior predictive for new data from each model
3. Combine using the training-data weights

The weights do not need to be recomputed for new data — they represent the relative predictive skill of each model.

## Diagnostics

### Are Weights Stable?

```python
# Bootstrap the comparison to check weight stability
# Run compare() on subsets of observations
n_obs = len(y)
n_boot = 100
weight_samples = np.zeros((n_boot, n_models))

for i in range(n_boot):
    idx = rng.choice(n_obs, size=n_obs, replace=True)
    # Subset pointwise log-likelihood and recompute
    # (This is conceptual — actual implementation depends on your setup)
```

If weights vary substantially across bootstrap samples, the models are hard to distinguish.

### Is Averaging Better Than Selection?

Compare the LOO ELPD of the averaged model to the best single model. If the improvement is small, model selection (picking the best) may be simpler and nearly as good.

## Common Pitfalls

1. **Averaging parameters instead of predictions**: Never average regression coefficients across models with different structures. Average predictions instead.
2. **Ignoring LOO warnings**: If any model has Pareto k warnings, fix those before computing stacking weights.
3. **Too many similar models**: Including many near-identical models can dilute weights away from genuinely different models. Stacking is relatively robust to this, but pseudo-BMA is not.
4. **Using weights as model probabilities**: Stacking weights are not posterior model probabilities. They are optimal prediction weights.
