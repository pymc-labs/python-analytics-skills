# R2D2M2CP vs Regularized Horseshoe — Shrinkage Prior Comparison

## Overview

Both R2D2M2CP and the regularized horseshoe are shrinkage priors for regression. They handle high-dimensional regression (many predictors) by shrinking coefficients toward zero. They differ in philosophy, behavior, and when each is appropriate.

## R2D2M2CP Prior

### Concept

R2D2M2CP (R2-induced Dirichlet Decomposition with Multi-Modal Concentrated Priors) reasons about the total variance explained (R2) and decomposes it across predictors.

- Place a prior on R2 directly (intuitive)
- Decompose R2 into shares per predictor via Dirichlet
- Derive coefficient priors from these shares

### Implementation

```python
import pymc as pm
import pymc_extras as pmx

with pm.Model() as r2d2_model:
    beta, r2 = pmx.R2D2M2CP(
        "beta",
        X,                          # Design matrix (n x p)
        y_obs,                      # Observed response
        r2=0.5,                     # Prior mean for R2 (0 to 1)
        variance_concentration=None, # None = equal concentration
    )

    mu = pm.math.dot(X, beta)
    sigma = pm.HalfNormal("sigma", sigma=1)
    y = pm.Normal("y", mu=mu, sigma=sigma, observed=y_obs)
```

### Parameters

- **r2**: Prior expectation for R-squared. Set based on domain knowledge:
  - 0.1-0.3: Expect low predictability (social science, noisy data)
  - 0.3-0.7: Moderate predictability (typical regression)
  - 0.7-0.9: High predictability (physical sciences, well-understood systems)

- **variance_concentration**: Controls how R2 is distributed across predictors:
  - `None` or equal: Each predictor gets equal prior share
  - Custom array: Encode prior beliefs about which predictors matter more

### Behavior

- Provides soft, global shrinkage
- All coefficients shrink together (global-local structure)
- Does NOT produce exact zeros — all coefficients remain nonzero
- Well-suited when most predictors have small but nonzero effects

## Regularized Horseshoe Prior

### Concept

The horseshoe prior has a sharp spike at zero and heavy tails, allowing both strong shrinkage of noise predictors and minimal shrinkage of signal predictors.

- Global shrinkage (tau): overall sparsity level
- Local shrinkage (lambda_j): per-coefficient, allows some to escape shrinkage
- Slab component (c): prevents unbounded coefficients (the "regularized" part)

### Implementation

```python
import pymc as pm
import numpy as np

with pm.Model() as horseshoe_model:
    n, p = X.shape

    # Expected number of relevant predictors
    p0 = 5  # domain knowledge: ~5 of p predictors matter
    tau0 = p0 / (p - p0) / np.sqrt(n)

    # Global shrinkage
    tau = pm.HalfCauchy("tau", beta=tau0)

    # Local shrinkage (one per predictor)
    lam = pm.HalfCauchy("lam", beta=1, shape=p)

    # Regularizing slab
    c2 = pm.InverseGamma("c2", alpha=2, beta=1)
    lam_tilde = lam * pm.math.sqrt(c2 / (c2 + tau**2 * lam**2))

    # Coefficients
    beta = pm.Normal("beta", mu=0, sigma=tau * lam_tilde, shape=p)

    mu = pm.math.dot(X, beta)
    sigma = pm.HalfNormal("sigma", sigma=1)
    y = pm.Normal("y", mu=mu, sigma=sigma, observed=y_obs)
```

### Key Tuning Parameters

- **p0** (expected number of relevant predictors): The most important choice. Determines global shrinkage strength.
  - Too small: over-shrinks real signals
  - Too large: under-shrinks noise
  - When unsure, err on the side of larger p0

- **c2** (slab scale): Controls the scale of the "slab" for non-zero coefficients.
  - `InverseGamma(2, 1)`: Weakly informative (default)
  - Smaller beta: tighter slab, stronger regularization even for signal predictors

### Behavior

- Creates near-exact sparsity (some coefficients very close to zero)
- Heavy Cauchy tails allow strong signals to escape shrinkage
- The slab prevents coefficients from becoming unreasonably large
- Well-suited when most predictors are truly irrelevant (sparse signals)

## Head-to-Head Comparison

| Aspect | R2D2M2CP | Regularized Horseshoe |
|---|---|---|
| **Philosophy** | "How much variance does the model explain?" | "How many predictors are relevant?" |
| **Shrinkage type** | Soft, continuous | Spike-and-slab-like (near-exact zeros) |
| **Sparsity** | Not sparse — all coefficients nonzero | Near-sparse — many coefficients near zero |
| **Key input** | Prior on R2 (intuitive) | Expected # of relevant predictors (p0) |
| **Best for** | Dense signals, moderate p | Sparse signals, large p |
| **Ease of use** | Single function call | Manual implementation |
| **Sampling** | Generally well-behaved | Can have divergences (funnel geometry) |
| **Interpretability** | R2 is intuitive | Sparsity pattern is interpretable |

## Decision Guide

```
How many predictors are truly relevant?
├── Most predictors have small effects → R2D2M2CP
│   (dense signal, soft shrinkage)
├── Few predictors have large effects, rest are noise → Horseshoe
│   (sparse signal, hard shrinkage)
└── Unsure → Try both, compare via LOO

How many predictors do you have?
├── p < 20 → Either works; R2D2M2CP is simpler
├── 20 < p < 200 → Horseshoe if sparse, R2D2M2CP if dense
└── p > 200 → Horseshoe (designed for high-dimensional sparse problems)

Do you have strong prior knowledge about R2?
├── YES → R2D2M2CP (leverages this directly)
└── NO → Horseshoe (only needs p0 estimate)
```

## Sampling Considerations

### R2D2M2CP

- Generally samples well with NUTS
- Fewer divergences than horseshoe
- May need more warmup when p is large

### Horseshoe

- Prone to funnel geometry (tau-beta interaction)
- Reparameterization helps:
  ```python
  # Non-centered parameterization
  beta_raw = pm.Normal("beta_raw", 0, 1, shape=p)
  beta = pm.Deterministic("beta", beta_raw * tau * lam_tilde)
  ```
- Increase `target_accept` to 0.95-0.99 if divergences occur
- Consider using nutpie sampler for better performance

## Combining with Other Model Components

Both priors work in larger models:

```python
# R2D2M2CP with hierarchical structure
with pm.Model() as hier_r2d2:
    # Group-level intercepts
    mu_group = pm.Normal("mu_group", 0, 5, shape=n_groups)

    # R2D2M2CP for predictors
    beta, r2 = pmx.R2D2M2CP("beta", X, y_obs, r2=0.3)

    mu = mu_group[group_idx] + pm.math.dot(X, beta)
    ...
```

```python
# Horseshoe with splines
with pm.Model() as sparse_spline:
    # Horseshoe for linear predictors
    # ... (horseshoe setup as above)
    linear_part = pm.math.dot(X, beta)

    # Spline for nonlinear effect
    w = pm.GaussianRandomWalk("w", sigma=tau_spline, shape=n_basis)
    spline_part = pm.math.dot(basis_matrix, w)

    mu = linear_part + spline_part
    ...
```
