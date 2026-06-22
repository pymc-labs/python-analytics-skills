---
name: pymc-extras
description: >
  Load when the user is working with pymc-extras (pmx) features: splines / BSplineBasis,
  distributional regression / GAMLSS, R2D2M2CP or horseshoe priors, discrete variable
  marginalization, or Laplace approximation via fit_laplace. Triggers include: pymc_extras,
  pymc-extras, pmx, splines, BSplineBasis, distributional regression, GAMLSS, R2D2,
  horseshoe (regularized/Finnish), marginalize, fit_laplace, penalized splines.
---

# PyMC-Extras (pmx) — Advanced Extensions

```python
import pymc_extras as pmx
```

## Splines

### BSplineBasis

Create B-spline basis matrices for nonlinear effects:

```python
import numpy as np
import pymc as pm
import pymc_extras as pmx

x = np.linspace(0, 1, 100)

# Create B-spline basis with 10 knots, degree 3 (cubic)
B = pmx.BSplineBasis(n_knots=10, degree=3)
basis_matrix = B.build(x)  # shape: (100, n_basis)

with pm.Model() as spline_model:
    # Coefficients for each basis function
    w = pm.Normal("w", mu=0, sigma=1, shape=basis_matrix.shape[1])
    mu = pm.math.dot(basis_matrix, w)
    sigma = pm.HalfNormal("sigma", sigma=1)
    y = pm.Normal("y", mu=mu, sigma=sigma, observed=y_obs)
```

### Penalized Splines with Smoothing Priors

Prevent overfitting by penalizing roughness via random walk priors on coefficients:

```python
with pm.Model() as penalized_spline:
    # Smoothing parameter controls wiggliness
    tau = pm.HalfCauchy("tau", beta=1)

    # Random walk prior on spline coefficients (penalizes second differences)
    w = pm.GaussianRandomWalk("w", sigma=tau, shape=basis_matrix.shape[1])

    mu = pm.math.dot(basis_matrix, w)
    sigma = pm.HalfNormal("sigma", sigma=1)
    y = pm.Normal("y", mu=mu, sigma=sigma, observed=y_obs)
```

### ZeroSumNormalBasis

For identifiable models where basis coefficients must sum to zero:

```python
# Ensures sum-to-zero constraint on spline coefficients
B_zs = pmx.ZeroSumNormalBasis(n_knots=10, degree=3)
```

See `references/splines.md` for detailed spline reference.

## Distributional Regression (GAMLSS-style)

Model ALL distribution parameters as functions of covariates, not just the mean.

### Basic Pattern

```python
with pm.Model() as dist_reg:
    # Model both mean AND variance as functions of covariates
    # Location model
    beta_mu = pm.Normal("beta_mu", 0, 1, shape=X.shape[1])
    mu = pm.math.dot(X, beta_mu)

    # Scale model (log link to ensure positivity)
    beta_sigma = pm.Normal("beta_sigma", 0, 0.5, shape=Z.shape[1])
    log_sigma = pm.math.dot(Z, beta_sigma)
    sigma = pm.math.exp(log_sigma)

    y = pm.Normal("y", mu=mu, sigma=sigma, observed=y_obs)
```

### Common Distributional Patterns

```python
# Beta regression: both mu and phi vary with covariates
with pm.Model() as beta_reg:
    # Mean model (logit link)
    beta_mu = pm.Normal("beta_mu", 0, 1, shape=X.shape[1])
    mu = pm.math.sigmoid(pm.math.dot(X, beta_mu))

    # Precision model (log link)
    beta_phi = pm.Normal("beta_phi", 0, 0.5, shape=Z.shape[1])
    phi = pm.math.exp(pm.math.dot(Z, beta_phi))

    # Reparameterize: alpha = mu * phi, beta = (1-mu) * phi
    y = pm.Beta("y", alpha=mu * phi, beta=(1 - mu) * phi, observed=y_obs)
```

```python
# Negative binomial: mu and alpha both depend on covariates
with pm.Model() as nb_dist:
    beta_mu = pm.Normal("beta_mu", 0, 1, shape=X.shape[1])
    mu = pm.math.exp(pm.math.dot(X, beta_mu))

    beta_alpha = pm.Normal("beta_alpha", 0, 0.5, shape=Z.shape[1])
    alpha = pm.math.exp(pm.math.dot(Z, beta_alpha))

    y = pm.NegativeBinomial("y", mu=mu, alpha=alpha, observed=y_obs)
```

See `references/distributional.md` for more patterns.

## R2D2M2CP Prior

A prior for regression coefficients that reasons about the proportion of variance explained (R2) and decomposes it across predictors.

```python
with pm.Model() as r2d2_model:
    # R2D2M2CP: specify expected R2 and concentration
    beta, r2 = pmx.R2D2M2CP(
        "beta",
        X,                      # Design matrix
        y_obs,                  # Observed response
        r2=0.5,                 # Prior mean for R2
        variance_concentration=None,  # Equal concentration (default)
    )
    mu = pm.math.dot(X, beta)
    sigma = pm.HalfNormal("sigma", sigma=1)
    y = pm.Normal("y", mu=mu, sigma=sigma, observed=y_obs)
```

### When to Use R2D2M2CP vs Horseshoe

| Aspect | R2D2M2CP | Horseshoe |
|---|---|---|
| Interpretability | Reasons about R2 directly | Reasons about shrinkage |
| Sparsity | Soft shrinkage | Strong sparsity (near-zero coefficients) |
| Best for | Dense signals, moderate p | Sparse signals, large p |
| Tuning | Prior on R2 + concentration | Global/local shrinkage scales |
| Implementation | `pmx.R2D2M2CP()` | Manual or pmx helper |

For general prior specification guidance and elicitation workflows, see the [prior-elicitation skill](../prior-elicitation/SKILL.md).

## Regularized Horseshoe Prior

For sparse regression where most coefficients are expected to be near zero:

```python
with pm.Model() as horseshoe_model:
    n, p = X.shape

    # Expected number of relevant predictors
    p0 = 5
    # Global shrinkage — controls overall sparsity
    tau0 = p0 / (p - p0) / np.sqrt(n)
    tau = pm.HalfCauchy("tau", beta=tau0)

    # Local shrinkage — per-coefficient
    lam = pm.HalfCauchy("lam", beta=1, shape=p)

    # Regularized: slab component prevents unbounded coefficients
    c2 = pm.InverseGamma("c2", alpha=2, beta=1)
    lam_tilde = lam * pm.math.sqrt(c2 / (c2 + tau**2 * lam**2))

    beta = pm.Normal("beta", mu=0, sigma=tau * lam_tilde, shape=p)
    mu = pm.math.dot(X, beta)
    sigma = pm.HalfNormal("sigma", sigma=1)
    y = pm.Normal("y", mu=mu, sigma=sigma, observed=y_obs)
```

See `references/r2d2_horseshoe.md` for detailed comparison.

## Marginalization of Discrete Parameters

`pmx.marginalize()` analytically integrates out discrete latent variables, avoiding the need for specialized samplers.

```python
with pm.Model() as mixture:
    # Mixture weights
    w = pm.Dirichlet("w", a=np.ones(3))

    # Component means
    mu = pm.Normal("mu", mu=[-5, 0, 5], sigma=1, shape=3)
    sigma = pm.HalfNormal("sigma", sigma=1, shape=3)

    # Discrete component assignment (will be marginalized)
    comp = pm.Categorical("comp", p=w, shape=len(y_obs))

    # Likelihood
    y = pm.Normal("y", mu=mu[comp], sigma=sigma[comp], observed=y_obs)

    # Marginalize out the discrete variable
    pmx.marginalize(model=mixture, rvs_to_marginalize=["comp"])

    # Now sample with standard NUTS — no need for CategoricalGibbsMetropolis
    idata = pm.sample()
```

### Supported Distributions for Marginalization

- `Categorical` assignments in mixture models
- `Bernoulli` indicators in spike-and-slab
- Discrete latent states in hidden Markov models

### Benefits

- Enables NUTS sampling for models with discrete parameters
- Often dramatically improves sampling efficiency
- Removes label switching issues in mixture models

## Laplace Approximation

Fast approximate inference via second-order Taylor expansion at the MAP:

```python
with pm.Model() as model:
    beta = pm.Normal("beta", 0, 1, shape=5)
    mu = pm.math.dot(X, beta)
    sigma = pm.HalfNormal("sigma", sigma=1)
    y = pm.Normal("y", mu=mu, sigma=sigma, observed=y_obs)

    # Laplace approximation
    idata_laplace = pmx.fit_laplace(model)
```

### When to Use Laplace

| Scenario | Use Laplace? |
|---|---|
| Quick approximate inference | Yes |
| Large datasets, simple model | Yes — fast and accurate |
| Posterior is near-Gaussian | Yes — Laplace is exact for Gaussian posteriors |
| Multimodal posterior | No — Laplace finds only one mode |
| Heavy-tailed posteriors | No — underestimates tail uncertainty |
| Hierarchical models with few groups | No — posterior is often non-Gaussian |
| Model comparison (for screening) | Yes — fast screening before full MCMC |

### Comparing Laplace to MCMC

```python
# Quick Laplace fit for model development
idata_laplace = pmx.fit_laplace(model)

# Full MCMC for final inference
idata_mcmc = pm.sample()

# Compare: Laplace posteriors should be similar to MCMC
# if the approximation is good
import arviz as az
az.plot_forest(
    [idata_laplace, idata_mcmc],
    model_names=["Laplace", "MCMC"],
)
```

Laplace is excellent for rapid iteration during model development. Use full MCMC for final inference and reporting.
