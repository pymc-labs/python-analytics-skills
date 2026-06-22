# Distributional Regression Patterns (GAMLSS-style)

## Overview

In standard regression, only the mean of the response depends on covariates. In distributional regression (GAMLSS), ALL distribution parameters can depend on covariates — mean, variance, shape, etc.

This is useful when:
- Variance changes with covariates (heteroscedasticity)
- The shape of the distribution depends on predictors
- You want to model the full conditional distribution, not just the mean

## Basic Pattern: Normal with Varying Mean and Variance

```python
import pymc as pm
import numpy as np

with pm.Model() as gamlss_normal:
    # Design matrices
    # X: covariates for the mean
    # Z: covariates for the variance (can be same as X or different)

    # Location model (identity link)
    beta_mu = pm.Normal("beta_mu", 0, 1, shape=X.shape[1])
    mu = pm.math.dot(X, beta_mu)

    # Scale model (log link ensures positivity)
    beta_sigma = pm.Normal("beta_sigma", 0, 0.5, shape=Z.shape[1])
    log_sigma = pm.math.dot(Z, beta_sigma)
    sigma = pm.math.exp(log_sigma)

    # Likelihood
    y = pm.Normal("y", mu=mu, sigma=sigma, observed=y_obs)
```

### Key Design Choices

1. **Link functions**: Use appropriate links to constrain parameters
   - Mean: identity (no link) or log for positive outcomes
   - Scale/variance: log link (ensures positivity)
   - Shape: log link for positive parameters
   - Probability: logit link for [0,1] parameters

2. **Prior scale**: Priors on log/logit-scale parameters should be tighter than priors on identity-scale parameters. `Normal(0, 0.5)` on the log scale corresponds to a multiplicative effect of roughly exp(1) = 2.7x.

## Gamma Regression with Varying Shape

```python
with pm.Model() as gamma_gamlss:
    # Mean model (log link)
    beta_mu = pm.Normal("beta_mu", 0, 1, shape=X.shape[1])
    mu = pm.math.exp(pm.math.dot(X, beta_mu))

    # Shape model (log link)
    beta_alpha = pm.Normal("beta_alpha", 0, 0.5, shape=Z.shape[1])
    alpha = pm.math.exp(pm.math.dot(Z, beta_alpha))

    # Gamma parameterized by mean and shape
    # beta = alpha / mu
    y = pm.Gamma("y", alpha=alpha, beta=alpha / mu, observed=y_obs)
```

## Beta Regression (Proportions)

```python
with pm.Model() as beta_gamlss:
    # Mean model (logit link for [0,1] constraint)
    beta_mu = pm.Normal("beta_mu", 0, 1, shape=X.shape[1])
    mu = pm.math.sigmoid(pm.math.dot(X, beta_mu))

    # Precision model (log link)
    beta_phi = pm.Normal("beta_phi", 0, 0.5, shape=Z.shape[1])
    phi = pm.math.exp(pm.math.dot(Z, beta_phi))

    # Beta: alpha = mu * phi, beta = (1 - mu) * phi
    y = pm.Beta("y", alpha=mu * phi, beta=(1 - mu) * phi, observed=y_obs)
```

## Negative Binomial (Count Data with Overdispersion)

```python
with pm.Model() as nb_gamlss:
    # Mean model (log link for counts)
    beta_mu = pm.Normal("beta_mu", 0, 1, shape=X.shape[1])
    mu = pm.math.exp(pm.math.dot(X, beta_mu))

    # Overdispersion model (log link)
    beta_alpha = pm.Normal("beta_alpha", 0, 0.5, shape=Z.shape[1])
    alpha = pm.math.exp(pm.math.dot(Z, beta_alpha))

    y = pm.NegativeBinomial("y", mu=mu, alpha=alpha, observed=y_obs)
```

## Student-t (Robust Regression with Varying Degrees of Freedom)

```python
with pm.Model() as studentt_gamlss:
    # Location model
    beta_mu = pm.Normal("beta_mu", 0, 1, shape=X.shape[1])
    mu = pm.math.dot(X, beta_mu)

    # Scale model (log link)
    beta_sigma = pm.Normal("beta_sigma", 0, 0.5, shape=Z.shape[1])
    sigma = pm.math.exp(pm.math.dot(Z, beta_sigma))

    # Degrees of freedom (log link, shifted to ensure nu > 2)
    beta_nu = pm.Normal("beta_nu", 0, 0.3, shape=W.shape[1])
    nu = 2 + pm.math.exp(pm.math.dot(W, beta_nu))

    y = pm.StudentT("y", nu=nu, mu=mu, sigma=sigma, observed=y_obs)
```

## With Splines

Combine distributional regression with nonlinear effects using splines:

```python
import pymc_extras as pmx

B = pmx.BSplineBasis(n_knots=10, degree=3)
basis_matrix = B.build(x)

with pm.Model() as spline_gamlss:
    # Nonlinear mean via spline
    tau_mu = pm.HalfCauchy("tau_mu", beta=1)
    w_mu = pm.GaussianRandomWalk("w_mu", sigma=tau_mu, shape=basis_matrix.shape[1])
    mu = pm.math.dot(basis_matrix, w_mu)

    # Nonlinear variance via spline
    tau_sigma = pm.HalfCauchy("tau_sigma", beta=0.5)
    w_sigma = pm.GaussianRandomWalk("w_sigma", sigma=tau_sigma, shape=basis_matrix.shape[1])
    log_sigma = pm.math.dot(basis_matrix, w_sigma)
    sigma = pm.math.exp(log_sigma)

    y = pm.Normal("y", mu=mu, sigma=sigma, observed=y_obs)
```

## Mixture of Experts

A special case where mixture weights depend on covariates:

```python
with pm.Model() as moe:
    # Gating network: covariates determine mixture weights
    beta_gate = pm.Normal("beta_gate", 0, 1, shape=(X.shape[1], K - 1))
    logits = pm.math.dot(X, beta_gate)
    w = pm.math.softmax(
        pm.math.concatenate([logits, pm.math.zeros((len(X), 1))], axis=1),
        axis=1,
    )

    # Expert networks: each component has its own regression
    for k in range(K):
        beta_k = pm.Normal(f"beta_{k}", 0, 1, shape=X.shape[1])
        mu_k = pm.math.dot(X, beta_k)
    ...
```

## Practical Advice

1. **Start simple**: Model only the mean first. Add distributional components one at a time.
2. **Check residuals**: Plot residuals vs covariates. Patterns suggest which parameters need covariate dependence.
3. **Identifiability**: With many parameters varying, ensure you have enough data. Rule of thumb: at least 10 observations per parameter.
4. **Tight priors on scale/shape**: Use `Normal(0, 0.5)` or tighter on log-scale parameters. Wide priors on log scale allow extreme values.
5. **Different design matrices**: The covariates affecting the mean don't have to be the same as those affecting the variance. Use domain knowledge to decide.
6. **Prior predictive checks**: Even more important here — simulate from the prior and check that predictions are plausible across the covariate range.
