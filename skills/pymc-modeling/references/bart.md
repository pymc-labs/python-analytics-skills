# BART (Bayesian Additive Regression Trees)

BART is a nonparametric regression approach using an ensemble of trees with a Bayesian prior. Available via `pymc-bart`.

## Table of Contents
- [Basic Usage](#basic-usage)
- [Regression](#regression)
- [Classification](#classification)
- [Variable Importance](#variable-importance)
- [Partial Dependence](#partial-dependence)
- [Configuration](#configuration)

## Basic Usage

```python
import pymc as pm
import pymc_bart as pmb

with pm.Model() as bart_model:
    # BART prior over the regression function
    mu = pmb.BART("mu", X=X, Y=y, m=50)

    # Observation noise
    sigma = pm.HalfNormal("sigma", sigma=1)

    # Likelihood
    y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y)

    idata = pm.sample()
```

## Regression

### Continuous Outcome

```python
with pm.Model() as regression_bart:
    mu = pmb.BART("mu", X=X_train, Y=y_train, m=50)
    sigma = pm.HalfNormal("sigma", 1)
    y = pm.Normal("y", mu=mu, sigma=sigma, observed=y_train)

    idata = pm.sample()

# Predictions
with regression_bart:
    pmb.set_data({"mu": X_test})
    ppc = pm.sample_posterior_predictive(idata)
```

### Heteroscedastic Regression

```python
with pm.Model() as hetero_bart:
    # Mean function
    mu = pmb.BART("mu", X=X, Y=y, m=50)

    # Variance function (also BART)
    log_sigma = pmb.BART("log_sigma", X=X, Y=y, m=20)
    sigma = pm.Deterministic("sigma", pm.math.exp(log_sigma))

    y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y)
```

## Classification

### Binary Classification

```python
with pm.Model() as binary_bart:
    # BART on latent scale
    mu = pmb.BART("mu", X=X, Y=y, m=50)

    # Probit or logit link
    p = pm.Deterministic("p", pm.math.sigmoid(mu))

    y_obs = pm.Bernoulli("y_obs", p=p, observed=y)

    idata = pm.sample()
```

### Multiclass Classification

```python
with pm.Model(coords={"class": classes}) as multiclass_bart:
    # Separate BART for each class (one-vs-rest style)
    mu = pmb.BART("mu", X=X, Y=y_onehot, m=50, dims="class")

    # Softmax
    p = pm.Deterministic("p", pm.math.softmax(mu, axis=-1))

    y_obs = pm.Categorical("y_obs", p=p, observed=y)
```

## Variable Importance

### Compute Variable Importance

```python
# After sampling
vi = pmb.compute_variable_importance(idata, X, method="VI")

# Plot
pmb.plot_variable_importance(vi, X)
```

### Methods

- `"VI"`: Based on inclusion frequency in trees
- `"backward"`: Backward elimination importance

```python
# Backward elimination (more expensive but often better)
vi_backward = pmb.compute_variable_importance(
    idata, X, method="backward", random_seed=42
)
```

## Partial Dependence

### 1D Partial Dependence

```python
# Partial dependence for variable at index 0
pmb.plot_pdp(idata, X=X, Y=y, xs_interval="quantiles", var_idx=[0])

# Multiple variables
pmb.plot_pdp(idata, X=X, Y=y, var_idx=[0, 1, 2])
```

### 2D Partial Dependence (Interaction)

```python
# Interaction between variables 0 and 1
pmb.plot_pdp(idata, X=X, Y=y, var_idx=[0, 1], grid="wide")
```

### Individual Conditional Expectation (ICE)

```python
pmb.plot_ice(idata, X=X, Y=y, var_idx=0)
```

## Configuration

### Key Parameters

```python
mu = pmb.BART(
    "mu",
    X=X,
    Y=y,
    m=50,              # number of trees (default 50, more = smoother)
    alpha=0.95,        # prior probability tree has depth 1
    beta=2.0,          # controls depth of trees
    split_prior=None,  # prior on split variable selection
)
```

### Number of Trees (m)

- `m=50`: Good default
- `m=100-200`: Smoother fit, more computation
- `m=20-30`: Faster, may underfit

```python
# More trees for complex functions
mu = pmb.BART("mu", X=X, Y=y, m=100)
```

### Tree Depth (alpha, beta)

Controls tree complexity via prior P(node is terminal at depth d) = alpha * (1 + d)^(-beta)

- Higher `alpha` or lower `beta`: Deeper trees
- Default `alpha=0.95, beta=2` works well

### Split Prior

Control which variables are preferred for splitting:

```python
# Uniform (default)
split_prior = None

# Favor first 3 variables
split_prior = [2, 2, 2, 1, 1, 1, 1]  # length = n_features

mu = pmb.BART("mu", X=X, Y=y, split_prior=split_prior)
```

## Combining BART with Parametric Components

### BART + Linear

```python
with pm.Model() as semi_parametric:
    # Linear component for known effects
    beta = pm.Normal("beta", 0, 1, shape=p_linear)
    linear = pm.math.dot(X_linear, beta)

    # BART for nonlinear/interaction effects
    nonlinear = pmb.BART("nonlinear", X=X_nonlinear, Y=y, m=50)

    mu = linear + nonlinear
    sigma = pm.HalfNormal("sigma", 1)
    y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y)
```

### BART + Random Effects

```python
with pm.Model(coords={"group": groups}) as bart_mixed:
    # Group random effects
    sigma_group = pm.HalfNormal("sigma_group", 1)
    alpha = pm.Normal("alpha", 0, sigma_group, dims="group")

    # BART for fixed effects
    mu_bart = pmb.BART("mu_bart", X=X, Y=y, m=50)

    mu = mu_bart + alpha[group_idx]
    sigma = pm.HalfNormal("sigma", 1)
    y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y)
```

## Out-of-Sample Prediction

```python
# Fit model
with bart_model:
    idata = pm.sample()

# Predict on new data
with bart_model:
    pmb.set_data({"mu": X_new})
    ppc = pm.sample_posterior_predictive(idata, var_names=["y_obs"])

# Extract predictions
y_pred = ppc.posterior_predictive["y_obs"]
```

## Convergence Diagnostics

BART uses a particle Gibbs sampler, so standard MCMC diagnostics apply:

```python
import arviz as az

az.plot_trace(idata, var_names=["sigma"])
az.summary(idata, var_names=["sigma"])

# For BART predictions, check posterior predictive
az.plot_ppc(idata)
```
