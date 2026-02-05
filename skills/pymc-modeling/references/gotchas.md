# Common Pitfalls

## Structural Issues

### Using Python Conditionals Inside Models

Python `if/else` statements are evaluated at model construction time, not during sampling.

```python
# BAD: Python conditional evaluated once at construction
if x > threshold:  # This is NOT evaluated per sample!
    mu = a
else:
    mu = b

# GOOD: Use PyTensor symbolic operations
import pytensor.tensor as pt

mu = pt.switch(x > threshold, a, b)

# For complex conditionals
from pytensor.ifelse import ifelse
result = ifelse(condition, true_branch, false_branch)
```

For iterative logic depending on random variables, use `pytensor.scan`.

### Hard Clipping Creates Non-Differentiable Regions

Clipping functions create flat gradient regions where NUTS cannot navigate.

```python
# BAD: Hard clipping causes divergences
mu = pm.math.clip(linear_pred, 0, np.inf)
# Gradient is zero in clipped regions

# GOOD: Soft alternatives
from pytensor.tensor.nnet import softplus
mu = softplus(linear_pred)  # smooth, positive

# Or use naturally constrained distributions
sigma = pm.HalfNormal("sigma", 1)  # always positive
rate = pm.LogNormal("rate", 0, 1)  # always positive
```

### Inconsistent Group Index Factorization

Separate factorization of train/test sets creates inconsistent mappings.

```python
# BAD: Independent factorization
train_idx, _ = pd.factorize(train_df["group"])
test_idx, _ = pd.factorize(test_df["group"])
# "group 0" may differ between train and test!

# GOOD: Use categorical types
df["group"] = pd.Categorical(df["group"])
train_idx = train_df["group"].cat.codes
test_idx = test_df["group"].cat.codes

# Or factorize once on all data
all_idx, labels = pd.factorize(full_df["group"], sort=True)
train_idx = all_idx[train_mask]
test_idx = all_idx[~train_mask]
```

### Dimension Mismatch: Parameters vs Observations

In hierarchical models, parameters have group-level dimensions while likelihoods need observation-level dimensions.

```python
# BAD: Alpha has K groups, y_obs has N observations
alpha = pm.Normal("alpha", 0, 1, dims="group")
y = pm.Normal("y", mu=alpha, sigma=1, observed=y_obs)  # Shape error!

# GOOD: Index into group parameters
group_idx = df["group"].cat.codes
y = pm.Normal("y", mu=alpha[group_idx], sigma=1, observed=y_obs)
```

## Statistical Issues

### Centered Parameterization with Weak Data

```python
# Causes divergences with few observations per group
alpha = pm.Normal("alpha", mu_alpha, sigma_alpha, dims="group")  # BAD

# Non-centered parameterization
alpha_offset = pm.Normal("alpha_offset", 0, 1, dims="group")
alpha = pm.Deterministic("alpha", mu_alpha + sigma_alpha * alpha_offset, dims="group")
```

### Flat Priors on Scale Parameters

```python
# Problematic in hierarchical models
sigma = pm.Uniform("sigma", 0, 100)  # BAD
sigma = pm.HalfFlat("sigma")  # BAD

# Weakly informative alternatives
sigma = pm.HalfNormal("sigma", sigma=1)
sigma = pm.HalfCauchy("sigma", beta=1)
sigma = pm.Exponential("sigma", lam=1)
```

### Label Switching in Mixture Models

```python
# Unordered components cause label switching
mu = pm.Normal("mu", 0, 10, dims="component")  # BAD

# Order constraint
mu_raw = pm.Normal("mu_raw", 0, 10, dims="component")
mu = pm.Deterministic("mu", pt.sort(mu_raw), dims="component")
```

### Redundant Intercepts in Hierarchical Models

Defining individual intercepts for every predictor creates non-identifiability.

```python
# BAD: Separate intercept per predictor (non-identifiable)
intercept_age = pm.Normal("int_age", 0, 1)
intercept_income = pm.Normal("int_income", 0, 1)
# These compete with each other and any group intercepts

# GOOD: Single intercept structure
group_intercept = pm.Normal("group_int", mu_global, sigma_group, dims="group")
slope_age = pm.Normal("slope_age", 0, 1)
slope_income = pm.Normal("slope_income", 0, 1)

mu = group_intercept[group_idx] + slope_age * age + slope_income * income
```

| Error | Impact | Solution |
|-------|--------|----------|
| Intercept per predictor | Non-identifiability | Single intercept per group |
| No global intercept | Poor pooling | Include global hyperpriors |
| Unshared variances | Increased complexity | Use LKJ priors for correlations |

### Horseshoe Prior Geometry

The Horseshoe prior has a massive spike at zero and heavy tails, creating a "double-funnel" geometry that is extremely difficult for NUTS.

```python
# Horseshoe often requires very high target_accept
idata = pm.sample(target_accept=0.99)

# Consider Regularized Horseshoe for better geometry (manual implementation)
# See priors.md for full regularized horseshoe code

# Or simpler Laplace prior if sufficient
beta = pm.Laplace("beta", mu=0, b=1, dims="features")
```

### Missing Prior Predictive Checks

Always check prior implications before fitting:

```python
with model:
    prior_pred = pm.sample_prior_predictive()

az.plot_ppc(prior_pred, group="prior")
```

### Over-Tight Priors with Link Functions

Priors that are too narrow can cause link functions to saturate on new data.

```python
# If slope prior is very tight and test X values are larger than training:
# sigmoid(X_test @ beta) can be exactly 0 or 1
# This causes log(0) = -inf in the log-likelihood

# Solution: Prior predictive checks across expected predictor range
X_range = np.linspace(expected_min, expected_max, 100)
# Verify prior predictions remain valid across this range
```

## Performance Issues

### Full GP on Large Datasets

```python
# O(n³) - slow for n > 1000
gp = pm.gp.Marginal(cov_func=cov)
y = gp.marginal_likelihood("y", X=X_large, y=y_obs)

# O(nm) - use HSGP instead
gp = pm.gp.HSGP(m=[30], c=1.5, cov_func=cov)
f = gp.prior("f", X=X_large)
```

### Saving Large Deterministics

```python
# Stores n_obs x n_draws array
mu = pm.Deterministic("mu", X @ beta, dims="obs")  # SLOW

# Don't save intermediate computations
mu = X @ beta  # Not saved, use posterior_predictive if needed
```

### Recompiling for Each Dataset

```python
# Recompiles every iteration
for dataset in datasets:
    with pm.Model() as model:
        # ...
        idata = pm.sample()

# Use pm.Data to avoid recompilation
with pm.Model() as model:
    x = pm.Data("x", x_initial)
    # ...

for dataset in datasets:
    pm.set_data({"x": dataset["x"]})
    idata = pm.sample()
```

## Identifiability Issues

### Symptoms

- Strong parameter correlations in pair plots
- Very wide posteriors despite lots of data
- Different chains converging to different solutions
- R-hat > 1.01 despite long chains

### Common Causes

**Overparameterized models**: More parameters than the data can support.

```python
# Too many group-level effects for small groups
alpha_group = pm.Normal("alpha_group", 0, 1, dims="group")  # 100 groups, 3 obs each
beta_group = pm.Normal("beta_group", 0, 1, dims="group")    # Can't estimate both
```

**Multicollinearity**: Correlated predictors make individual effects unidentifiable.

**Redundant random effects**: Nested effects without constraints.

### Fixes

**Sum-to-zero constraints** for categorical effects:

```python
import pytensor.tensor as pt

# Constrain group effects to sum to zero
alpha_raw = pm.Normal("alpha_raw", 0, 1, shape=n_groups - 1)
alpha = pm.Deterministic("alpha", pt.concatenate([alpha_raw, -alpha_raw.sum(keepdims=True)]))
```

**QR decomposition** for regression with correlated predictors:

```python
# Orthogonalize design matrix
Q, R = np.linalg.qr(X)

with pm.Model() as qr_model:
    beta_tilde = pm.Normal("beta_tilde", 0, 1, dims="features")
    beta = pm.Deterministic("beta", pt.linalg.solve(R, beta_tilde))
    mu = Q @ beta_tilde  # Use Q directly in likelihood
```

**Reduce model complexity**: Start simple, add complexity only if needed.

### Diagnosis

```python
# Check for strong correlations
az.plot_pair(idata, var_names=["alpha", "beta"], divergences=True)

# Look for banana-shaped or ridge-like posteriors
# These indicate non-identifiability
```

## Prior-Data Conflict

### Symptoms

- Posterior piled against prior boundary
- Prior and posterior distributions look very different
- Divergences concentrated near prior boundaries
- Effective sample size very low for some parameters

### Diagnosis

```python
# Compare prior and posterior
az.plot_dist_comparison(idata, var_names=["sigma"])

# Visual comparison for all parameters
fig, axes = plt.subplots(1, len(param_names), figsize=(4*len(param_names), 3))
for ax, var in zip(axes, param_names):
    az.plot_density(idata.prior, var_names=[var], ax=ax, colors="C0", label="Prior")
    az.plot_density(idata.posterior, var_names=[var], ax=ax, colors="C1", label="Posterior")
    ax.set_title(var)
```

### Common Scenarios

**Prior too narrow**: Data suggests values outside prior range.

```python
# Prior rules out likely values
sigma = pm.HalfNormal("sigma", sigma=0.1)  # If true sigma is ~5, this fights the data

# Fix: Use domain knowledge, not convenience
sigma = pm.HalfNormal("sigma", sigma=5)  # Allow for larger values
```

**Prior on wrong scale**: Common when using default priors without checking.

```python
# Default prior on standardized scale
beta = pm.Normal("beta", 0, 1)  # Fine if X is standardized

# But if X ranges from 10000 to 50000...
# Standardize predictors or adjust prior
X_scaled = (X - X.mean()) / X.std()
```

### Resolution

1. Check data for errors (outliers, coding mistakes)
2. Reconsider prior based on domain knowledge
3. Use prior predictive checks to validate
4. If justified, use more flexible prior

```python
# Validate prior choice
with model:
    prior_pred = pm.sample_prior_predictive()

# Check if prior predictive includes observed data range
prior_y = prior_pred.prior_predictive["y"].values.flatten()
print(f"Prior range: [{prior_y.min():.1f}, {prior_y.max():.1f}]")
print(f"Data range: [{y_obs.min():.1f}, {y_obs.max():.1f}]")
```

## Multicollinearity

### The Problem

Correlated predictors make individual coefficient estimates unstable, even though predictions remain valid.

### Detection

```python
import numpy as np

# Condition number (>30 suggests problems)
condition_number = np.linalg.cond(X)
print(f"Condition number: {condition_number:.1f}")

# Correlation matrix
import pandas as pd
corr = pd.DataFrame(X, columns=feature_names).corr()
print(corr)

# Variance inflation factors (VIF)
from statsmodels.stats.outliers_influence import variance_inflation_factor

vif_data = pd.DataFrame()
vif_data["feature"] = feature_names
vif_data["VIF"] = [variance_inflation_factor(X, i) for i in range(X.shape[1])]
print(vif_data)  # VIF > 5-10 indicates multicollinearity
```

### Symptoms in Posteriors

```python
# Strong negative correlation between coefficients
az.plot_pair(idata, var_names=["beta"])
# Look for elongated ellipses or banana shapes

# Wide credible intervals despite large N
summary = az.summary(idata, var_names=["beta"])
print(summary[["mean", "sd", "hdi_3%", "hdi_97%"]])
```

### Solutions

**Drop redundant predictors**:

```python
# If age and birth_year are both included, drop one
X = X[:, [i for i, name in enumerate(feature_names) if name != "birth_year"]]
```

**Use regularizing priors**:

```python
# Ridge-like prior (shrinks toward zero)
beta = pm.Normal("beta", mu=0, sigma=0.5, dims="features")

# Horseshoe prior (sparse, some coefficients near zero)
# Must be implemented manually - see priors.md for full code
tau = pm.HalfCauchy("tau", beta=1)
lam = pm.HalfCauchy("lam", beta=1, dims="features")
beta = pm.Normal("beta", mu=0, sigma=tau * lam, dims="features")
```

**QR parameterization** (orthogonalizes predictors):

```python
Q, R = np.linalg.qr(X)
R_inv = np.linalg.inv(R)

with pm.Model() as model:
    # Sample in orthogonal space
    theta = pm.Normal("theta", 0, 1, dims="features")

    # Transform back to original scale
    beta = pm.Deterministic("beta", pt.dot(R_inv, theta))

    # Likelihood uses Q (orthogonal)
    mu = pt.dot(Q, theta)
    y = pm.Normal("y", mu=mu, sigma=sigma, observed=y_obs)
```

**Interpret carefully**: If prediction is the goal, multicollinearity may not matter—just don't interpret individual coefficients.

---

## See Also

- [troubleshooting.md](troubleshooting.md) - Comprehensive problem-solution guide
- [diagnostics.md](diagnostics.md) - Post-sampling diagnostic workflow
- [priors.md](priors.md) - Prior selection guidance
- [inference.md](inference.md) - Sampler selection and configuration
