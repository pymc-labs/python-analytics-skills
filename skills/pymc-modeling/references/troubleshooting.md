# Troubleshooting Common PyMC Problems

This reference covers the most common model-building problems encountered by PyMC users, with precise diagnostics and solutions.

## Table of Contents
- [Quick Diagnostic Reference](#quick-diagnostic-reference)
- [Shape and Dimension Errors](#shape-and-dimension-errors)
- [Initialization Failures](#initialization-failures)
- [Mass Matrix and Numerical Issues](#mass-matrix-and-numerical-issues)
- [Divergences and Geometry Problems](#divergences-and-geometry-problems)
- [Non-Differentiable Operations](#non-differentiable-operations)
- [Discrete Variable Challenges](#discrete-variable-challenges)
- [Data Container and Prediction Issues](#data-container-and-prediction-issues)
- [Coordinate and Dimension Management](#coordinate-and-dimension-management)
- [Compositional Data Constraints](#compositional-data-constraints)
- [Prior-Related Pathologies](#prior-related-pathologies)

---

## Quick Diagnostic Reference

| Symptom | Primary Diagnostic | Expert Solution |
|---------|-------------------|-----------------|
| `ValueError: Shape mismatch` | Parameter vs observation alignment | Factorize indices; use coords/dims |
| `Initial evaluation failed: -inf` | Data outside distribution support | Check bounds; reduce initialization jitter |
| `Mass matrix contains zeros` | Unscaled predictors or flat energy | Standardize features; add weakly informative priors |
| High divergence count | Funnel geometry or hard boundaries | Non-centered reparameterization; soft clipping |
| Poor GP convergence | Inappropriate lengthscale prior | InverseGamma based on pairwise distance |
| `TypeError` in logic | Python `if/else` inside model | Use `pm.math.switch` or `pytensor.ifelse` |
| Slow discrete sampling | NUTS compatibility issues | Marginalize discrete variables or use compound steps |
| Inconsistent predictions | Shifting group labels | Use `sort=True` in `pd.factorize` or Categorical types |
| `NaN` in log-probability | Invalid parameter combinations | Check parameter constraints, add bounds |
| Prior predictive too extreme | Over-tight or too-loose priors | Prior predictive checking workflow |

---

## Shape and Dimension Errors

### The Parameter-Observation Dimension Mismatch

**Problem**: `ValueError` about shape mismatch during model initialization.

**Root cause**: Confusing the dimensions of parameter priors with the dimensions of the likelihood. In hierarchical models, parameters are defined at the group level (K groups) while the likelihood must evaluate at every observation (N data points).

**Example of the error**:
```python
# BAD: Likelihood sized to number of groups, not observations
with pm.Model() as bad_model:
    alpha = pm.Normal("alpha", 0, 1, shape=4)  # 4 groups
    y = pm.Normal("y", mu=alpha, sigma=1, observed=y_obs)  # y_obs has 100 points!
    # Error: alpha has shape (4,), y_obs has shape (100,)
```

**Solution**: Use index vectors to expand group parameters to observation size.

```python
import pandas as pd

# Create index mapping observations to groups
group_idx, group_labels = pd.factorize(df["group"], sort=True)

coords = {
    "group": group_labels,
    "obs": np.arange(len(df)),
}

with pm.Model(coords=coords) as correct_model:
    # Group-level parameter: shape (K,)
    alpha = pm.Normal("alpha", 0, 1, dims="group")
    sigma = pm.HalfNormal("sigma", 1)

    # Index into group parameters: alpha[group_idx] has shape (N,)
    mu = alpha[group_idx]

    # Likelihood: shape (N,)
    y = pm.Normal("y", mu=mu, sigma=sigma, observed=y_obs, dims="obs")
```

**Key insight**: The log-likelihood is a sum over individual data points. Every observation must have a corresponding mean value, so `alpha[group_idx]` expands the K group effects to N observations.

### Likelihood Size Independence from Parameters

**Common misconception**: New users often try to "shrink" the likelihood to match the number of groups.

**Reality**: The likelihood size is always determined by the number of observations, regardless of how many parameters the model has. A linear regression with 3 parameters and 100 observations has a likelihood of size 100.

```python
# The model evaluates (y_i - mu_group[i]) for every i in 1...N
# NOT once per group
```

---

## Initialization Failures

### "Initial evaluation of model at starting point failed"

**Problem**: Log-probability is `-inf` or `NaN` at the initial parameter values.

**Cause 1: Data outside distribution support**

```python
# BAD: Observations outside truncation bounds
with pm.Model() as bad_model:
    mu = pm.Normal("mu", 0, 1)
    # TruncatedNormal bounded below at 0
    y = pm.TruncatedNormal("y", mu=mu, sigma=1, lower=0, observed=y_obs)
    # If y_obs contains negative values, log-likelihood = -inf
```

**Solution**: Verify observed data matches the likelihood's support.

```python
# Check data before modeling
print(f"y range: [{y_obs.min()}, {y_obs.max()}]")

# For bounded distributions, ensure data is within bounds
assert y_obs.min() >= 0, "Data contains negative values!"
```

**Cause 2: Initialization jitter pushes parameters outside support**

PyMC's default `jitter+adapt_diag` initialization adds random noise to starting values. For constrained parameters (e.g., positive-only), this can create invalid starting points.

```python
# Solution 1: Reduce or eliminate jitter
idata = pm.sample(init="adapt_diag")  # no jitter

# Solution 2: Specify valid initial values
with model:
    idata = pm.sample(initvals={"sigma": 1.0})

# Solution 3: Use ADVI for more robust initialization
with model:
    idata = pm.sample(init="advi+adapt_diag")
```

**Cause 3: Constant response variable**

When the response variable has zero variance (all values identical), automatic prior selection in tools like Bambi may set scale parameters to zero.

```python
# Check for constant response
if y_obs.std() == 0:
    print("WARNING: Response variable is constant!")
```

### Debugging Initialization

```python
# Check which variables have invalid log-probabilities
model.point_logps()

# Run model diagnostics
model.debug()
```

---

## Mass Matrix and Numerical Issues

### "Mass matrix contains zeros on the diagonal"

**Problem**: HMC's mass matrix has zero diagonal elements, preventing exploration of those dimensions.

**Root cause**: The derivative of the log-probability with respect to some parameter is numerically zero, usually due to:

1. **Unscaled predictors**
2. **Underconstrained parameters** (flat likelihood regions)

### Unscaled Predictors

When predictors are on vastly different scales, gradients for large-scale variables become numerically indistinguishable from zero.

```python
# BAD: Features on different scales
X = np.column_stack([
    feature_0_to_1,      # range [0, 1]
    feature_0_to_million  # range [0, 1,000,000]
])
# Gradient w.r.t. large-scale feature may underflow
```

**Solution**: Standardize all predictors.

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Now all features have mean=0, std=1
# Priors can use unit-scale: beta ~ Normal(0, 1)
```

### Underconstrained Parameters (Flat Energy Landscape)

When data provides no information about a parameter (e.g., a group with no observations), the likelihood is flat and gradients are zero.

```python
# Check for empty groups
for group in group_labels:
    n_obs = (df["group"] == group).sum()
    if n_obs == 0:
        print(f"WARNING: Group '{group}' has no observations!")
```

**Solution**: Use weakly informative priors to provide "soft" curvature.

```python
# BAD: Flat or very diffuse priors
sigma = pm.Uniform("sigma", 0, 100)

# GOOD: Weakly informative prior provides curvature
sigma = pm.HalfNormal("sigma", sigma=1)
```

---

## Divergences and Geometry Problems

### Understanding Divergences

Divergences occur when the Hamiltonian integrator fails to conserve energy during a leapfrog step. This signals that the sampler encountered a region where the posterior geometry is too curved to navigate accurately.

### The Funnel Geometry

In hierarchical models, when group-level variance is small, individual effects are constrained to a narrow region. As variance increases, the region expands. This creates a "funnel" shape that is difficult for NUTS to navigate.

```python
# Centered parameterization creates funnel
# When sigma_group is small, alpha values are tightly constrained
alpha = pm.Normal("alpha", mu=mu_group, sigma=sigma_group, dims="group")
```

### Non-Centered Reparameterization

The standard fix transforms the funnel into an isotropic Gaussian.

```python
# BEFORE: Centered (funnel geometry)
alpha = pm.Normal("alpha", mu_group, sigma_group, dims="group")

# AFTER: Non-centered (spherical geometry)
z = pm.Normal("z", 0, 1, dims="group")  # standard normal
alpha = pm.Deterministic("alpha", mu_group + sigma_group * z, dims="group")
```

| Parameterization | Best For | Geometry |
|-----------------|----------|----------|
| Centered | Dense data per group | Direct hierarchy representation |
| Non-centered | Sparse/imbalanced data | Decouples effects from variance |

**When to use which**:
- Non-centered: Fewer than ~20 observations per group on average
- Centered: Many observations per group (likelihood "pins" parameters)

### Increasing Target Acceptance

For scattered divergences (not clustered in funnels), try increasing `target_accept`:

```python
# Default is 0.8; increase for difficult posteriors
idata = pm.sample(target_accept=0.95)

# For very difficult models
idata = pm.sample(target_accept=0.99)
```

**Trade-off**: Higher target acceptance means smaller step sizes and slower sampling.

---

## Non-Differentiable Operations

### The Clipping Problem

Using `pm.math.clip()` or similar hard constraints creates regions where gradients are exactly zero.

```python
# BAD: Hard clipping creates flat gradient regions
mu = pm.math.clip(x, 0, np.inf)
# Gradients are zero in clipped regions, causing divergences
```

**Solution 1: Use soft alternatives**

```python
# Softplus maps R -> R+ with smooth gradients
from pytensor.tensor.nnet import softplus

mu = softplus(x)  # log(1 + exp(x))
```

**Solution 2: Use distributions with natural constraints**

```python
# Instead of clipping to positive, use naturally positive distributions
sigma = pm.HalfNormal("sigma", 1)  # automatically positive
rate = pm.LogNormal("rate", 0, 1)  # automatically positive
```

### Python Conditionals in Models

Python `if/else` statements are evaluated at model construction time, not sampling time.

```python
# BAD: Python conditional doesn't work during sampling
if x > 0:  # evaluated once when model is built!
    result = a
else:
    result = b
```

**Solution**: Use PyTensor symbolic conditionals.

```python
import pytensor.tensor as pt

# GOOD: Symbolic conditional evaluated during sampling
result = pt.switch(x > 0, a, b)

# For more complex conditionals
from pytensor.ifelse import ifelse
result = ifelse(condition, true_value, false_value)
```

For iterative logic (loops that depend on random variables), use `pytensor.scan`.

---

## Discrete Variable Challenges

### NUTS Cannot Handle Discrete Latent Variables

NUTS is a gradient-based method and cannot differentiate through discrete nodes (Bernoulli, Poisson, Categorical with latent values).

**Symptom**: PyMC automatically reverts to compound sampling (NUTS for continuous, Metropolis for discrete), resulting in poor mixing.

### Solution 1: Marginalization

Analytically integrate out the discrete variable, replacing it with a continuous mixture.

```python
# BEFORE: Discrete latent variable (poor mixing)
with pm.Model() as discrete_model:
    z = pm.Bernoulli("z", p=0.5, shape=N)  # discrete latent
    mu = pm.math.switch(z, mu1, mu0)
    y = pm.Normal("y", mu=mu, sigma=1, observed=y_obs)

# AFTER: Marginalized mixture (NUTS-friendly)
with pm.Model() as marginalized_model:
    p = pm.Beta("p", 2, 2)
    mu0 = pm.Normal("mu0", 0, 1)
    mu1 = pm.Normal("mu1", 0, 1)
    sigma = pm.HalfNormal("sigma", 1)

    y = pm.NormalMixture("y", w=[1-p, p], mu=[mu0, mu1], sigma=sigma, observed=y_obs)
```

### Solution 2: Continuous Relaxation

Approximate discrete switches with continuous sigmoid functions.

```python
# Soft approximation of a discrete switch
temperature = 0.1  # lower = sharper
soft_z = pm.math.sigmoid((x - threshold) / temperature)
mu = soft_z * mu1 + (1 - soft_z) * mu0
```

### Prior Sampling with Discrete Variables

**Common mistake**: Using `pm.sample()` to sample priors from a model with discrete variables.

```python
# BAD: pm.sample() uses MCMC, struggles with discrete priors
with discrete_model:
    prior = pm.sample(draws=1000)  # slow, poor convergence

# GOOD: Use ancestral sampling for priors
with discrete_model:
    prior = pm.sample_prior_predictive(draws=1000)  # instant, exact
```

---

## Data Container and Prediction Issues

### Mutable Data for Out-of-Sample Prediction

Static NumPy arrays are baked into the model graph and cannot be changed for prediction.

```python
# BAD: Static data prevents out-of-sample prediction
with pm.Model() as static_model:
    X_train = np.array(...)  # fixed at model construction
    mu = pm.math.dot(X_train, beta)
    y = pm.Normal("y", mu=mu, sigma=1, observed=y_train)
```

**Solution**: Use `pm.Data` containers.

```python
with pm.Model() as mutable_model:
    X = pm.Data("X", X_train)
    y_obs = pm.Data("y_obs", y_train)

    beta = pm.Normal("beta", 0, 1, dims="features")
    sigma = pm.HalfNormal("sigma", 1)

    mu = pm.math.dot(X, beta)
    y = pm.Normal("y", mu=mu, sigma=sigma, observed=y_obs, shape=X.shape[0])

    idata = pm.sample()

# Predict on new data
with mutable_model:
    pm.set_data({"X": X_test, "y_obs": np.zeros(len(X_test))})
    ppc = pm.sample_posterior_predictive(idata)
```

### Shape Tracking for Variable-Size Predictions

**Critical**: The likelihood shape must track the data container shape.

```python
# BAD: Fixed shape doesn't update with new data
y = pm.Normal("y", mu=mu, sigma=sigma, observed=y_obs)  # shape inferred from y_obs at construction

# GOOD: Shape linked to data container
y = pm.Normal("y", mu=mu, sigma=sigma, observed=y_obs, shape=X.shape[0])
```

---

## Coordinate and Dimension Management

### Multi-Level Hierarchy Index Management

For nested hierarchies (e.g., observations within cities within states), avoid MultiIndex complications.

```python
# Create independent factorized indices for each level
city_idx, city_labels = pd.factorize(df["city"], sort=True)
state_idx, state_labels = pd.factorize(df["state"], sort=True)

coords = {
    "city": city_labels,
    "state": state_labels,
    "obs": np.arange(len(df)),
}

with pm.Model(coords=coords) as hierarchical:
    # State-level effects
    alpha_state = pm.Normal("alpha_state", 0, 1, dims="state")

    # City-level effects
    alpha_city = pm.Normal("alpha_city", 0, sigma_city, dims="city")

    # Combine at observation level
    mu = alpha_state[state_idx] + alpha_city[city_idx]

    y = pm.Normal("y", mu=mu, sigma=sigma, observed=y_obs, dims="obs")
```

### Consistent Factorization Across Data Splits

**Critical error**: Factorizing training and test sets separately creates inconsistent mappings.

```python
# BAD: Separate factorization
train_idx, train_labels = pd.factorize(train_df["group"])
test_idx, test_labels = pd.factorize(test_df["group"])
# "group 0" in train may be different from "group 0" in test!

# GOOD: Use categorical types for consistent mapping
df["group"] = pd.Categorical(df["group"])
train_df = df[train_mask]
test_df = df[~train_mask]

# Category codes are now consistent
train_idx = train_df["group"].cat.codes
test_idx = test_df["group"].cat.codes
```

---

## Compositional Data Constraints

### Multinomial Count Constraints

**Problem**: Observed counts don't sum to the specified `n` parameter.

```python
# Each row of observed counts must sum to n
observed_counts = np.array([[5, 3, 2], [4, 4, 2]])  # each row sums to 10
n_trials = observed_counts.sum(axis=1)  # [10, 10]

with pm.Model() as multinomial_model:
    p = pm.Dirichlet("p", a=np.ones(3))
    y = pm.Multinomial("y", n=n_trials, p=p, observed=observed_counts)
```

### Dirichlet Simplex Constraints

The probability vector `p` in Dirichlet-Multinomial models must sum to 1. Ensure any indexing operations preserve this constraint.

---

## Prior-Related Pathologies

### Log-Probability Overflows in Gamma/Beta Models

When modeling with Gamma or Beta likelihoods, parameters calculated through exponential transformations can overflow.

```python
# BAD: Exponential growth can overflow
mu_inv = pm.math.exp(lambda_param * pm.math.log(x))
# If lambda is large, mu_inv can exceed 10^20, causing overflow

# Solution 1: Use Softplus instead of Exp
mu_inv = pm.math.softplus(linear_predictor)

# Solution 2: Log-transform the data
log_y = np.log(y_obs)
y = pm.Normal("y", mu=mu_log, sigma=sigma, observed=log_y)
```

### Over-Tight Priors and Link Function Failure

Priors that are too narrow can "lock" the model, especially with link functions on new data.

```python
# If prior on slope is very tight: beta ~ Normal(0, 0.1)
# And new X values are much larger than training X
# Then sigmoid(X_new @ beta) can be exactly 0 or 1
# This causes log(0) = -inf in the likelihood
```

**Solution**: Use prior predictive checks across the expected range of predictors.

```python
with model:
    prior = pm.sample_prior_predictive(draws=500)

# Check that prior predictions are valid across expected X range
# Not just the training data range
```

### Label Switching in Mixture Models

Unordered mixture components cause label switching — the sampler swaps component identities between iterations, producing meaningless per-component summaries.

```python
# BAD: Unordered components
mu = pm.Normal("mu", 0, 10, dims="component")

# FIX: Order constraint
mu = pm.Normal("mu", 0, 10, dims="component",
               transform=pm.distributions.transforms.ordered,
               initval=np.linspace(y_obs.min(), y_obs.max(), K))
```

See [mixtures.md](mixtures.md) for detailed solutions.

### Missing Prior Predictive Checks

Always check prior implications before fitting:

```python
with model:
    prior_pred = pm.sample_prior_predictive()

az.plot_ppc(prior_pred, group="prior")
```

If prior predictive range is implausible (negative counts, probabilities > 1, extreme values), adjust priors before proceeding.

### The Horseshoe Prior Challenge

Horseshoe priors have a massive spike at zero and heavy tails, creating a "double-funnel" geometry.

```python
# Horseshoe often requires very high target_accept
idata = pm.sample(target_accept=0.99)

# Consider Regularized Horseshoe for better geometry (manual implementation)
# See priors.md for full regularized horseshoe code

# Or use simpler Laplace prior if sufficient
beta = pm.Laplace("beta", mu=0, b=1, dims="features")
```

### Redundant Intercepts in Hierarchical Models

**Problem**: Defining an intercept for every predictor creates non-identifiability.

```python
# BAD: Intercept per predictor (non-identifiable)
with pm.Model() as over_param:
    intercept_age = pm.Normal("int_age", 0, 1)
    intercept_income = pm.Normal("int_income", 0, 1)
    # These compete with each other and with group intercepts

# GOOD: Single intercept per group + slopes for predictors
with pm.Model() as correct:
    group_intercept = pm.Normal("group_intercept", mu_global, sigma_group, dims="group")
    slope_age = pm.Normal("slope_age", 0, 1)
    slope_income = pm.Normal("slope_income", 0, 1)

    mu = group_intercept[group_idx] + slope_age * age + slope_income * income
```

---

## PyMC API Issues

### Variable Name Same as Dimension Label

PyMC v5+ does not allow a variable to have the same name as its dimension label. This causes a `ValueError` at model creation.

```python
# ERROR: Variable `cohort` has the same name as its dimension label
coords = {"cohort": cohorts, "year": years}
with pm.Model(coords=coords) as model:
    cohort = rw2_fn("cohort", n_cohorts, sigma_c, dims="cohort")  # ValueError!

# FIX: Use different names for dimension labels
coords = {"cohort_idx": cohorts, "year_idx": years}
with pm.Model(coords=coords) as model:
    cohort = rw2_fn("cohort", n_cohorts, sigma_c, dims="cohort_idx")  # OK
    period = rw2_fn("period", n_years, sigma_t, dims="year_idx")  # OK
```

### ArviZ plot_ppc Parameter Names

ArviZ's `plot_ppc()` function does not accept `num_pp_samples` parameter. This parameter was removed in recent versions.

```python
# ERROR: Unexpected keyword argument
az.plot_ppc(idata, kind="cumulative", num_pp_samples=100)  # TypeError

# FIX: Remove num_pp_samples parameter
az.plot_ppc(idata, kind="cumulative")  # OK

# Subset to fewer draws if needed
idata_subset = idata.sel(draw=slice(0, 100))
az.plot_ppc(idata_subset, kind="cumulative")
```

### pm.MutableData / pm.ConstantData Deprecation

`pm.MutableData` and `pm.ConstantData` are deprecated in PyMC v5+. Use `pm.Data` instead, which is mutable by default.

```python
# DEPRECATED
x = pm.MutableData("x", x_obs)
c = pm.ConstantData("c", constants)

# CURRENT
x = pm.Data("x", x_obs)           # mutable by default
c = pm.Data("c", constants)       # use pm.Data for constants too
```

---

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

### Profiling Slow Models

```python
# Time individual operations in the log-probability computation
profile = model.profile(model.logp())
profile.summary()

# Identify bottlenecks in gradient computation
import pytensor
grad_profile = model.profile(pytensor.grad(model.logp(), model.continuous_value_vars))
grad_profile.summary()
```

---

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

---

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

---

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
# See priors.md for full code
```

**QR parameterization** (orthogonalizes predictors):

```python
Q, R = np.linalg.qr(X)
R_inv = np.linalg.inv(R)

with pm.Model() as model:
    theta = pm.Normal("theta", 0, 1, dims="features")
    beta = pm.Deterministic("beta", pt.dot(R_inv, theta))
    mu = pt.dot(Q, theta)
    y = pm.Normal("y", mu=mu, sigma=sigma, observed=y_obs)
```

**Interpret carefully**: If prediction is the goal, multicollinearity may not matter—just don't interpret individual coefficients.

---

## See Also

- [diagnostics.md](diagnostics.md) - Post-sampling diagnostic workflow
- [priors.md](priors.md) - Prior selection guidance
- [inference.md](inference.md) - Sampler selection and configuration
