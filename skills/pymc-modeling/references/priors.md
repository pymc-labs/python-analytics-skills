# Prior Selection Guide

## Table of Contents
- [The Foundational Role of Priors](#the-foundational-role-of-priors)
- [Hierarchy of Prior Informativeness](#hierarchy-of-prior-informativeness)
- [Weakly Informative Defaults](#weakly-informative-defaults)
- [Regression and GLM Priors](#regression-and-glm-priors)
- [High-Dimensional and Sparse Priors](#high-dimensional-and-sparse-priors)
- [Hierarchical Model Priors](#hierarchical-model-priors)
- [Prior Predictive Checking](#prior-predictive-checking)
- [Sensitivity Analysis](#sensitivity-analysis)
- [Expert Elicitation](#expert-elicitation)
- [Practical Implementation Guidelines](#practical-implementation-guidelines)
- [Domain-Specific Guidance](#domain-specific-guidance)

## The Foundational Role of Priors

In Bayesian inference, the prior distribution encodes pre-experimental knowledge, structural assumptions, and epistemic uncertainty. It acts as a weight function penalizing parameter regions inconsistent with theoretical constraints.

**Key insight**: While the prior's influence typically diminishes as data increases, it remains critical in:
- High-dimensional settings (p > n)
- Small-sample scenarios
- Models with weak identification

> **Analogy**: Choosing a prior is like selecting foundations for a building on unknown soil. A *flat prior* is building without foundations—risky unless ground is solid. A *weakly informative prior* provides sturdy pilings that prevent collapse while adjusting as the structure settles. An *informative prior* builds on existing foundations—faster if correct, catastrophic if flawed.

## Hierarchy of Prior Informativeness

| Prior Level | Intended Use | Recommendation |
|-------------|-------------|----------------|
| **Flat** | "Objective" estimation; no preference | **Discouraged**. Can lead to improper posteriors and lacks regularization. |
| **Super-vague** | Minimal influence (e.g., N(0, 10⁶)) | **Not recommended**. Causes numerical instability and funnel geometries. |
| **Weakly Informative** | Regularization; ruling out absurdities | **Preferred default**. Provides stability while letting data dominate. |
| **Specifically Informative** | Expert belief or historical data | **Use with caution**. Requires formal elicitation or meta-analysis. |

```python
# Examples of the hierarchy
# AVOID: Flat/super-vague
beta_bad = pm.Flat("beta")                    # improper, avoid
beta_vague = pm.Normal("beta", mu=0, sigma=1e6)  # numerical issues

# PREFERRED: Weakly informative
beta = pm.Normal("beta", mu=0, sigma=2.5)     # regularizing, stable

# USE CAREFULLY: Informative (requires justification)
beta_informed = pm.Normal("beta", mu=0.3, sigma=0.1)  # from prior study
```

## Weakly Informative Defaults

### Location Parameters (means, intercepts, coefficients)

```python
# Standardize predictors first, then use unit-scale priors
beta = pm.Normal("beta", mu=0, sigma=1)       # coefficients on standardized scale
intercept = pm.Normal("intercept", mu=0, sigma=2.5)  # wider for intercepts

# Student-t for robustness to outlier effects (nu between 3-7)
beta = pm.StudentT("beta", nu=4, mu=0, sigma=2.5)
```

### Scale Parameters (standard deviations)

**Modern practice favors half-Normal, half-t, or Exponential priors.**

```python
# Half-Normal (recommended default)
sigma = pm.HalfNormal("sigma", sigma=1)

# Half-Student-t (robust, good for hierarchical variance components)
sigma = pm.HalfStudentT("sigma", nu=4, sigma=1)

# Exponential (stronger regularization toward 0)
sigma = pm.Exponential("sigma", lam=1)
```

**Avoid InverseGamma for variances**: Despite historical popularity, InverseGamma can be surprisingly informative near zero, pushing posteriors away from zero even when groups are identical.

```python
# AVOID in hierarchical models:
variance = pm.InverseGamma("variance", alpha=0.01, beta=0.01)  # problematic

# PREFER:
sigma = pm.HalfNormal("sigma", sigma=1)
```

### Correlation Matrices (LKJ Prior)

The LKJ prior is standard for correlation matrices:
- **η = 1**: Uniform over all valid correlation matrices
- **η > 1**: Concentrates toward identity matrix (lower correlations expected)
- **η < 1**: Favors extreme correlations

```python
# LKJ prior on correlation matrix
chol, corr, stds = pm.LKJCholeskyCov(
    "chol", n=n_dims, eta=2.0,  # mild shrinkage toward identity
    sd_dist=pm.HalfNormal.dist(sigma=1)
)

# Use in Multivariate Normal
vals = pm.MvNormal("vals", mu=0, chol=chol, observed=data)
```

### Probability Parameters

```python
# Beta prior for bounded [0, 1]
p = pm.Beta("p", alpha=1, beta=1)           # uniform
p = pm.Beta("p", alpha=2, beta=2)           # slight mode at 0.5

# Logit-normal for softer constraints
logit_p = pm.Normal("logit_p", mu=0, sigma=1.5)
p = pm.Deterministic("p", pm.math.sigmoid(logit_p))
```

### Count/Rate Parameters

```python
# Gamma for rates (positive, right-skewed)
rate = pm.Gamma("rate", alpha=2, beta=1)

# Log-normal when multiplicative effects expected
rate = pm.LogNormal("rate", mu=0, sigma=1)
```

## Regression and GLM Priors

### Coefficients

For standardized predictors, default to N(0, 1) or N(0, 2.5):

```python
# Linear regression (standardized predictors)
beta = pm.Normal("beta", mu=0, sigma=1, dims="predictors")

# Logistic regression (wider to allow strong effects)
beta = pm.Normal("beta", mu=0, sigma=2.5, dims="predictors")

# Robust to outlier predictors (Student-t, 3 < nu < 7)
beta = pm.StudentT("beta", nu=4, mu=0, sigma=2.5, dims="predictors")
```

### Intercepts

```python
# If outcome standardized
intercept = pm.Normal("intercept", mu=0, sigma=1)

# If outcome on original scale, center on domain-reasonable value
# Example: human height in cm
intercept = pm.Normal("intercept", mu=170, sigma=20)
```

## High-Dimensional and Sparse Priors

When p > n, regularization is essential for the bias-variance trade-off.

### Laplace (Lasso) Prior

Continuous shrinkage with tall mode at zero and thick tails. Efficient but shrinks even large coefficients.

```python
# Manual Laplace prior
beta = pm.Laplace("beta", mu=0, b=1, dims="features")
```

### Horseshoe Prior

Global-local shrinkage: global hyperparameter shrinks all coefficients while local half-Cauchy allows large signals to escape.

```python
# Manual horseshoe (no built-in pmx.Horseshoe exists)
tau = pm.HalfCauchy("tau", beta=1)  # global shrinkage
lam = pm.HalfCauchy("lam", beta=1, dims="features")  # local shrinkage
beta = pm.Normal("beta", mu=0, sigma=tau * lam, dims="features")
```

**Sampling challenges**: The Horseshoe creates a "double-funnel" geometry (massive spike at zero + heavy tails) that is extremely difficult for NUTS. Divergences are common unless using very high `target_accept` (0.99+). Consider:

1. Using the Regularized Horseshoe parameterization (see below)
2. Using simpler Laplace prior if full sparsity isn't required
3. Increasing `target_accept` to 0.99 and allowing longer sampling

### Regularized (Finnish) Horseshoe

Adds a finite "slab" width to prevent infinite estimates in cases of data separation (e.g., logistic regression). Must be implemented manually:

```python
import pytensor.tensor as pt

# Regularized horseshoe (manual implementation)
# D0 = expected number of non-zero coefficients, D = total features, N = observations
tau = pm.HalfStudentT("tau", nu=2, sigma=D0 / (D - D0) * sigma / np.sqrt(N))
lam = pm.HalfStudentT("lam", nu=5, dims="features")
c2 = pm.InverseGamma("c2", alpha=1, beta=1)  # slab variance
z = pm.Normal("z", 0, 1, dims="features")

# Regularized shrinkage factor
lam_tilde = pt.sqrt(c2 / (c2 + tau**2 * lam**2))
beta = pm.Deterministic("beta", z * tau * lam * lam_tilde, dims="features")
```

### R2D2 Prior

Induces a prior directly on R² (variance explained), often more interpretable. Available in pymc-extras.

```python
import pymc_extras as pmx

# X must be centered; returns (residual_sigma, coefficients)
# output_sigma and input_sigma are required
output_sigma = y.std()
input_sigma = X.std(axis=0)

residual_sigma, beta = pmx.R2D2M2CP(
    "r2d2",
    output_sigma=output_sigma,
    input_sigma=input_sigma,
    dims="features",
    r2=0.5,        # prior mean R²
    r2_std=0.2,    # uncertainty in R²
)
```

### Spike-and-Slab

"Gold standard" for variable selection using discrete mixture of point mass at zero and diffuse slab. Computationally demanding.

```python
# Spike-and-slab (requires careful implementation)
import pytensor.tensor as pt

inclusion = pm.Bernoulli("inclusion", p=0.5, dims="features")
beta_slab = pm.Normal("beta_slab", mu=0, sigma=2, dims="features")
beta = pm.Deterministic("beta", inclusion * beta_slab, dims="features")
```

## Hierarchical Model Priors

Hierarchical models use "partial pooling" where group estimates are informed by both their own data and the population distribution.

### Variance Components

```python
# Half-Student-t with nu=3-4 is robust default
# Prevents variances from collapsing to zero while remaining broad
sigma_group = pm.HalfStudentT("sigma_group", nu=4, sigma=1)

# For moderate pooling
sigma_group = pm.HalfNormal("sigma_group", sigma=0.5)

# AVOID: uniform on large range
sigma_group = pm.Uniform("sigma_group", 0, 100)  # problematic
```

### Small Number of Groups

With few groups (< 5-10), broader Cauchy priors can cause underpooling. Use tighter priors:

```python
# Few groups: more informative prior prevents underpooling
sigma_group = pm.HalfNormal("sigma_group", sigma=0.5)

# Many groups: can use broader priors
sigma_group = pm.HalfCauchy("sigma_group", beta=1)
```

### Non-Centered vs Centered Parameterization

When group-level variance is small relative to data, use non-centered:

```python
# Non-centered (preferred when data per group is sparse)
sigma_group = pm.HalfNormal("sigma_group", sigma=1)
z = pm.Normal("z", mu=0, sigma=1, dims="groups")
group_effect = pm.Deterministic("group_effect", z * sigma_group, dims="groups")

# Centered (can work when lots of data per group)
sigma_group = pm.HalfNormal("sigma_group", sigma=1)
group_effect = pm.Normal("group_effect", mu=0, sigma=sigma_group, dims="groups")
```

## Prior Predictive Checking

**Always simulate from priors before fitting** to validate that priors generate plausible data.

```python
with model:
    prior_pred = pm.sample_prior_predictive(draws=500)

# Visualize prior predictive distribution
import arviz as az
az.plot_ppc(prior_pred, group="prior", kind="cumulative")

# Numerical summary
prior_y = prior_pred.prior_predictive["y"].values.flatten()
print(f"Prior predictive range: [{prior_y.min():.2f}, {prior_y.max():.2f}]")
print(f"Prior predictive mean: {prior_y.mean():.2f}")
print(f"Prior predictive std: {prior_y.std():.2f}")
```

### Warning Signs

| Problem | Symptom | Solution |
|---------|---------|----------|
| **Implausible values** | Negative counts, >100% probabilities, heights > 10m | Tighten priors or add constraints |
| **Too concentrated** | Prior predictive doesn't cover observed data range | Widen priors |
| **Extremely wide** | Prior generates absurd values (humans 100m tall) | Use domain knowledge to constrain |
| **Miscentered** | Prior predictive systematically biased | Adjust location parameters |

```python
# Example: checking for implausible values
prior_y = prior_pred.prior_predictive["y"].values.flatten()

# For count data - should be non-negative
if (prior_y < 0).any():
    print("WARNING: Prior allows negative counts!")

# For proportions - should be in [0, 1]
if (prior_y < 0).any() or (prior_y > 1).any():
    print("WARNING: Prior allows invalid proportions!")

# For physical measurements - check against domain knowledge
if prior_y.max() > 300:  # e.g., human height in cm
    print("WARNING: Prior allows implausible heights!")
```

## Sensitivity Analysis

Evaluates posterior stability under prior perturbations. Critical in small datasets where prior influence is strongest.

### Basic Sensitivity Check

```python
def fit_with_prior(prior_sd):
    """Fit model with different prior scales."""
    with pm.Model() as m:
        beta = pm.Normal("beta", mu=0, sigma=prior_sd)
        sigma = pm.HalfNormal("sigma", sigma=1)
        y = pm.Normal("y", mu=beta * X, sigma=sigma, observed=y_obs)
        trace = pm.sample()
    return trace

# Compare posteriors under different priors
traces = {}
for sd in [0.5, 1.0, 2.0, 5.0]:
    traces[sd] = fit_with_prior(sd)

# Check if posteriors are similar
for sd, trace in traces.items():
    print(f"Prior SD={sd}: posterior mean={trace.posterior['beta'].mean():.3f}")
```

### Prior-Likelihood Conflict

Occurs when data substantially contradicts the prior. Can detect using prior-posterior comparison:

```python
# Compare prior and posterior
az.plot_dist_comparison(trace, var_names=["beta"])

# Large divergence suggests prior-likelihood conflict
prior_mean = 0  # from prior specification
posterior_mean = trace.posterior["beta"].mean().item()
posterior_sd = trace.posterior["beta"].std().item()

# Check if prior is many SDs from posterior
if abs(prior_mean - posterior_mean) > 3 * posterior_sd:
    print("WARNING: Potential prior-likelihood conflict")
```

## Expert Elicitation

When data is scarce (rare diseases, novel phenomena), priors must be constructed through formal elicitation.

### SHELF Framework

The Sheffield Elicitation Framework (SHELF) guides experts to translate beliefs into distributions:

1. **Define the quantity clearly** (what exactly are we eliciting?)
2. **Establish plausible range** (minimum, maximum)
3. **Elicit central tendency** (median or mode)
4. **Elicit uncertainty** (quartiles or probability intervals)
5. **Fit distribution** to elicited quantities

### Roulette Method

Often more intuitive than direct percentile elicitation:

```python
# Example: expert places "chips" into bins
bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
chips = [0, 1, 3, 5, 8, 10, 8, 5, 3, 1, 0]  # expert allocation

# Normalize to probability and fit Beta
import numpy as np
from scipy import stats

probs = np.array(chips) / sum(chips)
bin_centers = (np.array(bins[:-1]) + np.array(bins[1:])) / 2
mean = np.sum(probs * bin_centers)
var = np.sum(probs * (bin_centers - mean)**2)

# Fit Beta distribution
alpha = mean * (mean * (1 - mean) / var - 1)
beta_param = (1 - mean) * (mean * (1 - mean) / var - 1)
print(f"Elicited prior: Beta({alpha:.2f}, {beta_param:.2f})")
```

### Aggregating Multiple Experts

```python
# Simple mathematical pooling (weighted average)
expert_params = [
    (2, 5),   # Expert 1: Beta(2, 5)
    (3, 4),   # Expert 2: Beta(3, 4)
    (2, 3),   # Expert 3: Beta(2, 3)
]
weights = [0.4, 0.3, 0.3]  # based on expertise

# Linear pool (mixture of distributions)
# In PyMC, can use Mixture distribution
alphas = [p[0] for p in expert_params]
betas = [p[1] for p in expert_params]

p = pm.Mixture(
    "p",
    w=weights,
    comp_dists=[pm.Beta.dist(alpha=a, beta=b) for a, b in zip(alphas, betas)]
)
```

## Practical Implementation Guidelines

### 1. Scale Your Data

Rescale predictors and outcomes to unit scale (order of magnitude ~1):

```python
# Standardize predictors
X_scaled = (X - X.mean(axis=0)) / X.std(axis=0)

# Standardize outcome (optional but helpful)
y_scaled = (y - y.mean()) / y.std()

# Now use simple priors
with pm.Model() as model:
    beta = pm.Normal("beta", mu=0, sigma=1, dims="predictors")
    intercept = pm.Normal("intercept", mu=0, sigma=1)
    sigma = pm.HalfNormal("sigma", sigma=1)

    mu = intercept + pm.math.dot(X_scaled, beta)
    y_obs = pm.Normal("y", mu=mu, sigma=sigma, observed=y_scaled)
```

### 2. Use Coords and Dims

Makes prior specification clearer and InferenceData more interpretable:

```python
coords = {
    "predictors": ["age", "income", "education"],
    "obs_id": range(len(y))
}

with pm.Model(coords=coords) as model:
    beta = pm.Normal("beta", mu=0, sigma=1, dims="predictors")
```

### 3. Document Your Prior Choices

```python
# Good practice: document prior justification
with pm.Model() as model:
    # Prior: N(0, 2.5) allows coefficients up to ~5 on standardized scale
    # Justification: covers effect sizes seen in similar studies
    beta = pm.Normal("beta", mu=0, sigma=2.5, dims="predictors")

    # Prior: HalfNormal(1) - weakly informative, allows broad range
    # On standardized outcome, SD > 3 would be extreme
    sigma = pm.HalfNormal("sigma", sigma=1)
```

### 4. Reparameterize When Needed

If models show poor convergence (funnels, divergences), reparameterize:

```python
# Centered (may cause funnels with small group sizes)
with pm.Model() as model_centered:
    mu = pm.Normal("mu", mu=0, sigma=1)
    sigma_group = pm.HalfNormal("sigma_group", sigma=1)
    group_effect = pm.Normal("group_effect", mu=mu, sigma=sigma_group, dims="groups")

# Non-centered (more robust)
with pm.Model() as model_noncentered:
    mu = pm.Normal("mu", mu=0, sigma=1)
    sigma_group = pm.HalfNormal("sigma_group", sigma=1)
    z = pm.Normal("z", mu=0, sigma=1, dims="groups")  # standard normal
    group_effect = pm.Deterministic("group_effect", mu + z * sigma_group, dims="groups")
```

## Domain-Specific Guidance

### Biostatistics/Epidemiology

```python
# Log-odds ratios: N(0, 2.5) allows ORs from ~0.01 to ~100
log_or = pm.Normal("log_or", mu=0, sigma=2.5)

# Hazard ratios: similar on log scale
log_hr = pm.Normal("log_hr", mu=0, sigma=1)

# Prevalence: Beta unless strong prior info
prevalence = pm.Beta("prevalence", alpha=1, beta=1)  # or (2, 2) for mode at 0.5
```

### Economics/Social Science

```python
# Elasticities on log-log models
elasticity = pm.Normal("elasticity", mu=0, sigma=1)

# Treatment effects: center on 0, SD based on plausible effect sizes
treatment_effect = pm.Normal("treatment", mu=0, sigma=0.5)

# Time trends: small per-period changes
time_trend = pm.Normal("trend", mu=0, sigma=0.1)
```

### Physical Sciences

```python
# Incorporate physical constraints
# Example: diffusion coefficient (must be positive)
D = pm.LogNormal("D", mu=np.log(1e-9), sigma=0.5)  # centered on typical value

# Use informative priors from previous experiments
# Example: speed of sound with known uncertainty
c = pm.Normal("c", mu=343, sigma=5)  # m/s in air at 20°C
```

### Machine Learning / Prediction

```python
# Focus on predictive performance
# Use shrinkage priors for automatic relevance determination
import pymc_extras as pmx

# R2D2 prior - requires output_sigma and input_sigma
output_sigma = y.std()
input_sigma = X.std(axis=0)

residual_sigma, beta = pmx.R2D2M2CP(
    "r2d2",
    output_sigma=output_sigma,
    input_sigma=input_sigma,
    dims="features",
    r2=0.5,
    r2_std=0.25,
)

# Evaluate via LOO-CV
trace = pm.sample()
loo = az.loo(trace)
```
