# Gaussian Processes in PyMC

## Table of Contents
- [When to Use Which GP Implementation](#when-to-use-which-gp-implementation)
- [HSGP: The Scalable Default](#hsgp-the-scalable-default)
- [HSGPPeriodic: Periodic Components](#hsgpperiodic-periodic-components)
- [HSGP Parameter Selection](#hsgp-parameter-selection)
- [Assessing HSGP Approximation Quality](#assessing-hsgp-approximation-quality)
- [Covariance Functions](#covariance-functions)
- [Priors for GP Hyperparameters](#priors-for-gp-hyperparameters)
- [GP Implementations Reference](#gp-implementations-reference)
- [Common Patterns](#common-patterns)
- [Troubleshooting](#troubleshooting)

## When to Use Which GP Implementation

| Scenario | Recommended | Why |
|----------|-------------|-----|
| n > 500 points, 1-3D | **HSGP** | O(nm) vs O(n³); practical for large datasets |
| n < 500 points | Marginal/Latent GP | Full GP more accurate; overhead of HSGP not justified |
| Periodic patterns | **HSGPPeriodic** | Purpose-built for periodic covariance |
| Non-Gaussian likelihood (any n) | HSGP or Latent GP | Cannot marginalize latent function |
| > 3 input dimensions | Full GP or inducing points | HSGP scales poorly with dimension |
| Non-stationary patterns | Full GP | HSGP requires stationary kernels |

**Default recommendation**: Use HSGP for most real-world problems. It's fast, integrates cleanly with other model components, and handles the typical case (time series, 1-2D spatial) well.

## HSGP: The Scalable Default

Hilbert Space Gaussian Process (HSGP) approximates a GP using a fixed set of basis functions. The key insight: basis functions don't depend on kernel hyperparameters, so the model becomes a linear combination of pre-computed basis vectors—fast to sample and easy to combine with other components.

### Basic Usage

```python
import pymc as pm
import numpy as np

with pm.Model() as model:
    # Hyperparameters
    ell = pm.InverseGamma("ell", alpha=5, beta=5)
    eta = pm.HalfNormal("eta", sigma=2)
    sigma = pm.HalfNormal("sigma", sigma=0.5)

    # Covariance function
    cov = eta**2 * pm.gp.cov.Matern52(1, ls=ell)

    # HSGP approximation
    gp = pm.gp.HSGP(m=[20], c=1.5, cov_func=cov)
    f = gp.prior("f", X=X[:, None])  # X must be 2D

    # Likelihood
    y_ = pm.Normal("y", mu=f, sigma=sigma, observed=y)

    idata = pm.sample(nuts_sampler="nutpie")
```

### Key Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `m` | list[int] | Number of basis functions per dimension. Higher = better approximation of small lengthscales |
| `c` | float | Boundary extension factor. `L = c * max(\|X\|)`. Minimum 1.2; typically 1.3-2.0 |
| `L` | list[float] | Explicit boundary. Alternative to `c`. Domain is `[-L, L]` |
| `drop_first` | bool | Drop first basis vector (useful when model has intercept) |
| `cov_func` | Covariance | Must implement `power_spectral_density` method |

**Either `c` or `L` is required, not both.**

### Supported Kernels

HSGP works with stationary kernels that have a power spectral density:

- `pm.gp.cov.ExpQuad` (RBF/squared exponential)
- `pm.gp.cov.Matern52` (recommended default)
- `pm.gp.cov.Matern32`
- `pm.gp.cov.Matern12` / `pm.gp.cov.Exponential`
- Sums of the above

**Not supported**: `Periodic` (use HSGPPeriodic), non-stationary kernels, product kernels.

### Prediction at New Locations

```python
# After sampling
X_new = np.linspace(X.min(), X.max(), 200)[:, None]

with model:
    f_star = gp.conditional("f_star", Xnew=X_new)
    pred = pm.sample_posterior_predictive(idata, var_names=["f_star"])
```

### Using pm.Data for Out-of-Sample Prediction

For prediction without recompiling:

```python
with pm.Model() as model:
    X_data = pm.Data("X", X[:, None])

    gp = pm.gp.HSGP(m=[25], c=1.5, cov_func=cov)
    f = gp.prior("f", X=X_data)

    # ... rest of model ...
    idata = pm.sample()

# Predict at new locations
with model:
    pm.set_data({"X": X_new[:, None]})
    pred = pm.sample_posterior_predictive(idata, var_names=["f"])
```

### Linearized Form (Advanced)

Access basis functions directly for custom models:

```python
with pm.Model() as model:
    gp = pm.gp.HSGP(m=[25], c=1.5, cov_func=cov)

    # Get basis matrix (phi) and spectral weights (sqrt_psd)
    phi, sqrt_psd = gp.prior_linearized(X=X[:, None])

    # Manual linear combination
    coeffs = pm.Normal("coeffs", 0, 1, shape=gp.n_basis_vectors)
    f = pm.Deterministic("f", phi @ (coeffs * sqrt_psd))
```

## HSGPPeriodic: Periodic Components

For periodic patterns (seasonality, cycles), use `HSGPPeriodic`. It uses a different basis approximation (stochastic resonators) designed for the periodic kernel.

### Basic Usage

```python
with pm.Model() as model:
    # Hyperparameters
    amplitude = pm.HalfNormal("amplitude", sigma=1)
    ls = pm.InverseGamma("ls", alpha=5, beta=1)

    # Periodic covariance (period is known, e.g., 365.25 days for yearly)
    cov = pm.gp.cov.Periodic(input_dim=1, period=365.25, ls=ls)

    # HSGPPeriodic
    gp = pm.gp.HSGPPeriodic(m=20, scale=amplitude, cov_func=cov)
    f = gp.prior("f", X=X[:, None])
```

### Key Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `m` | int | Number of basis functions. Higher = captures more complex periodic structure |
| `scale` | TensorLike | Amplitude (standard deviation) of the GP. Default 1.0 |
| `cov_func` | Periodic | Must be `pm.gp.cov.Periodic` |

### Important Limitations

- **1D only**: HSGPPeriodic only works with 1-dimensional inputs
- **Known period**: The period in `cov_func` should be fixed or have a tight prior
- **No boundary parameter**: Unlike HSGP, no `c` or `L` needed

### Choosing m for HSGPPeriodic

- `m=10-15`: Smooth periodic patterns
- `m=20-30`: Moderate complexity (typical for yearly seasonality)
- `m=50+`: Complex periodic structure with sharp features

## HSGP Parameter Selection

Choosing `m` and `c` is crucial for HSGP accuracy.

### Automatic Selection with Heuristics

PyMC provides a helper function for initial parameter estimates:

```python
# Get recommended m and c based on data range and lengthscale prior
m, c = pm.gp.hsgp_approx.approx_hsgp_hyperparams(
    x_range=[X.min(), X.max()],
    lengthscale_range=[ell_min, ell_max],  # from your prior
    cov_func="matern52"  # or "expquad", "matern32"
)
```

**Warning**: These are minimum recommendations. In practice, you may need to increase `c` (especially for extrapolation) or verify with prior predictive checks.

### Manual Selection Guidelines

**For `c` (boundary factor)**:

| Situation | Recommended `c` |
|-----------|-----------------|
| Interpolation only | 1.2-1.5 |
| Modest extrapolation | 1.5-2.0 |
| Significant extrapolation | 2.0-4.0 |
| Very long lengthscales | 3.0+ |

`c < 1.2` causes basis functions to pinch to zero at domain edges—avoid.

**For `m` (basis functions)**:

| Lengthscale regime | Recommended `m` |
|-------------------|-----------------|
| Very smooth (ℓ ≫ data range) | 10-20 |
| Moderate smoothness | 20-40 |
| Can be wiggly (ℓ small) | 40-80 |
| Highly variable | 80+ |

**Key insight**: Increasing `c` requires increasing `m` to maintain approximation quality. They work together.

### Compute L from Data

If specifying `L` explicitly instead of `c`:

```python
# Center data first
X_centered = X - X.mean()
X_range = X_centered.max() - X_centered.min()

# L should be at least half the centered range, with buffer
L = 1.5 * (X_range / 2)

gp = pm.gp.HSGP(m=[30], L=[L], cov_func=cov)
```

### Multi-Dimensional HSGP

For 2D+ inputs, specify `m` and `L`/`c` per dimension:

```python
# 2D spatial data
gp = pm.gp.HSGP(
    m=[15, 15],           # 15 basis functions per dimension
    c=1.5,                # same c for all dimensions
    cov_func=pm.gp.cov.Matern52(2, ls=[ell_x, ell_y])
)
f = gp.prior("f", X=X_2d)  # X_2d is (n, 2)
```

**Computational cost**: Total basis functions = product of m values. For `m=[15, 15]`, that's 225 basis functions. HSGP becomes inefficient beyond 2-3 dimensions.

### The drop_first Parameter

The first HSGP basis function is nearly constant (like an intercept). If your model already has an intercept, set `drop_first=True` to improve sampling:

```python
with pm.Model() as model:
    intercept = pm.Normal("intercept", 0, 10)

    gp = pm.gp.HSGP(m=[25], c=1.5, drop_first=True, cov_func=cov)
    f = gp.prior("f", X=X[:, None])

    mu = intercept + f  # intercept handled separately
```

## Assessing HSGP Approximation Quality

### Prior Predictive Checks (Essential)

Before fitting, verify the HSGP produces sensible prior samples:

```python
with model:
    prior = pm.sample_prior_predictive(draws=100)

# Plot prior function draws
import matplotlib.pyplot as plt
for i in range(20):
    plt.plot(X, prior.prior["f"][0, i], alpha=0.3)
plt.title("HSGP Prior Samples")
```

**Check for**:
- Reasonable smoothness (not too wiggly or flat)
- Amplitude covers plausible range
- No artifacts at domain boundaries

### Gram Matrix Comparison (Diagnostic)

Compare the true covariance matrix K to the HSGP approximation:

```python
# True covariance matrix
K_true = cov(X[:, None]).eval()

# HSGP approximation: Phi * Lambda * Phi^T
phi, sqrt_psd = gp.prior_linearized(X=X[:, None])
phi_val = phi.eval()
psd_val = sqrt_psd.eval()
K_approx = phi_val @ np.diag(psd_val**2) @ phi_val.T

# Visual comparison
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].imshow(K_true); axes[0].set_title("True K")
axes[1].imshow(K_approx); axes[1].set_title("HSGP Approximation")
```

If matrices look qualitatively different, increase `m` or adjust `c`.

### Comparing to Full GP (Ground Truth)

For critical applications, compare HSGP results to exact GP on a data subset:

```python
# Subsample for tractable exact GP
idx = np.random.choice(len(X), size=min(300, len(X)), replace=False)
X_sub, y_sub = X[idx], y[idx]

# Fit exact GP
with pm.Model() as exact_model:
    # same priors...
    gp_exact = pm.gp.Marginal(cov_func=cov)
    y_ = gp_exact.marginal_likelihood("y", X=X_sub[:, None], y=y_sub, sigma=sigma)
    idata_exact = pm.sample()

# Compare posteriors of hyperparameters
```

## Covariance Functions

### Stationary Kernels (HSGP-compatible)

```python
# Squared Exponential / RBF - infinitely differentiable, very smooth
cov = pm.gp.cov.ExpQuad(input_dim=1, ls=ell)

# Matern family - finite differentiability (recommended default)
cov = pm.gp.cov.Matern52(input_dim=1, ls=ell)  # twice differentiable
cov = pm.gp.cov.Matern32(input_dim=1, ls=ell)  # once differentiable
cov = pm.gp.cov.Matern12(input_dim=1, ls=ell)  # continuous but rough

# Rational Quadratic - mixture of length scales
cov = pm.gp.cov.RatQuad(input_dim=1, ls=ell, alpha=alpha)
```

**Matern52 is the recommended default**: more numerically stable than ExpQuad (heavier spectral tails), realistic smoothness for most applications.

### Periodic Kernel (HSGPPeriodic only)

```python
# Pure periodic
cov = pm.gp.cov.Periodic(input_dim=1, period=period, ls=ell)
```

For decaying periodicity (periodic structure that fades over time), combine HSGPPeriodic with HSGP trend—see [Common Patterns](#common-patterns).

### Combining Kernels

**Addition (independent effects)**:
```python
# Works with HSGP
cov = cov_smooth + cov_rough
```

**Product (modulation)**: Not directly supported by HSGP. Use separate GPs added together, or full GP.

### Multi-dimensional with ARD

Automatic Relevance Determination uses separate length scales per input dimension:

```python
# One length scale per feature
ell = pm.InverseGamma("ell", alpha=5, beta=5, dims="features")
cov = pm.gp.cov.Matern52(input_dim=D, ls=ell)
```

## Priors for GP Hyperparameters

### Length Scale (ℓ)

The length scale controls correlation distance. Smaller = more wiggly.

**Recommended: InverseGamma prior** (Betancourt methodology)

Standard priors like HalfNormal or Exponential put significant mass near zero, which causes the GP to become overly "wiggly" and leads to poor convergence. The InverseGamma prior concentrates mass between the shortest and longest pairwise distances in the data, ensuring the GP only learns features that are resolvable by the provided data points.

```python
# InverseGamma: prevents lengthscale going to 0 or infinity (recommended)
ell = pm.InverseGamma("ell", alpha=5, beta=5)

# Calibrate to data scale: concentrate prior between data point spacing
min_dist = 0.1  # approximate minimum pairwise distance
max_dist = X.max() - X.min()  # data range

# Choose alpha and beta to put mass in [min_dist, max_dist]
# Rule of thumb: beta ~ desired_mode * (alpha - 1)
ell = pm.InverseGamma("ell", alpha=5, beta=max_dist / 2)

# LogNormal: for order-of-magnitude uncertainty
ell = pm.LogNormal("ell", mu=np.log(expected_ell), sigma=1)
```

**Why InverseGamma works**: The prior prevents the sampler from drifting into regions where:
- Lengthscale is so small the GP becomes white noise (overfitting)
- Lengthscale is so large the GP becomes a flat line (underfitting)

### Amplitude (η / marginal standard deviation)

Controls the magnitude of function variation.

```python
# HalfNormal: weakly informative
eta = pm.HalfNormal("eta", sigma=2)

# Scale to outcome
y_std = y.std()
eta = pm.HalfNormal("eta", sigma=2 * y_std)
```

### Noise (σ)

Observation noise standard deviation.

```python
sigma = pm.HalfNormal("sigma", sigma=1)

# If you have measurement error estimates
sigma = pm.HalfNormal("sigma", sigma=estimated_noise)
```

### Period

For periodic kernels:

```python
# Fixed (known period)
period = 365.25  # yearly

# Uncertain but approximately known
period = pm.Normal("period", mu=365.25, sigma=5)

# Very uncertain
period = pm.LogNormal("period", mu=np.log(365), sigma=0.1)
```

## GP Implementations Reference

### Marginal GP

Best for standard regression with Gaussian noise and moderate data size:

```python
with pm.Model() as model:
    cov = eta**2 * pm.gp.cov.Matern52(1, ls=ell)
    gp = pm.gp.Marginal(cov_func=cov)

    # Marginal likelihood integrates out latent f
    y_ = gp.marginal_likelihood("y", X=X[:, None], y=y, sigma=sigma)

    idata = pm.sample()

# Prediction
with model:
    f_star = gp.conditional("f_star", X_new)
    pred = pm.sample_posterior_predictive(idata, var_names=["f_star"])
```

### Latent GP

Required when likelihood is non-Gaussian (classification, counts):

```python
with pm.Model() as model:
    cov = eta**2 * pm.gp.cov.Matern52(1, ls=ell)
    gp = pm.gp.Latent(cov_func=cov)

    # Sample latent function explicitly
    f = gp.prior("f", X=X[:, None])

    # Non-Gaussian likelihood
    y_ = pm.Bernoulli("y", p=pm.math.sigmoid(f), observed=y)
```

## Common Patterns

### Time Series with Trend + Seasonality

Combine HSGP (trend) with HSGPPeriodic (seasonality):

```python
# Standardize time for numerical stability
time_mean, time_std = t.mean(), t.std()
t_scaled = (t - time_mean) / time_std

coords = {"obs": np.arange(len(y))}

with pm.Model(coords=coords) as model:
    # === TREND (HSGP) ===
    eta_trend = pm.HalfNormal("eta_trend", sigma=1)
    ell_trend = pm.LogNormal("ell_trend", mu=np.log(2), sigma=1)  # long lengthscale
    cov_trend = eta_trend**2 * pm.gp.cov.ExpQuad(1, ls=ell_trend)
    gp_trend = pm.gp.HSGP(m=[20], c=1.5, cov_func=cov_trend)
    f_trend = gp_trend.prior("f_trend", X=t_scaled[:, None])

    # === YEARLY SEASONALITY (HSGPPeriodic) ===
    eta_year = pm.HalfNormal("eta_year", sigma=0.5)
    ell_year = pm.InverseGamma("ell_year", alpha=5, beta=1)
    period_year = 365.25 / time_std  # period in scaled units
    cov_year = pm.gp.cov.Periodic(1, period=period_year, ls=ell_year)
    gp_year = pm.gp.HSGPPeriodic(m=20, scale=eta_year, cov_func=cov_year)
    f_year = gp_year.prior("f_year", X=t_scaled[:, None])

    # === COMBINE ===
    mu = f_trend + f_year

    sigma = pm.HalfNormal("sigma", sigma=0.5)
    y_ = pm.Normal("y", mu=mu, sigma=sigma, observed=y, dims="obs")
```

### GP with Parametric Mean

Combine linear trend with GP for residual structure:

```python
with pm.Model() as model:
    # Linear component
    alpha = pm.Normal("alpha", 0, 10)
    beta = pm.Normal("beta", 0, 1)
    mu_linear = alpha + beta * X

    # GP for nonlinear residual
    gp = pm.gp.HSGP(m=[25], c=1.5, drop_first=True, cov_func=cov)
    f = gp.prior("f", X=X[:, None])

    mu = mu_linear + f
    y_ = pm.Normal("y", mu=mu, sigma=sigma, observed=y)
```

### GP Classification

```python
with pm.Model() as model:
    cov = eta**2 * pm.gp.cov.Matern52(1, ls=ell)
    gp = pm.gp.HSGP(m=[30], c=1.5, cov_func=cov)
    f = gp.prior("f", X=X[:, None])

    # Logistic (or probit) link
    p = pm.Deterministic("p", pm.math.sigmoid(f))
    y_ = pm.Bernoulli("y", p=p, observed=y)
```

### Heteroscedastic GP (Input-dependent Noise)

Model both mean and noise as GPs:

```python
with pm.Model() as model:
    # GP for mean function
    cov_mean = eta_mean**2 * pm.gp.cov.Matern52(1, ls=ell_mean)
    gp_mean = pm.gp.HSGP(m=[25], c=1.5, cov_func=cov_mean)
    f_mean = gp_mean.prior("f_mean", X=X[:, None])

    # GP for log-noise (ensures positivity)
    cov_noise = eta_noise**2 * pm.gp.cov.Matern52(1, ls=ell_noise)
    gp_noise = pm.gp.HSGP(m=[15], c=1.5, cov_func=cov_noise)
    f_log_noise = gp_noise.prior("f_log_noise", X=X[:, None])

    sigma = pm.math.exp(f_log_noise)
    y_ = pm.Normal("y", mu=f_mean, sigma=sigma, observed=y)
```

### 2D Spatial GP

```python
# X_spatial is (n, 2) with coordinates
coords = {"obs": np.arange(n)}

with pm.Model(coords=coords) as model:
    ell = pm.InverseGamma("ell", alpha=5, beta=5, shape=2)
    eta = pm.HalfNormal("eta", sigma=2)

    cov = eta**2 * pm.gp.cov.Matern52(input_dim=2, ls=ell)
    gp = pm.gp.HSGP(m=[12, 12], c=1.5, cov_func=cov)
    f = gp.prior("f", X=X_spatial)

    sigma = pm.HalfNormal("sigma", sigma=1)
    y_ = pm.Normal("y", mu=f, sigma=sigma, observed=y, dims="obs")
```

## Troubleshooting

### Divergences in GP Models

| Symptom | Likely Cause | Solution |
|---------|--------------|----------|
| Divergences at short ℓ | Prior allows implausibly small lengthscales | Use InverseGamma prior; increase α |
| Divergences at boundaries | c too small | Increase c to 2.0+ |
| Many divergences | Poor posterior geometry | Try non-centered parameterization |

### Poor HSGP Approximation

**Symptoms**: Posterior very different from exact GP; boundary artifacts; unrealistic wiggliness.

**Solutions**:
1. Increase `m` (especially if lengthscale prior includes small values)
2. Increase `c` (especially for long lengthscales or extrapolation)
3. Check prior predictive—if it looks wrong, the approximation parameters are wrong

### Slow Sampling

- Reduce `m` if approximation quality allows
- Use `nutpie` sampler (2-5x faster)
- Consider if full GP would be faster for small n

### Numerical Issues

**ExpQuad underflow**: ExpQuad's spectral density decays exponentially fast; large lengthscales can cause underflow. Switch to Matern52 which has heavier tails.

**Add jitter for stability**:
```python
cov = eta**2 * pm.gp.cov.Matern52(1, ls=ell) + pm.gp.cov.WhiteNoise(sigma=1e-6)
```

### First Basis Vector Identifiability

If trace plots show the first HSGP coefficient poorly identified alongside an intercept:
```python
gp = pm.gp.HSGP(m=[25], c=1.5, drop_first=True, cov_func=cov)
```

## References

- Ruitort-Mayol et al. (2022): "Practical Hilbert Space Approximate Bayesian Gaussian Processes for Probabilistic Programming"
- [PyMC HSGP Documentation](https://www.pymc.io/projects/docs/en/stable/api/gp/generated/pymc.gp.HSGP.html)
- [HSGP Reference & First Steps](https://www.pymc.io/projects/examples/en/latest/gaussian_processes/HSGP-Basic.html)
- [HSGP Advanced Usage](https://www.pymc.io/projects/examples/en/latest/gaussian_processes/HSGP-Advanced.html)
