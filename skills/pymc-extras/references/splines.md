# Spline Basis Functions — Detailed Reference

## Overview

Splines model nonlinear relationships by combining piecewise polynomial basis functions. `pymc-extras` provides tools for constructing B-spline bases and integrating them with PyMC models.

```python
import pymc_extras as pmx
import numpy as np
```

## BSplineBasis

### Constructor

```python
B = pmx.BSplineBasis(
    n_knots=10,     # Number of interior knots
    degree=3,       # Polynomial degree (3 = cubic, default)
)
```

### Building the Basis Matrix

```python
x = np.linspace(0, 1, 200)
basis_matrix = B.build(x)  # shape: (200, n_knots + degree - 1)
```

The number of basis functions equals `n_knots + degree - 1` (for cubic splines with 10 knots, you get 12 basis functions).

### Choosing the Number of Knots

| Data size | Suggested n_knots | Rationale |
|---|---|---|
| n < 50 | 5-8 | Avoid overfitting |
| 50 < n < 500 | 8-15 | Good balance |
| n > 500 | 15-30 | Can capture fine detail |

More knots = more flexibility = more risk of overfitting. Use penalized splines (below) to control this automatically.

### Knot Placement

By default, knots are placed at quantiles of the data (good for uneven spacing). For evenly spaced data, uniform knots work fine.

```python
# Default: quantile-based knots
B = pmx.BSplineBasis(n_knots=10, degree=3)

# The build() method handles knot placement based on the input data
basis_matrix = B.build(x)
```

## Spline Regression in PyMC

### Basic (Unpenalized) Spline

```python
import pymc as pm

x_train = np.linspace(0, 1, 100)
B = pmx.BSplineBasis(n_knots=10, degree=3)
basis_matrix = B.build(x_train)

with pm.Model() as spline_model:
    # Independent normal priors on coefficients
    w = pm.Normal("w", mu=0, sigma=1, shape=basis_matrix.shape[1])

    mu = pm.math.dot(basis_matrix, w)
    sigma = pm.HalfNormal("sigma", sigma=1)
    y = pm.Normal("y", mu=mu, sigma=sigma, observed=y_train)
```

### Penalized Spline (Random Walk Prior)

Controls wiggliness by penalizing differences between adjacent coefficients:

```python
with pm.Model() as penalized_spline:
    # Smoothing parameter — larger tau = smoother curve
    tau = pm.HalfCauchy("tau", beta=1)

    # First-order random walk: penalizes first differences
    w = pm.GaussianRandomWalk("w", sigma=tau, shape=basis_matrix.shape[1])

    mu = pm.math.dot(basis_matrix, w)
    sigma = pm.HalfNormal("sigma", sigma=1)
    y = pm.Normal("y", mu=mu, sigma=sigma, observed=y_train)
```

### Second-Order Penalization

For smoother curves, penalize second differences:

```python
with pm.Model() as smooth_spline:
    tau = pm.HalfCauchy("tau", beta=1)

    # Second-order random walk: penalizes changes in slope
    # First coefficient
    w0 = pm.Normal("w0", mu=0, sigma=5)
    # Second coefficient
    w1 = pm.Normal("w1", mu=0, sigma=5)
    # Remaining coefficients via second-order random walk
    n_basis = basis_matrix.shape[1]
    diffs = pm.Normal("diffs", mu=0, sigma=tau, shape=n_basis - 2)

    # Reconstruct full coefficient vector
    import pytensor.tensor as pt
    w = pt.concatenate([[w0], [w1], pt.cumsum(pt.cumsum(diffs) + w1 - w0) + w1])
    # Alternative: use GaussianRandomWalk with order=2 if available
```

## ZeroSumNormalBasis

Ensures the spline coefficients sum to zero, useful for identifiability in models with an intercept:

```python
B_zs = pmx.ZeroSumNormalBasis(n_knots=10, degree=3)
basis_matrix = B_zs.build(x)

with pm.Model():
    intercept = pm.Normal("intercept", mu=0, sigma=5)
    # Coefficients are constrained to sum to zero
    w = pm.ZeroSumNormal("w", sigma=1, shape=basis_matrix.shape[1])
    mu = intercept + pm.math.dot(basis_matrix, w)
    ...
```

## Multidimensional Splines

For surfaces (2D splines), use tensor product bases:

```python
# Create bases for each dimension
B_x = pmx.BSplineBasis(n_knots=8, degree=3)
B_y = pmx.BSplineBasis(n_knots=8, degree=3)

basis_x = B_x.build(x1)  # shape: (n, k1)
basis_y = B_y.build(x2)  # shape: (n, k2)

# Tensor product: element-wise product of all pairs
# shape: (n, k1 * k2)
basis_tensor = np.einsum("ni,nj->nij", basis_x, basis_y).reshape(len(x1), -1)

with pm.Model():
    w = pm.Normal("w", mu=0, sigma=1, shape=basis_tensor.shape[1])
    mu = pm.math.dot(basis_tensor, w)
    ...
```

## Prediction at New Points

```python
# Build basis for new data using the same BSplineBasis object
x_new = np.linspace(0, 1, 500)
basis_new = B.build(x_new)

# Posterior predictions
with spline_model:
    pm.set_data({"basis": basis_new})  # if using pm.Data
    ppc = pm.sample_posterior_predictive(idata)
```

Or manually:

```python
# Extract posterior coefficients
w_samples = dt["posterior"]["w"].values  # (chain, draw, n_basis)

# Predict: (chain, draw, n_basis) @ (n_basis, n_new) -> (chain, draw, n_new)
mu_pred = np.einsum("cdj,nj->cdn", w_samples, basis_new)
```

## Splines for Cyclic Data

For periodic patterns (time of day, day of year), use cyclic splines where the first and last basis functions connect smoothly:

```python
# Wrap the domain so endpoints match
# E.g., for hour of day (0-24), ensure knots at 0 and 24 connect
x_cyclic = np.linspace(0, 2 * np.pi, 100)
# Use standard B-spline on the circular domain
# Alternatively, use Fourier basis for truly periodic patterns
```

## Common Pitfalls

1. **Too many knots without penalization**: Leads to overfitting. Always use penalized splines or keep knots conservative.
2. **Extrapolation**: B-splines extrapolate linearly beyond the data range. Do not trust predictions outside the observed x range.
3. **Knot placement on sparse regions**: Knots in data-sparse regions create poorly estimated basis functions. Quantile-based placement handles this automatically.
4. **Not standardizing x**: For numerical stability, scale x to [0, 1] or similar before building the basis.
