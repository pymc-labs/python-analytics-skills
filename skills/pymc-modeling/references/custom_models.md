# Custom Distributions and Model Components

Extending PyMC with custom likelihoods, soft constraints, and simulation-based inference.

## Table of Contents
- [pm.DensityDist (Custom Likelihoods)](#pmdensitydist-custom-likelihoods)
- [pm.Potential (Soft Constraints)](#pmpotential-soft-constraints)
- [pm.Simulator (Simulation-Based Inference)](#pmsimulator-simulation-based-inference)
- [CustomDist for Priors](#customdist-for-priors)

---

## pm.DensityDist (Custom Likelihoods)

### When to Use

- Likelihood function not available in PyMC
- Complex observational models (e.g., detection limits, selection effects)
- Combining multiple data sources with custom joint likelihood
- Implementing distributions from literature not yet in PyMC

### Basic Pattern

```python
import pymc as pm
import pytensor.tensor as pt

def custom_logp(value, param1, param2):
    """Return log-probability of value given parameters."""
    # Must return a tensor, not a Python float
    return -0.5 * ((value - param1) / param2) ** 2 - pt.log(param2)

with pm.Model() as model:
    param1 = pm.Normal("param1", 0, 1)
    param2 = pm.HalfNormal("param2", 1)

    y = pm.DensityDist(
        "y",
        param1, param2,  # positional args passed to logp
        logp=custom_logp,
        observed=y_obs,
    )
```

### Adding Random Generation (for Prior/Posterior Predictive)

```python
import numpy as np

def custom_logp(value, mu, sigma):
    return pm.logp(pm.Normal.dist(mu=mu, sigma=sigma), value)

def custom_random(mu, sigma, rng=None, size=None):
    """Generate random samples for predictive checks."""
    return rng.normal(loc=mu, scale=sigma, size=size)

with pm.Model() as model:
    mu = pm.Normal("mu", 0, 1)
    sigma = pm.HalfNormal("sigma", 1)

    y = pm.DensityDist(
        "y",
        mu, sigma,
        logp=custom_logp,
        random=custom_random,
        observed=y_obs,
    )

    # Now prior/posterior predictive works
    prior_pred = pm.sample_prior_predictive()
```

### Example: Skew-Normal Distribution

```python
import pytensor.tensor as pt
from scipy import stats

def skew_normal_logp(value, mu, sigma, alpha):
    """Log-probability of skew-normal distribution."""
    z = (value - mu) / sigma
    # Log of PDF: log(2) + log(phi(z)) + log(Phi(alpha*z)) - log(sigma)
    log_pdf = (
        pt.log(2)
        - 0.5 * pt.log(2 * np.pi)
        - 0.5 * z**2
        + pt.log(0.5 * (1 + pt.erf(alpha * z / pt.sqrt(2))))
        - pt.log(sigma)
    )
    return log_pdf

def skew_normal_random(mu, sigma, alpha, rng=None, size=None):
    return stats.skewnorm.rvs(a=alpha, loc=mu, scale=sigma, size=size, random_state=rng)

with pm.Model() as model:
    mu = pm.Normal("mu", 0, 5)
    sigma = pm.HalfNormal("sigma", 2)
    alpha = pm.Normal("alpha", 0, 3)  # skewness parameter

    y = pm.DensityDist(
        "y",
        mu, sigma, alpha,
        logp=skew_normal_logp,
        random=skew_normal_random,
        observed=y_obs,
    )
```

### DensityDist with Multiple Observations

When each observation has different parameters:

```python
def vectorized_logp(value, mu, sigma):
    """value, mu, sigma can all be vectors."""
    return pm.logp(pm.Normal.dist(mu=mu, sigma=sigma), value).sum()

with pm.Model() as model:
    mu = pm.Normal("mu", 0, 1, shape=n_obs)
    sigma = pm.HalfNormal("sigma", 1)

    y = pm.DensityDist(
        "y",
        mu, sigma,
        logp=vectorized_logp,
        observed=y_obs,  # shape (n_obs,)
    )
```

---

## pm.Potential (Soft Constraints)

### When to Use

- Soft constraints on parameters (penalty terms)
- Adding arbitrary log-probability terms to the model
- Jacobian adjustments for custom transformations
- Encoding prior correlations between parameters
- Implementing complex priors that aren't standard distributions

### Important Cautions

1. **Potentials don't generate samples**: They only modify log-probability, so they have no effect on `sample_prior_predictive()` or `sample_posterior_predictive()`
2. **Use for constraints, not likelihoods**: For observed data, use `DensityDist` instead
3. **Watch for improper posteriors**: Strong negative potentials can create improper distributions

### Sum-to-Zero Constraint

Common in hierarchical models to improve identifiability:

```python
import pytensor.tensor as pt

with pm.Model(coords={"group": groups}) as model:
    # Group effects without built-in sum-to-zero
    alpha_raw = pm.Normal("alpha_raw", 0, 1, dims="group")

    # Soft constraint: penalize deviation from sum=0
    # Larger penalty_strength = harder constraint
    penalty_strength = 100
    pm.Potential("sum_to_zero", -penalty_strength * pt.sqr(alpha_raw.sum()))

    # Or use a hard constraint via centering
    alpha = pm.Deterministic("alpha", alpha_raw - alpha_raw.mean(), dims="group")
```

### Soft Ordering Constraint

When you want parameters approximately ordered but not strictly:

```python
with pm.Model(coords={"component": range(K)}) as model:
    # Component means without ordering transform
    mu = pm.Normal("mu", 0, 10, dims="component")

    # Soft ordering: penalize violations of mu[i] < mu[i+1]
    # This is gentler than hard ordering transforms
    differences = mu[1:] - mu[:-1]
    pm.Potential(
        "soft_order",
        -10 * pt.sum(pt.switch(differences < 0, differences**2, 0))
    )
```

### Prior Correlation Between Parameters

```python
with pm.Model() as model:
    alpha = pm.Normal("alpha", 0, 1)
    beta = pm.Normal("beta", 0, 1)

    # Encourage alpha and beta to be similar
    pm.Potential("correlation", -0.5 * (alpha - beta)**2)
```

### Truncation via Potential

When you need truncation not supported by `pm.Truncated`:

```python
with pm.Model() as model:
    x = pm.Normal("x", 0, 1)

    # Truncate x to [a, b] by adding -inf penalty outside bounds
    a, b = -2, 2
    pm.Potential(
        "truncation",
        pt.switch(
            pt.and_(x >= a, x <= b),
            0,
            -np.inf
        )
    )
```

### Jacobian Adjustment

When applying custom transformations:

```python
with pm.Model() as model:
    # Sample in log space
    log_sigma = pm.Normal("log_sigma", 0, 1)
    sigma = pm.Deterministic("sigma", pt.exp(log_sigma))

    # Add Jacobian for the transformation
    # d(sigma)/d(log_sigma) = sigma, so log|Jacobian| = log(sigma) = log_sigma
    pm.Potential("jacobian", log_sigma)
```

---

## pm.Simulator (Simulation-Based Inference)

### When to Use

- No tractable likelihood function available
- Complex simulation models (agent-based, differential equations with stochastic elements)
- When you can simulate from the model but can't write down the likelihood
- Approximate Bayesian Computation (ABC) scenarios

### Basic Pattern

```python
import numpy as np
import pymc as pm

def my_simulator(rng, param1, param2, size=None):
    """
    Simulate data from the model.

    Parameters
    ----------
    rng : numpy.random.Generator
        Random number generator (required)
    param1, param2 : float or array
        Model parameters
    size : tuple, optional
        Output shape

    Returns
    -------
    Simulated data matching observed data shape
    """
    # Your simulation code here
    simulated = param1 + param2 * rng.normal(size=size)
    return simulated

with pm.Model() as model:
    param1 = pm.Normal("param1", 0, 1)
    param2 = pm.HalfNormal("param2", 1)

    sim = pm.Simulator(
        "sim",
        my_simulator,
        params=[param1, param2],
        observed=observed_data,
    )

    # Must use SMC sampler for Simulator
    idata = pm.sample_smc(draws=1000, cores=4)
```

### SMC-ABC Sampling Details

```python
with pm.Model() as model:
    param1 = pm.Uniform("param1", 0, 10)
    param2 = pm.Uniform("param2", 0, 5)

    sim = pm.Simulator(
        "sim",
        my_simulator,
        params=[param1, param2],
        observed=observed_data,
    )

    # SMC with custom settings
    idata = pm.sample_smc(
        draws=2000,
        kernel="metropolis",  # or "IMH" for independent MH
        cores=4,
        chains=4,
        progressbar=True,
    )
```

### Custom Distance Function

By default, PyMC uses sum of squared differences. For custom distances:

```python
def summary_statistics(data):
    """Reduce data to summary statistics for comparison."""
    return np.array([np.mean(data), np.std(data), np.median(data)])

def my_distance(observed_summary, simulated_summary):
    """Distance between observed and simulated summaries."""
    return np.sum((observed_summary - simulated_summary)**2)

# Compute observed summaries once
obs_summary = summary_statistics(observed_data)

def simulator_with_summary(rng, param1, param2, size=None):
    simulated = param1 + param2 * rng.normal(size=size)
    return summary_statistics(simulated)

with pm.Model() as model:
    param1 = pm.Uniform("param1", 0, 10)
    param2 = pm.HalfNormal("param2", 2)

    sim = pm.Simulator(
        "sim",
        simulator_with_summary,
        params=[param1, param2],
        distance=my_distance,
        observed=obs_summary,
    )

    idata = pm.sample_smc()
```

### Example: Stochastic Differential Equation

```python
import numpy as np

def sde_simulator(rng, drift, volatility, size=None):
    """Simulate Geometric Brownian Motion."""
    n_steps = 100
    dt = 0.01

    # Initialize
    x = np.ones(size if size else 1)
    trajectory = [x.copy()]

    for _ in range(n_steps):
        dW = rng.normal(scale=np.sqrt(dt), size=x.shape)
        x = x * (1 + drift * dt + volatility * dW)
        trajectory.append(x.copy())

    return np.array(trajectory)

with pm.Model() as model:
    drift = pm.Normal("drift", 0, 0.5)
    volatility = pm.HalfNormal("volatility", 0.5)

    sim = pm.Simulator(
        "sim",
        sde_simulator,
        params=[drift, volatility],
        observed=observed_trajectory,
    )

    idata = pm.sample_smc(draws=1000)
```

---

## CustomDist for Priors

### When to Use

- Custom prior distribution not in PyMC
- Complex hierarchical prior structures
- Reparameterized distributions for better sampling

### Basic Pattern

```python
import pymc as pm
import pytensor.tensor as pt
import numpy as np

def horseshoe_logp(value, tau, lam):
    """Horseshoe prior log-probability."""
    scale = tau * lam
    return pm.logp(pm.Normal.dist(0, scale), value)

def horseshoe_random(tau, lam, rng=None, size=None):
    scale = tau * lam
    return rng.normal(0, scale, size=size)

with pm.Model() as model:
    tau = pm.HalfCauchy("tau", 1)
    lam = pm.HalfCauchy("lam", 1, shape=p)

    beta = pm.CustomDist(
        "beta",
        tau, lam,
        logp=horseshoe_logp,
        random=horseshoe_random,
        shape=p,
    )
```

### Distribution with Custom Support

```python
def triangular_logp(value, lower, mode, upper):
    """Triangular distribution log-probability."""
    # Check bounds
    in_support = pt.and_(value >= lower, value <= upper)

    # Left side of triangle
    left = pt.and_(value >= lower, value < mode)
    logp_left = pt.log(2) + pt.log(value - lower) - pt.log(upper - lower) - pt.log(mode - lower)

    # Right side of triangle
    right = pt.and_(value >= mode, value <= upper)
    logp_right = pt.log(2) + pt.log(upper - value) - pt.log(upper - lower) - pt.log(upper - mode)

    return pt.switch(
        in_support,
        pt.switch(left, logp_left, logp_right),
        -np.inf
    )

def triangular_random(lower, mode, upper, rng=None, size=None):
    return rng.triangular(lower, mode, upper, size=size)

with pm.Model() as model:
    x = pm.CustomDist(
        "x",
        0, 0.3, 1,  # lower, mode, upper
        logp=triangular_logp,
        random=triangular_random,
    )
```

---

## Best Practices

### 1. Test Custom Distributions

```python
# Verify logp integrates to 1 (approximately)
import scipy.integrate as integrate

def test_logp_normalization(logp_func, params, bounds=(-10, 10)):
    """Check that exp(logp) integrates to ~1."""
    def pdf(x):
        # Compile and evaluate the logp tensor
        import pytensor
        import pytensor.tensor as pt
        x_var = pt.dscalar("x")
        logp_expr = logp_func(x_var, *[pt.constant(p) for p in params])
        logp_fn = pytensor.function([x_var], logp_expr)
        return np.exp(logp_fn(x))

    integral, _ = integrate.quad(pdf, bounds[0], bounds[1])
    assert np.isclose(integral, 1.0, rtol=0.01), f"Integral = {integral}"
```

### 2. Check Gradient Computation

```python
import pytensor
import pytensor.tensor as pt

# Ensure logp is differentiable
def check_gradient(logp_func, params):
    x = pt.dscalar("x")
    logp = logp_func(x, *[pt.constant(p) for p in params])
    grad = pytensor.grad(logp, x)

    # Compile and test
    grad_fn = pytensor.function([x], grad)
    test_values = np.linspace(-3, 3, 10)
    for v in test_values:
        g = grad_fn(v)
        assert np.isfinite(g), f"Non-finite gradient at x={v}"
```

### 3. Prefer Built-in When Possible

Before implementing custom distributions, check:
- `pymc` core distributions
- `pymc_extras` for specialized distributions
- Transformations of existing distributions

### 4. Document Custom Components

```python
def my_custom_logp(value, param1, param2):
    """
    Custom distribution log-probability.

    This implements the Foo distribution with PDF:
        f(x; a, b) = ...

    Parameters
    ----------
    value : tensor
        Point at which to evaluate logp
    param1 : tensor
        First parameter (location)
    param2 : tensor
        Second parameter (scale, must be > 0)

    Returns
    -------
    tensor
        Log-probability

    References
    ----------
    Smith (2020). "The Foo Distribution". Journal of Stats.
    """
    pass
```
