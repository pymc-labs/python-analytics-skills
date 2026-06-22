# PreliZ — Prior Elicitation Library Reference

PreliZ is a Python library for prior elicitation in Bayesian modeling. It provides tools to translate domain knowledge into probability distributions.

```python
import preliz as pz
```

## Maximum Entropy (maxent)

Find the least-informative (maximum entropy) distribution consistent with given constraints.

```python
pz.maxent(
    distribution,   # PreliZ distribution instance (e.g., pz.Normal())
    lower,          # Lower bound of constraint
    upper,          # Upper bound of constraint
    mass,           # Probability mass within [lower, upper]
)
```

### Examples

```python
# Least-informative Normal with 94% mass in [-1, 1]
dist = pz.maxent(pz.Normal(), -1, 1, 0.94)
print(dist)  # Normal(mu=0, sigma=0.53)

# Least-informative HalfNormal with 94% mass below 5
dist = pz.maxent(pz.HalfNormal(), 0, 5, 0.94)
print(dist)  # HalfNormal(sigma=2.66)

# Least-informative Gamma with 90% mass between 1 and 100
dist = pz.maxent(pz.Gamma(), 1, 100, 0.90)

# Least-informative Beta with 80% mass between 0.2 and 0.8
dist = pz.maxent(pz.Beta(), 0.2, 0.8, 0.80)

# LogNormal: 95% mass between 0.01 and 10
dist = pz.maxent(pz.LogNormal(), 0.01, 10, 0.95)
```

### Using maxent Results in PyMC

```python
dist = pz.maxent(pz.Normal(), -2, 2, 0.94)

with pm.Model():
    # Extract fitted parameters
    beta = pm.Normal("beta", mu=dist.mu, sigma=dist.sigma)
```

## Roulette (Chip-and-Bin Elicitation)

Interactive method for eliciting distributions from domain experts. The expert allocates "chips" across bins to express their beliefs.

```python
pz.roulette(
    x_min=0,        # Lower end of the range
    x_max=100,      # Upper end of the range
    nrows=10,       # Number of chips to allocate
    dist_names=None # List of distribution names to fit (default: tries many)
)
```

### Workflow

1. Call `pz.roulette()` — opens an interactive widget
2. Expert clicks to place chips in bins representing their belief
3. PreliZ fits candidate distributions to the chip allocation
4. Returns the best-fitting distribution

```python
# Basic roulette elicitation
result = pz.roulette(x_min=0, x_max=50, nrows=15)

# Restrict to specific distribution families
result = pz.roulette(
    x_min=0, x_max=50, nrows=15,
    dist_names=["Normal", "LogNormal", "Gamma"]
)
```

## Predictive Finder (predictive_elicitation)

Elicit priors by reasoning about observable predictions rather than parameters directly. Particularly useful when experts can reason about outcomes but not model parameters.

```python
pz.predictive_finder(
    model_func,     # Function: parameters -> predictions
    target,         # Target distribution for predictions
)
```

### Example

```python
import preliz as pz
import numpy as np

# Define the model as a function
def linear_model(x, beta0, beta1, sigma):
    mu = beta0 + beta1 * x
    return pz.Normal(mu=mu, sigma=sigma)

# Use predictive_finder to determine priors on beta0, beta1, sigma
# by reasoning about what predictions should look like
pz.predictive_finder(linear_model, target=pz.Normal())
```

The interactive interface lets the expert specify:
- "When x=0, I expect y to be around 10 (give or take 3)"
- "When x=10, I expect y to be around 50 (give or take 5)"

PreliZ then back-solves for parameter priors consistent with these predictive beliefs.

## Quartile Method

Fit a distribution by specifying expert-provided quartiles.

```python
pz.quartile(
    distribution,   # PreliZ distribution instance
    q1,             # First quartile (25th percentile)
    q2,             # Second quartile (median, 50th percentile)
    q3,             # Third quartile (75th percentile)
)
```

### Examples

```python
# Expert: "median is 50, Q1 is 30, Q3 is 70"
dist = pz.quartile(pz.Normal(), q1=30, q2=50, q3=70)

# Expert: "median is 10, Q1 is 5, Q3 is 25" (right-skewed)
dist = pz.quartile(pz.LogNormal(), q1=5, q2=10, q3=25)

# Expert: "median is 0.5, Q1 is 0.3, Q3 is 0.7" (bounded)
dist = pz.quartile(pz.Beta(), q1=0.3, q2=0.5, q3=0.7)
```

## Available Distributions

PreliZ supports all common distributions:

| Distribution | Support | Key Parameters |
|---|---|---|
| `pz.Normal()` | (-inf, inf) | mu, sigma |
| `pz.HalfNormal()` | [0, inf) | sigma |
| `pz.LogNormal()` | (0, inf) | mu, sigma |
| `pz.Gamma()` | (0, inf) | alpha, beta |
| `pz.Beta()` | [0, 1] | alpha, beta |
| `pz.StudentT()` | (-inf, inf) | nu, mu, sigma |
| `pz.Exponential()` | [0, inf) | lam |
| `pz.HalfCauchy()` | [0, inf) | beta |
| `pz.Uniform()` | [lower, upper] | lower, upper |
| `pz.TruncatedNormal()` | [lower, upper] | mu, sigma, lower, upper |

## Visualization

```python
# Plot a single distribution
dist = pz.maxent(pz.Normal(), -2, 2, 0.94)
dist.plot_pdf()

# Plot with shaded credible interval
dist.plot_pdf(pointinterval=True)

# Compare multiple distributions
pz.plot_comparison([
    pz.Normal(mu=0, sigma=1),
    pz.StudentT(nu=3, mu=0, sigma=1),
    pz.Normal(mu=0, sigma=2),
])
```

## Integration with PyMC

PreliZ distributions map directly to PyMC distributions with the same parameter names:

```python
import preliz as pz
import pymc as pm

# Elicit with PreliZ
prior_dist = pz.maxent(pz.Normal(), -2, 2, 0.94)

# Use in PyMC (parameter names match)
with pm.Model():
    beta = pm.Normal("beta", mu=prior_dist.mu, sigma=prior_dist.sigma)
```

## Best Practices

1. **Start with maxent**: When you only have bounds, maxent gives the least-opinionated prior
2. **Use predictive_finder for complex models**: Experts think in terms of outcomes, not parameters
3. **Roulette for non-technical experts**: The visual interface is intuitive
4. **Quartile for quick specification**: When experts can give Q1/Q2/Q3 estimates
5. **Always validate**: Show the fitted distribution back to the expert before using it
