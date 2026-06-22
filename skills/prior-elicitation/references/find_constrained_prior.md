# pm.find_constrained_prior — Detailed Reference

## Function Signature

```python
pm.find_constrained_prior(
    distribution,       # PyMC distribution class (e.g., pm.Normal)
    lower,              # Lower bound of the constraint interval
    upper,              # Upper bound of the constraint interval
    mass,               # Probability mass within [lower, upper]
    init_guess,         # Dict of initial parameter guesses
    fixed_params=None,  # Dict of parameters to hold fixed
)
```

## Parameters

- **distribution**: A PyMC distribution class (not an instance). E.g., `pm.Normal`, `pm.LogNormal`, `pm.Gamma`.
- **lower**: Lower bound of the credible interval.
- **upper**: Upper bound of the credible interval.
- **mass**: Desired probability mass within the interval. E.g., 0.95 means 95% of the distribution falls between lower and upper.
- **init_guess**: Dictionary of starting values for the optimizer. Keys must match the distribution's parameter names.
- **fixed_params**: Optional dictionary of parameters to hold fixed during optimization. Useful when you want to fix one parameter and solve for others.

## Return Value

Returns a dictionary of optimized parameter values that can be unpacked directly into a PyMC distribution.

## Examples

### Normal Distribution

```python
# 95% of mass between -2 and 2
params = pm.find_constrained_prior(
    pm.Normal,
    lower=-2, upper=2,
    mass=0.95,
    init_guess={"mu": 0, "sigma": 1},
)
# Returns: {"mu": 0.0, "sigma": ~1.02}

# Asymmetric: 90% between 0 and 10
params = pm.find_constrained_prior(
    pm.Normal,
    lower=0, upper=10,
    mass=0.90,
    init_guess={"mu": 5, "sigma": 2},
)
```

### LogNormal Distribution

```python
# Positive parameter: 90% between 0.1 and 100
params = pm.find_constrained_prior(
    pm.LogNormal,
    lower=0.1, upper=100,
    mass=0.90,
    init_guess={"mu": 1, "sigma": 1},
)
```

### Gamma Distribution

```python
# Rate parameter: 95% between 0.01 and 2.0
params = pm.find_constrained_prior(
    pm.Gamma,
    lower=0.01, upper=2.0,
    mass=0.95,
    init_guess={"alpha": 2, "beta": 2},
)
```

### Beta Distribution

```python
# Probability: 90% between 0.1 and 0.9
params = pm.find_constrained_prior(
    pm.Beta,
    lower=0.1, upper=0.9,
    mass=0.90,
    init_guess={"alpha": 2, "beta": 2},
)
```

### With Fixed Parameters

```python
# Fix mu=0, solve for sigma only
params = pm.find_constrained_prior(
    pm.Normal,
    lower=-3, upper=3,
    mass=0.99,
    init_guess={"sigma": 1},
    fixed_params={"mu": 0},
)
# Returns: {"sigma": ~1.16, "mu": 0}
```

### Student-t Distribution

```python
# Heavy-tailed: fix nu=3, find mu and sigma
params = pm.find_constrained_prior(
    pm.StudentT,
    lower=-5, upper=5,
    mass=0.95,
    init_guess={"mu": 0, "sigma": 1},
    fixed_params={"nu": 3},
)
```

## Using Results in a Model

```python
params = pm.find_constrained_prior(
    pm.Normal, lower=-2, upper=2, mass=0.95,
    init_guess={"mu": 0, "sigma": 1},
)

with pm.Model() as model:
    # Unpack directly
    beta = pm.Normal("beta", **params)

    # Or use individual values
    beta = pm.Normal("beta", mu=params["mu"], sigma=params["sigma"])
```

## Common Pitfalls

1. **Wrong init_guess**: If optimization fails, try different starting values.
2. **Infeasible constraints**: Some distribution-constraint combinations have no solution (e.g., 99% of a HalfNormal between 0 and 0.01 with sigma > 0.001).
3. **Parameter names**: Must match the PyMC distribution parameter names exactly. Check `pm.Normal.rv_op.ndims_params` or the PyMC docs.
4. **Mass too close to 1.0**: Very high mass values (>0.999) can cause numerical issues.

## Relationship to PreliZ

`pm.find_constrained_prior` solves a single constraint (mass within bounds). For more complex elicitation (multiple quantiles, maximum entropy, interactive), use PreliZ instead. The two complement each other:

- `find_constrained_prior`: Quick, simple, built into PyMC
- `preliz.maxent`: Least-informative distribution given constraints
- `preliz.quartile`: Fit from multiple quantiles
