---
name: prior-elicitation
description: >
  Load when the user is choosing priors, running prior predictive checks, calling
  find_constrained_prior, using PreliZ, or otherwise eliciting domain knowledge into a
  Bayesian model. Covers weakly informative priors, constrained priors, sensitivity analysis,
  and elicitation workflows. Triggers include: prior selection, elicitation,
  find_constrained_prior, PreliZ, prior predictive, expert/informative priors,
  weakly informative priors, constrained priors.
---

# Prior Elicitation for PyMC Models

## Decision Flowchart: Choosing a Prior Strategy

```
Do you have domain expertise or expert access?
├── YES: Can the expert quantify beliefs precisely?
│   ├── YES → Expert elicitation (SHELF protocol, PreliZ roulette/quartile)
│   └── NO → Constrained priors (find_constrained_prior, PreliZ maxent)
└── NO: Do you know the plausible scale of the parameter?
    ├── YES → Weakly informative priors (Normal, HalfNormal, Student-t)
    └── NO → Use prior predictive checks to calibrate
        └── Generate predictions → Do they cover plausible outcomes?
            ├── YES → Prior is acceptable
            └── NO → Tighten or widen prior, repeat
```

### When to Use Each Strategy

| Strategy | Use When | Example |
|---|---|---|
| Weakly informative | You know rough scale but not shape | `pm.Normal("beta", 0, 10)` for standardized predictors |
| Constrained | "95% sure the value is between A and B" | `pm.find_constrained_prior(pm.Normal, ...)` |
| MaxEnt | You have bounds and want least-informative prior | `preliz.maxent(preliz.Normal(), -1, 1, 0.94)` |
| Expert-elicited | Domain expert provides quantiles or probabilities | SHELF protocol with PreliZ |
| Hierarchical | Group-level parameters with partial pooling | `pm.Normal("mu_group", mu=mu_hyper, sigma=sigma_hyper)` |

## pm.find_constrained_prior

Finds distribution parameters such that a specified mass falls within given bounds.

```python
import pymc as pm

# "I'm 95% sure the effect is between -2 and 2"
params = pm.find_constrained_prior(
    pm.Normal,
    lower=-2, upper=2,
    mass=0.95,
    init_guess={"mu": 0, "sigma": 1},
)
# Returns: {"mu": 0.0, "sigma": 1.02}

# Use directly in model
with pm.Model() as model:
    beta = pm.Normal("beta", **params)
```

### Common Patterns

```python
# Positive parameter, 90% between 0.1 and 10
params = pm.find_constrained_prior(
    pm.LogNormal,
    lower=0.1, upper=10,
    mass=0.90,
    init_guess={"mu": 0, "sigma": 1},
)

# Rate parameter, 80% between 0.01 and 0.5
params = pm.find_constrained_prior(
    pm.Gamma,
    lower=0.01, upper=0.5,
    mass=0.80,
    init_guess={"alpha": 2, "mu": 0.1},
)

# Bounded parameter (probability), 95% between 0.2 and 0.8
params = pm.find_constrained_prior(
    pm.Beta,
    lower=0.2, upper=0.8,
    mass=0.95,
    init_guess={"alpha": 5, "beta": 5},
)
```

See `references/find_constrained_prior.md` for full API details.

## PreliZ Integration

PreliZ is a library for prior elicitation. It provides tools to translate domain knowledge into probability distributions.

### Maximum Entropy (maxent)

Find the least-informative distribution consistent with constraints:

```python
import preliz as pz

# Least-informative Normal with 94% mass in [-1, 1]
dist = pz.maxent(pz.Normal(), -1, 1, 0.94)

# Least-informative HalfNormal with 94% mass below 5
dist = pz.maxent(pz.HalfNormal(), 0, 5, 0.94)

# Use result in PyMC
with pm.Model():
    sigma = pm.HalfNormal("sigma", sigma=dist.sigma)
```

### Roulette (chip-and-bin elicitation)

Interactive method where an expert allocates "chips" to bins:

```python
# Define bins and expert-allocated chip counts
pz.roulette(x_min=0, x_max=100, nrows=10)
# Opens interactive widget — expert distributes chips across bins
# Returns fitted distribution
```

### Predictive Finder (predictive_elicitation)

Elicit priors by reasoning about observable predictions:

```python
def my_model(x, beta0, beta1, sigma):
    mu = beta0 + beta1 * x
    return pz.Normal(mu=mu, sigma=sigma)

# Expert specifies: "when x=5, y is typically between 10 and 30"
pz.predictive_finder(my_model, target=pz.Normal())
```

### Quartile Method

Specify distribution via expert-provided quartiles:

```python
# Expert says: "median is 50, Q1 is 30, Q3 is 70"
dist = pz.quartile(pz.Normal(), q1=30, q2=50, q3=70)
```

See `references/preliz.md` for comprehensive PreliZ reference.

## Prior Predictive Interpretation Checklist

Always run prior predictive checks before sampling:

```python
with model:
    prior_pred = pm.sample_prior_predictive(draws=500, random_seed=42)

# prior_pred is a DataTree (ArviZ 1.1)
prior_samples = prior_pred["prior"].ds
prior_predictive = prior_pred["prior_predictive"].ds
```

### What to Check

1. **Range of predictions**: Do simulated outcomes cover the plausible data range?
   - Plot: `az.plot_ppc_dist(prior_pred, group="prior_predictive", kind="ecdf")`
2. **Impossible values**: Are any simulated outcomes physically impossible?
   - Negative counts, probabilities outside [0,1], negative durations
3. **Scale**: Is the prior predictive spread reasonable relative to the data?
   - Too wide = vague, slow convergence. Too narrow = overly informative
4. **Shape**: Do simulated datasets look qualitatively like real data?
   - Bimodal when data is unimodal? Heavy tails when data is bounded?

### Red Flags

- Prior predictive covers 10+ orders of magnitude — priors too vague
- >10% of samples produce impossible values — wrong distribution family or scale
- Prior predictive is concentrated on a narrow range far from data — prior is miscalibrated
- All prior predictive samples look identical — prior is too tight (dogmatic)

## Expert Elicitation Workflow

### SHELF-like Protocol

1. **Define the parameter**: What does it represent? What are its units?
2. **Establish plausible range**: "What is the lowest/highest value you would expect?"
3. **Elicit quantiles**: "What value are you 25%/50%/75% sure the parameter is below?"
4. **Elicit tail behavior**: "Is there any chance of extreme values? How extreme?"
5. **Fit distribution**: Use `preliz.quartile()` or `pm.find_constrained_prior()`
6. **Validate**: Show the fitted distribution back to the expert for confirmation
7. **Prior predictive check**: Generate predictions and ask "do these look realistic?" For prior predictive checking workflows in PyMC, see the [pymc-modeling skill](../pymc-modeling/SKILL.md).

### Translating Domain Knowledge

| Expert says | Translation |
|---|---|
| "Roughly between A and B" | `find_constrained_prior(..., lower=A, upper=B, mass=0.90)` |
| "Usually around X, rarely above Y" | LogNormal or Gamma with median near X, 95th percentile near Y |
| "Positive, with diminishing probability for large values" | HalfNormal, Exponential, or HalfCauchy |
| "Could go either way, maybe 50-50" | `pm.Beta("p", alpha=1, beta=1)` or `pm.Normal("effect", 0, ...)` |
| "No more than X in absolute value" | `pm.TruncatedNormal` or `pm.Uniform(-X, X)` |

See `references/elicitation_workflows.md` for detailed protocols.

## Sensitivity Analysis

### Checking Prior Sensitivity

Conclusions are robust if they hold under different reasonable priors.

```python
# Fit model with original priors
with pm.Model() as model_original:
    beta = pm.Normal("beta", 0, 1)
    ...
    idata_orig = pm.sample()

# Fit with wider priors
with pm.Model() as model_wide:
    beta = pm.Normal("beta", 0, 10)
    ...
    idata_wide = pm.sample()

# Fit with different family
with pm.Model() as model_robust:
    beta = pm.StudentT("beta", nu=3, mu=0, sigma=1)
    ...
    idata_robust = pm.sample()

# Compare posteriors visually
import arviz as az
az.plot_forest(
    [idata_orig, idata_wide, idata_robust],
    model_names=["Original", "Wide", "Robust"],
    var_names=["beta"],
)
```

### What to Report

- If posteriors are similar across priors: conclusions are data-driven (robust)
- If posteriors differ substantially: conclusions are prior-sensitive — report sensitivity
- Always compare the posterior-to-prior contraction: strong contraction = data is informative

### Power-scaling Sensitivity

For systematic sensitivity analysis, scale the prior log-density by a factor alpha:

```python
# alpha > 1: stronger prior influence
# alpha < 1: weaker prior influence
# Compare posteriors across alpha in [0.5, 0.75, 1.0, 1.25, 1.5]
```

When posteriors are stable across alpha values, the inference is robust to prior choice.
