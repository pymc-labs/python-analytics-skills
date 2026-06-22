# Expert Elicitation Workflows

Step-by-step protocols for translating domain knowledge into probability distributions for Bayesian models.

## SHELF Protocol (Adapted)

The SHELF (Sheffield Elicitation Framework) protocol is a structured method for eliciting probability distributions from domain experts. This adaptation is tailored for PyMC modeling.

### Step 1: Define the Quantity

- Clearly define the parameter being elicited
- Specify units, direction, and interpretation
- Ensure the expert understands what the parameter represents in the model

**Example script**: "We need a prior for the treatment effect, measured in mmHg reduction in blood pressure. A positive value means the treatment lowers blood pressure."

### Step 2: Establish the Range

Ask sequentially:
1. "What is the lowest plausible value?" (not impossible, just very unlikely)
2. "What is the highest plausible value?"
3. "Could the value be outside this range?" (calibrate extremes)

Record: `lower_plausible`, `upper_plausible`

### Step 3: Elicit the Median

"What value do you think is equally likely to be above or below the true value?"

Record: `median`

### Step 4: Elicit Quartiles

"What value are you 25% sure the parameter is below?" (lower quartile)
"What value are you 75% sure the parameter is below?" (upper quartile)

Record: `q1`, `q3`

### Step 5: Fit Distribution

```python
import preliz as pz

# Method 1: Quartile fitting
dist = pz.quartile(pz.Normal(), q1=q1, q2=median, q3=q3)

# Method 2: If expert only gave range
import pymc as pm
params = pm.find_constrained_prior(
    pm.Normal,
    lower=lower_plausible, upper=upper_plausible,
    mass=0.95,
    init_guess={"mu": median, "sigma": (upper_plausible - lower_plausible) / 4},
)
```

### Step 6: Validate with Expert

Show the fitted distribution to the expert:
- "Does this look right?"
- "Is the spread about right?"
- "Are the tails reasonable?"

```python
dist.plot_pdf(pointinterval=True)
```

### Step 7: Prior Predictive Check

Generate model predictions using the elicited prior and ask:
- "Do these predicted outcomes look plausible?"
- "Are any predictions clearly unreasonable?"

## Workflow for Multiple Parameters

When eliciting priors for multiple model parameters:

### Independent Elicitation

When parameters are conceptually independent, elicit each separately:

```python
# Intercept: "typical baseline value is around 120, range 90-150"
params_intercept = pm.find_constrained_prior(
    pm.Normal, lower=90, upper=150, mass=0.95,
    init_guess={"mu": 120, "sigma": 15},
)

# Slope: "effect per unit is small, maybe 0-5"
params_slope = pm.find_constrained_prior(
    pm.LogNormal, lower=0.1, upper=5, mass=0.90,
    init_guess={"mu": 0.5, "sigma": 0.5},
)
```

### Predictive Elicitation (Preferred for Complex Models)

When parameters interact, elicit at the prediction level:

```python
import preliz as pz

def model(x, beta0, beta1, sigma):
    mu = beta0 + beta1 * x
    return pz.Normal(mu=mu, sigma=sigma)

# Expert reasons about predictions:
# "When x=0, outcome is about 120 (SD ~10)"
# "When x=10, outcome is about 140 (SD ~15)"
pz.predictive_finder(model, target=pz.Normal())
```

This back-solves for parameter priors that are consistent with the expert's predictive beliefs.

## Translating Common Expert Statements

### "Usually between A and B"

```python
# 80-90% mass in [A, B]
params = pm.find_constrained_prior(
    pm.Normal, lower=A, upper=B, mass=0.85,
    init_guess={"mu": (A + B) / 2, "sigma": (B - A) / 4},
)
```

### "Rarely exceeds X"

```python
# For positive quantities
dist = pz.maxent(pz.HalfNormal(), 0, X, 0.95)
# or
dist = pz.maxent(pz.Exponential(), 0, X, 0.95)
```

### "Could be positive or negative, but small"

```python
# Symmetric around zero
params = pm.find_constrained_prior(
    pm.Normal, lower=-X, upper=X, mass=0.95,
    init_guess={"mu": 0, "sigma": X / 2},
)
```

### "Almost certainly positive, probably around X"

```python
# LogNormal centered near X
params = pm.find_constrained_prior(
    pm.LogNormal,
    lower=X * 0.1, upper=X * 10,
    mass=0.95,
    init_guess={"mu": np.log(X), "sigma": 1},
)
```

### "A proportion, probably around P"

```python
# Beta distribution centered near P
# Spread depends on confidence
dist = pz.quartile(pz.Beta(), q1=P - 0.1, q2=P, q3=P + 0.1)
```

### "No real idea, but it must be positive"

```python
# Weakly informative: HalfNormal or HalfCauchy
# Scale based on order of magnitude of the data
with pm.Model():
    sigma = pm.HalfNormal("sigma", sigma=data_sd * 2)
    # or for heavier tails:
    sigma = pm.HalfCauchy("sigma", beta=data_sd)
```

## Hierarchical Prior Elicitation

For hierarchical models, elicit at the group level:

1. "What is the typical value across all groups?" -> hyperprior mean
2. "How much do groups vary?" -> hyperprior scale
3. "Could any group be very different from the rest?" -> tail behavior (Normal vs Student-t)

```python
with pm.Model():
    # "Typical group mean is around 50, groups vary by about 10"
    mu_hyper = pm.Normal("mu_hyper", mu=50, sigma=10)
    sigma_hyper = pm.HalfNormal("sigma_hyper", sigma=10)

    # Group-level parameters
    mu_group = pm.Normal("mu_group", mu=mu_hyper, sigma=sigma_hyper, shape=n_groups)
```

## Documentation Template

When recording elicited priors, document:

```
Parameter: [name]
Expert: [who provided the information]
Date: [when]
Definition: [what the parameter represents, units]
Elicitation method: [SHELF/roulette/maxent/predictive]
Expert statements:
  - "..."
  - "..."
Fitted distribution: [family(params)]
Validation: [expert approved? prior predictive reasonable?]
```

## Common Pitfalls

1. **Anchoring**: Don't show the expert a default value first — it biases their response
2. **Overconfidence**: Experts typically give ranges that are too narrow — consider widening by 20-50%
3. **Coherence**: Check that elicited priors for related parameters are mutually consistent via prior predictive
4. **Base rate neglect**: Experts may not account for base rates — verify with data where available
5. **Conflating parameters and predictions**: Always clarify whether the expert is talking about a parameter value or an observable outcome
