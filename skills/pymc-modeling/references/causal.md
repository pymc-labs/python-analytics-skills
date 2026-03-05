# Causal Inference with PyMC

PyMC supports structural causal models via `pm.do` and `pm.observe`.

## pm.do (Interventions)

Apply do-calculus interventions to set variables to fixed values, breaking incoming causal edges:

```python
with pm.Model() as causal_model:
    x = pm.Normal("x", 0, 1)
    y = pm.Normal("y", x, 1)
    z = pm.Normal("z", y, 1)

# Intervene: set x = 2 (breaks incoming edges to x)
with pm.do(causal_model, {"x": 2}) as intervention_model:
    idata = pm.sample_prior_predictive()
    # Samples from P(y, z | do(x=2))
```

## pm.observe (Conditioning)

Condition on observed values without breaking causal structure:

```python
# Condition: observe y = 1 (doesn't break causal structure)
with pm.observe(causal_model, {"y": 1}) as conditioned_model:
    idata = pm.sample(nuts_sampler="nutpie")
    # Samples from P(x, z | y=1)
```

## Combining do and observe

```python
# Intervention + observation for causal queries
with pm.do(causal_model, {"x": 2}) as m1:
    with pm.observe(m1, {"z": 0}) as m2:
        idata = pm.sample(nuts_sampler="nutpie")
        # P(y | do(x=2), z=0)
```

## Causal Effect Estimation

```python
# Average causal effect via do-calculus
with pm.do(causal_model, {"treatment": 1}) as treated:
    idata_treated = pm.sample_prior_predictive(draws=2000)

with pm.do(causal_model, {"treatment": 0}) as control:
    idata_control = pm.sample_prior_predictive(draws=2000)

# ATE = E[Y | do(T=1)] - E[Y | do(T=0)]
ate = (idata_treated.prior_predictive["outcome"].mean() -
       idata_control.prior_predictive["outcome"].mean())
```

## Key Considerations

- `pm.do` replaces the variable's distribution with a constant — all upstream causal paths are severed
- `pm.observe` adds observed data — the variable remains stochastic but conditioned on the value
- For backdoor adjustment, condition on confounders using `pm.observe` or include them in the model directly
- For front-door adjustment or instrumental variables, combine `pm.do` with appropriate model structure
