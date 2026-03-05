# Bayesian Modeling Workflow

Principled, iterative model building based on Gelman, Vehtari & McElreath (2025) and Gelman et al. (2020).

## The Workflow

1. **Start simple** — fit the simplest plausible model first
2. **Prior predictive check** — verify priors produce plausible data (see SKILL.md § Prior Predictive)
3. **Fit and diagnose** — check convergence before interpreting (see SKILL.md § Diagnostics)
4. **Posterior predictive check** — find specific, interpretable misfits (see SKILL.md § Posterior Predictive)
5. **Expand and compare** — add one piece of complexity, compare via LOO, repeat

## Start Simple

Write the simplest model that could address the question. Fit it. Understand it before adding complexity.

- **Grouped data**: start with complete pooling (single intercept), then no pooling, then partial pooling
- **Regression**: start with one or two key predictors, not all of them
- **Time series**: start with a simple trend or random walk before adding seasonal or autoregressive structure

Complex models are understood in relation to simpler ones. Each expansion reveals what the added complexity buys. Computational problems are easier to diagnose in simple models.

Save `results.nc` immediately after your first successful sampling run.

## Expand and Compare

Label each model explicitly (e.g., "Model 1: complete pooling", "Model 2: partial pooling") so the progression is clear.

Add ONE piece of complexity at a time. Fit the expanded model, diagnose, check posterior predictions, then compare:

```python
# Compute log-likelihood if using nutpie
pm.compute_log_likelihood(idata_simple, model=model_simple)
pm.compute_log_likelihood(idata_complex, model=model_complex)

comparison = az.compare({
    "simple": idata_simple,
    "complex": idata_complex,
})
print(comparison[["rank", "elpd_loo", "elpd_diff", "se_diff", "weight"]])
```

**Decision rule**: If two models have similar stacking weights, they are effectively equivalent.

Fit the expanded model even when you believe the simpler one is sufficient. The comparison itself is informative:
- If the expansion doesn't improve fit, you've demonstrated the simpler model is adequate
- If it does improve fit, you've identified important structure
- The *difference* between models tells you something about the data-generating process

## Reporting the Workflow

Report the sequence of models, not just the final one. The modeling journey IS the analysis.

Summarize with a model progression table:

```
| Model | Description | ELPD_LOO | elpd_diff | weight |
|-------|-------------|----------|-----------|--------|
| Model 2 | Partial pooling | -222.2 | 0.0 | 0.91 |
| Model 1 | Complete pooling | -234.5 | -12.3 | 0.09 |
```

Include: prior predictive findings, posterior predictive misfits that motivated each expansion, model comparison results, final parameter estimates with 94% HDIs, and conclusions about what the data support.

## Simulating Fake Data

For novel or complex model structures, simulate from your model with known parameter values before touching real data:

1. Choose plausible parameter values (informed by domain knowledge)
2. Simulate a dataset from the generative model
3. Fit the model to simulated data
4. Check: are the true parameters recovered within posterior intervals?

This catches specification bugs, non-identifiability, and prior-likelihood conflicts before real-data messiness obscures the picture. Most valuable for custom likelihoods, latent variable models, and models with complex indexing.

## Hierarchical Build-Up

When building hierarchical models iteratively:

1. **Complete pooling** — single intercept, ignoring group structure
2. **No pooling** — group-specific intercepts, no sharing — compare to pooled
3. **Partial pooling** — hierarchical intercepts — compare to both
4. **Varying slopes** — allow slopes to vary by group if the question warrants it
5. **Group-level predictors** — explain between-group variance

At each step, compare via LOO and check whether the added structure reduces between-group variance.

## Compressed Storage

For large InferenceData objects (many draws, large posterior predictive):

```python
# Compress with zlib (reduces file size 50-80%)
idata.to_netcdf(
    "results/model_v1.nc",
    engine="h5netcdf",
    encoding={var: {"zlib": True, "complevel": 4}
              for group in ["posterior", "posterior_predictive"]
              if hasattr(idata, group)
              for var in getattr(idata, group).data_vars}
)
```

## References

- Gelman, Vehtari, and McElreath (2025). "Statistical Workflow." *Philosophical Transactions of the Royal Society A.*
- Gelman, Vehtari, Simpson, et al. (2020). "Bayesian Workflow." arXiv:2011.01808.
- Gabry, Simpson, Vehtari, Betancourt, and Gelman (2019). "Visualization in Bayesian workflow." *JRSS A.*
