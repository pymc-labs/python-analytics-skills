---
name: pymc-modeling
description: >
  Bayesian statistical modeling with PyMC v5+. Use when building probabilistic models,
  specifying priors, running MCMC inference, diagnosing convergence, or comparing models.
  Covers PyMC, ArviZ, pymc-bart, pymc-extras, nutpie, and JAX/NumPyro backends. Triggers
  on tasks involving: Bayesian inference, posterior sampling, hierarchical/multilevel models,
  GLMs, time series, Gaussian processes, BART, mixture models, prior/posterior predictive
  checks, MCMC diagnostics, LOO-CV, WAIC, model comparison, or causal inference with do/observe.
---

# PyMC Modeling

Bayesian modeling workflow for PyMC v5+ with modern API patterns.

Claude understands the fundamentals of Bayesian inference—priors, likelihoods, posterior distributions, and Bayes' theorem. It knows MCMC is the standard approach for posterior sampling and can explain what a hierarchical model is. But getting from these concepts to a correctly-specified, well-diagnosed, and efficiently-sampled PyMC model requires domain-specific knowledge that changes over time.

This skill bridges that gap. It encodes modern best practices like using nutpie as the default sampler because it runs two to five times faster than the default NUTS implementation, choosing non-centered parameterization for hierarchical models to avoid pathological geometry, and reaching for HSGP instead of exact Gaussian processes for any dataset larger than a few hundred points. It covers the common pitfalls you will actually hit—why you are getting divergences and how to fix them, the specific error messages that indicate a shape mismatch or initialization failure, and when the centered parameterization actually performs better despite the folklore. It also details the correct API usage: how to structure coords and dims for readable InferenceData, why nutpie silently ignores log_likelihood requests and what to do about it, and the proper workflow for saving results to NetCDF.

Without this skill, Claude might suggest outdated defaults like the slow default NUTS sampler, miss critical diagnostics such as ESS and r_hat checks, or recommend inefficient parameterizations that lead to divergences. With it, you get concise, battle-tested patterns that actually work in practice.

**Modeling strategy**: Build models iteratively — start simple, check prior
predictions, fit and diagnose, check posterior predictions, expand one piece at
a time. See [references/workflow.md](references/workflow.md) for the full workflow.

**Notebook preference**: Use marimo for interactive modeling unless the project already uses Jupyter.

## Model Specification

### Basic Structure

```python
import pymc as pm
import arviz as az

with pm.Model(coords=coords) as model:
    # Data containers (for out-of-sample prediction)
    x = pm.Data("x", x_obs, dims="obs")

    # Priors
    beta = pm.Normal("beta", mu=0, sigma=1, dims="features")
    sigma = pm.HalfNormal("sigma", sigma=1)

    # Likelihood
    mu = pm.math.dot(x, beta)
    y = pm.Normal("y", mu=mu, sigma=sigma, observed=y_obs, dims="obs")

    # Inference
    idata = pm.sample(nuts_sampler="nutpie", random_seed=42)
```

### Coords and Dims

Use coords/dims for interpretable InferenceData when model has meaningful structure:

```python
coords = {
    "obs": np.arange(n_obs),
    "features": ["intercept", "age", "income"],
    "group": group_labels,
}
```

Skip for simple models where overhead exceeds benefit.

### Parameterization

Prefer non-centered parameterization for hierarchical models with weak data:

```python
# Non-centered (better for divergences)
offset = pm.Normal("offset", 0, 1, dims="group")
alpha = mu_alpha + sigma_alpha * offset

# Centered (better with strong data)
alpha = pm.Normal("alpha", mu_alpha, sigma_alpha, dims="group")
```

## Inference

### Default Sampling (nutpie preferred)

Prefer nutpie for sampling — it is Rust-based and typically 2-5x faster than PyMC's default NUTS. It supports most standard PyMC models including time series, GPs, mixtures, and custom likelihoods:

```python
with model:
    idata = pm.sample(
        draws=1000, tune=1000, chains=4,
        nuts_sampler="nutpie",
        random_seed=42,
    )
idata.to_netcdf("results.nc")  # Save immediately after sampling
```

**Important**: nutpie does not store log_likelihood automatically (it silently ignores `idata_kwargs={"log_likelihood": True}`). If you need LOO-CV or model comparison, compute it after sampling:

```python
pm.compute_log_likelihood(idata, model=model)
```

### When to Use PyMC's Default NUTS Instead

**Model correctness takes precedence over sampler speed.** nutpie's numba backend cannot handle models with discrete parameters or certain transforms (e.g., the `ordered` transform used with `OrderedLogistic`/`OrderedProbit`). If your model uses discrete variables or these transforms, omit `nuts_sampler="nutpie"` and use PyMC's default NUTS:

```python
# Models with discrete params or ordered transforms: use default NUTS
idata = pm.sample(draws=1000, tune=1000, chains=4, random_seed=42)
```

Do not work around sampler incompatibilities by changing the model specification. Use the correct statistical model and let the sampler adapt.

### If nutpie Is Not Installed

Use `nuts_sampler="nutpie"` or `nuts_sampler="numpyro"` when the model supports it. If neither is installed, install before sampling:

```python
import subprocess, sys
try:
    import nutpie
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "nutpie"])
```

### Alternative MCMC Backends

See [references/inference.md](references/inference.md) for:
- **NumPyro/JAX**: GPU acceleration, vectorized chains

### Approximate Inference

For fast (but inexact) posterior approximations:
- **ADVI/DADVI**: Variational inference with Gaussian approximation
- **Pathfinder**: Quasi-Newton optimization for initialization or screening

## Diagnostics and ArviZ Workflow

**Minimum workflow checklist** — every model script should include:
1. Prior predictive check (`pm.sample_prior_predictive`)
2. Save results immediately after sampling (`idata.to_netcdf(...)`)
3. Divergence count + r_hat + ESS check
4. Posterior predictive check (`pm.sample_posterior_predictive`)

Follow this systematic workflow after every sampling run:

### Phase 1: Immediate Checks (Required)

```python
# 1. Check for divergences (must be 0 or near 0)
n_div = idata.sample_stats["diverging"].sum().item()
print(f"Divergences: {n_div}")

# 2. Summary with convergence diagnostics
summary = az.summary(idata, var_names=["~offset"])  # exclude auxiliary
print(summary[["mean", "sd", "hdi_3%", "hdi_97%", "ess_bulk", "ess_tail", "r_hat"]])

# 3. Visual convergence check
az.plot_trace(idata, compact=True)
az.plot_rank(idata, var_names=["beta", "sigma"])
```

**Pass criteria** (all must pass before proceeding):
- Zero divergences (or < 0.1% and randomly scattered)
- `r_hat < 1.01` for all parameters
- `ess_bulk > 400` and `ess_tail > 400`
- Trace plots show good mixing (overlapping densities, fuzzy caterpillar)

### Phase 2: Deep Convergence (If Phase 1 marginal)

```python
# ESS evolution (should grow linearly)
az.plot_ess(idata, kind="evolution")

# Energy diagnostic (HMC health)
az.plot_energy(idata)

# Autocorrelation (should decay rapidly)
az.plot_autocorr(idata, var_names=["beta"])
```

### Phase 3: Model Criticism (Required)

```python
# Generate posterior predictive
with model:
    pm.sample_posterior_predictive(idata, extend_inferencedata=True)

# Does the model capture the data?
az.plot_ppc(idata, kind="cumulative")

# Calibration check
az.plot_loo_pit(idata, y="y")
```

**Critical rule**: Never interpret parameters until Phases 1-3 pass.

### Phase 4: Parameter Interpretation

```python
# Posterior summaries
az.plot_posterior(idata, var_names=["beta"], ref_val=0)

# Forest plots for hierarchical parameters
az.plot_forest(idata, var_names=["alpha"], combined=True)

# Parameter correlations (identify non-identifiability)
az.plot_pair(idata, var_names=["alpha", "beta", "sigma"])
```

See [references/arviz.md](references/arviz.md) for comprehensive ArviZ usage.
See [references/diagnostics.md](references/diagnostics.md) for troubleshooting.

## Prior and Posterior Predictive Checks

### Prior Predictive (Before Fitting)

Always check prior implications before fitting:

```python
with model:
    prior_pred = pm.sample_prior_predictive(draws=500)

# Do prior predictions span reasonable outcome range?
az.plot_ppc(prior_pred, group="prior", kind="cumulative")

# Numerical sanity check
prior_y = prior_pred.prior_predictive["y"].values.flatten()
print(f"Prior predictive range: [{prior_y.min():.1f}, {prior_y.max():.1f}]")
```

**Warning signs**: Prior predictive covers implausible values (negative counts, probabilities > 1) or is extremely wide/narrow.

**Rule**: Run prior predictive checks before `pm.sample()` on any new model.
If the range is unreasonable, adjust priors and re-check before proceeding.

### Posterior Predictive (After Fitting)

```python
with model:
    pm.sample_posterior_predictive(idata, extend_inferencedata=True)

# Density comparison
az.plot_ppc(idata, kind="kde")

# Cumulative (better for systematic deviations)
az.plot_ppc(idata, kind="cumulative")

# Calibration diagnostic
az.plot_loo_pit(idata, y="y")
```

**Interpretation**: Observed data (dark line) should fall within posterior predictive distribution (light lines). See [references/arviz.md](references/arviz.md) for detailed interpretation.

## Model Debugging

### Inspecting Model Structure

```python
# Print model summary (variables, shapes, distributions)
print(model)

# Visualize model as directed graph
pm.model_to_graphviz(model)
```

### Checking for Specification Errors

Before sampling, validate the model:

```python
# Debug model: checks for common issues
model.debug()

# Check initial point log-probabilities
# Identifies which variables have invalid starting values
model.point_logps()
```

### Common Issues

| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| `ValueError: Shape mismatch` | Parameter vs observation dimensions | Use index vectors: `alpha[group_idx]` |
| `Initial evaluation failed` | Data outside distribution support | Check bounds; use `init="adapt_diag"` |
| `Mass matrix contains zeros` | Unscaled predictors or flat priors | Standardize features; use weakly informative priors |
| High divergence count | Funnel geometry | Non-centered parameterization |
| `NaN` in log-probability | Invalid parameter combinations | Check parameter constraints, add bounds |
| `-inf` log-probability | Observations outside likelihood support | Verify data matches distribution domain |
| Slow discrete sampling | NUTS incompatible with discrete | Marginalize discrete variables |

See [references/troubleshooting.md](references/troubleshooting.md) for comprehensive problem-solution guide.

### Debugging Divergences

```python
# Identify where divergences occur in parameter space
az.plot_pair(idata, var_names=["alpha", "beta", "sigma"], divergences=True)

# Check if divergences cluster in specific regions
# Clustering suggests parameterization or prior issues
```

### Profiling Slow Models

```python
# Time individual operations in the log-probability computation
profile = model.profile(model.logp())
profile.summary()

# Identify bottlenecks in gradient computation
import pytensor
grad_profile = model.profile(pytensor.grad(model.logp(), model.continuous_value_vars))
grad_profile.summary()
```

See [references/gotchas.md](references/gotchas.md) for additional troubleshooting.

## Model Comparison

### LOO-CV (Preferred)

```python
# Compute LOO with pointwise diagnostics
loo = az.loo(idata, pointwise=True)
print(f"ELPD: {loo.elpd_loo:.1f} ± {loo.se:.1f}")

# Check Pareto k values (must be < 0.7 for reliable LOO)
print(f"Bad k (>0.7): {(loo.pareto_k > 0.7).sum().item()}")
az.plot_khat(idata)
```

### Comparing Models

```python
# If using nutpie, compute log-likelihood first (nutpie doesn't store it automatically)
pm.compute_log_likelihood(idata_a, model=model_a)
pm.compute_log_likelihood(idata_b, model=model_b)

comparison = az.compare({
    "model_a": idata_a,
    "model_b": idata_b,
}, ic="loo")

print(comparison[["rank", "elpd_loo", "elpd_diff", "weight"]])
az.plot_compare(comparison)
```

**Decision rule**: If two models have similar stacking weights, they are effectively equivalent.

See [references/arviz.md](references/arviz.md) for detailed model comparison workflow.

### Iterative Model Building

Build complexity incrementally: fit the simplest plausible model first, diagnose
it, check posterior predictions, then add ONE piece of complexity at a time.
Compare each expansion via LOO. If stacking weights are similar, the models are effectively equivalent.
See [references/workflow.md](references/workflow.md) for the full iterative workflow.

## Saving and Loading Results

### InferenceData Persistence

Save sampling results for later analysis or sharing:

```python
# Save to NetCDF (recommended format)
idata.to_netcdf("results/model_v1.nc")

# Load
idata = az.from_netcdf("results/model_v1.nc")
```

### Compressed Storage

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

### What Gets Saved

InferenceData preserves the full Bayesian workflow:
- `posterior`: Parameter samples from MCMC
- `prior`, `prior_predictive`: Prior samples (if generated)
- `posterior_predictive`: Predictions (if generated)
- `observed_data`, `constant_data`: Data used in fitting
- `sample_stats`: Diagnostics (divergences, tree depth, energy)
- `log_likelihood`: Pointwise log-likelihood (for LOO-CV)
- All coordinates and dimensions

### Workflow Pattern

```python
# Save IMMEDIATELY after sampling — late crashes (post-MCMC) destroy valid results
with model:
    idata = pm.sample(nuts_sampler="nutpie")
idata.to_netcdf("results.nc")  # Save before any post-processing!

# Then do posterior predictive, diagnostics, etc.
with model:
    pm.sample_posterior_predictive(idata, extend_inferencedata=True)
idata.to_netcdf("results.nc")  # Update with posterior predictive

# Resume later
idata = az.from_netcdf("results.nc")
az.plot_ppc(idata)  # Continue analysis
```

## Prior Selection

See [references/priors.md](references/priors.md) for:
- Weakly informative defaults by distribution type
- Prior predictive checking workflow
- Domain-specific recommendations

## Common Patterns

### Hierarchical/Multilevel

```python
with pm.Model(coords={"group": groups, "obs": obs_idx}) as hierarchical:
    # Hyperpriors
    mu_alpha = pm.Normal("mu_alpha", 0, 1)
    sigma_alpha = pm.HalfNormal("sigma_alpha", 1)

    # Group-level (non-centered)
    alpha_offset = pm.Normal("alpha_offset", 0, 1, dims="group")
    alpha = pm.Deterministic("alpha", mu_alpha + sigma_alpha * alpha_offset, dims="group")

    # Likelihood
    y = pm.Normal("y", alpha[group_idx], sigma, observed=y_obs, dims="obs")
```

### GLMs

```python
# Logistic regression
with pm.Model() as logistic:
    alpha = pm.Normal("alpha", 0, 2.5)  # intercept
    beta = pm.Normal("beta", 0, 2.5, dims="features")
    
    # Logit link
    logit_p = alpha + pm.math.dot(X, beta)
    p = pm.math.sigmoid(logit_p)
    
    y = pm.Bernoulli("y", p=p, observed=y_obs)

# Poisson regression
with pm.Model() as poisson:
    beta = pm.Normal("beta", 0, 1, dims="features")
    mu = pm.math.exp(pm.math.dot(X, beta))
    y = pm.Poisson("y", mu=mu, observed=y_obs)
```

### Gaussian Processes

**Always prefer HSGP** for GP problems with 1-3D inputs. It's O(nm) instead of O(n³), and even at n=200 exact GP (`pm.gp.Marginal`) is prohibitively slow for MCMC:

```python
with pm.Model() as gp_model:
    # Hyperparameters
    ell = pm.InverseGamma("ell", alpha=5, beta=5)
    eta = pm.HalfNormal("eta", sigma=2)
    sigma = pm.HalfNormal("sigma", sigma=0.5)

    # Covariance function (Matern52 recommended)
    cov = eta**2 * pm.gp.cov.Matern52(1, ls=ell)

    # HSGP approximation
    gp = pm.gp.HSGP(m=[20], c=1.5, cov_func=cov)
    f = gp.prior("f", X=X[:, None])  # X must be 2D

    # Likelihood
    y = pm.Normal("y", mu=f, sigma=sigma, observed=y_obs)
```

For periodic patterns, use `pm.gp.HSGPPeriodic`. Only use `pm.gp.Marginal` or `pm.gp.Latent` for very small datasets (n < ~50) where exact inference is specifically needed.

See [references/gp.md](references/gp.md) for:
- **HSGP parameter selection** (choosing m and c, automatic heuristics)
- **HSGPPeriodic** for seasonal/cyclic patterns
- Approximation quality diagnostics
- Covariance functions and priors
- Common patterns (trend + seasonality, classification, heteroscedastic)

### Time Series

```python
with pm.Model(coords={"time": range(T)}) as ar_model:
    rho = pm.Uniform("rho", -1, 1)
    sigma = pm.HalfNormal("sigma", sigma=1)

    y = pm.AR("y", rho=[rho], sigma=sigma, constant=True,
              observed=y_obs, dims="time")
```

See [references/timeseries.md](references/timeseries.md) for:
- Autoregressive models (AR, ARMA)
- Random walk and local level models
- Structural time series (trend + seasonality)
- State space models
- GPs for time series
- Handling multiple seasonalities
- Forecasting patterns

### BART (Bayesian Additive Regression Trees)

```python
import pymc_bart as pmb

with pm.Model() as bart_model:
    mu = pmb.BART("mu", X=X, Y=y, m=50)
    sigma = pm.HalfNormal("sigma", 1)
    y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y)
```

See [references/bart.md](references/bart.md) for:
- Regression and classification
- Variable importance and partial dependence
- Combining BART with parametric components
- Configuration (number of trees, depth priors)

### Mixture Models

```python
import numpy as np

coords = {"component": range(K)}

with pm.Model(coords=coords) as gmm:
    # Mixture weights
    w = pm.Dirichlet("w", a=np.ones(K), dims="component")

    # Component parameters (with ordering to avoid label switching)
    mu = pm.Normal("mu", mu=0, sigma=10, dims="component",
                   transform=pm.distributions.transforms.ordered,
                   initval=np.linspace(y_obs.min(), y_obs.max(), K))
    sigma = pm.HalfNormal("sigma", sigma=2, dims="component")

    # Mixture likelihood
    y = pm.NormalMixture("y", w=w, mu=mu, sigma=sigma, observed=y_obs)
```

**Important**: Mixture models often need `target_accept=0.9` or higher to avoid divergences from the multimodal geometry. Always provide `initval` on ordered means — without it, components can start overlapping and the sampler struggles to separate them.

See [references/mixtures.md](references/mixtures.md) for:
- Finite mixture models and mixture of regressions
- Label switching problem and solutions (ordering constraints, relabeling)
- Marginalized mixtures (pymc-extras)
- Diagnostics for mixture models

### Sparse Regression / Horseshoe

Use the regularized (Finnish) horseshoe prior for high-dimensional regression with expected sparsity:

```python
import pytensor.tensor as pt

with pm.Model(coords={"features": feature_names}) as sparse_model:
    # Regularized horseshoe (Piironen & Vehtari, 2017)
    tau = pm.HalfStudentT("tau", nu=2, sigma=1)  # global shrinkage
    lam = pm.HalfStudentT("lam", nu=5, dims="features")  # local shrinkage
    c2 = pm.InverseGamma("c2", alpha=1, beta=1)  # slab variance
    z = pm.Normal("z", 0, 1, dims="features")

    # Regularized shrinkage factor
    lam_tilde = pt.sqrt(c2 / (c2 + tau**2 * lam**2))
    beta = pm.Deterministic("beta", z * tau * lam * lam_tilde, dims="features")

    mu = pm.math.dot(X, beta)
    y = pm.Normal("y", mu=mu, sigma=sigma, observed=y_obs)

    idata = pm.sample(nuts_sampler="nutpie", target_accept=0.95)
```

**Important**: Horseshoe priors create double-funnel geometry. Use `target_accept=0.95` or higher to avoid divergences.

See [references/priors.md](references/priors.md) for Laplace, R2D2, and spike-and-slab alternatives.

### Specialized Likelihoods

```python
# Zero-Inflated Poisson (excess zeros)
with pm.Model() as zip_model:
    psi = pm.Beta("psi", alpha=2, beta=2)  # P(structural zero)
    mu = pm.Exponential("mu", lam=1)
    y = pm.ZeroInflatedPoisson("y", psi=psi, mu=mu, observed=y_obs)

# Censored data (e.g., right-censored survival)
with pm.Model() as censored_model:
    mu = pm.Normal("mu", mu=0, sigma=10)
    sigma = pm.HalfNormal("sigma", sigma=5)
    y = pm.Censored("y", dist=pm.Normal.dist(mu=mu, sigma=sigma),
                    lower=None, upper=censoring_time, observed=y_obs)

# Ordinal regression
with pm.Model() as ordinal:
    beta = pm.Normal("beta", mu=0, sigma=2, dims="features")
    cutpoints = pm.Normal("cutpoints", mu=0, sigma=2,
                          transform=pm.distributions.transforms.ordered,
                          shape=n_categories - 1)
    y = pm.OrderedLogistic("y", eta=pm.math.dot(X, beta),
                           cutpoints=cutpoints, observed=y_obs)
```

**Note**: Don't use the same name for a variable and a dimension. For example, if you have a dimension called `"cutpoints"`, don't also name a variable `"cutpoints"` — this causes shape errors.

See [references/specialized_likelihoods.md](references/specialized_likelihoods.md) for:
- Zero-inflated models (Poisson, Negative Binomial, Binomial)
- Hurdle models for count data
- Censored and truncated data
- Ordinal regression
- Robust regression with Student-t likelihood

## Common Pitfalls

See [references/gotchas.md](references/gotchas.md) for:
- Centered vs non-centered parameterization
- Priors on scale parameters
- Label switching in mixtures
- Performance issues (GPs, large Deterministics)
- Python conditionals and hard clipping
- Redundant intercepts in hierarchical models

See [references/troubleshooting.md](references/troubleshooting.md) for comprehensive problem-solution guide covering:
- Shape and dimension errors
- Initialization failures
- Mass matrix and numerical issues
- Discrete variable challenges
- Data container and prediction issues

## Causal Inference Operations

### pm.do (Interventions)

Apply do-calculus interventions to set variables to fixed values:

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

### pm.observe (Conditioning)

Condition on observed values without intervention:

```python
# Condition: observe y = 1 (doesn't break causal structure)
with pm.observe(causal_model, {"y": 1}) as conditioned_model:
    idata = pm.sample(nuts_sampler="nutpie")
    # Samples from P(x, z | y=1)
```

### Combining do and observe

```python
# Intervention + observation for causal queries
with pm.do(causal_model, {"x": 2}) as m1:
    with pm.observe(m1, {"z": 0}) as m2:
        idata = pm.sample(nuts_sampler="nutpie")
        # P(y | do(x=2), z=0)
```

## pymc-extras

For specialized models and inference:

```python
import pymc_extras as pmx

# Marginalize discrete parameters from a model
model = pmx.marginalize(model, ["discrete_var"])

# R2D2 prior for regression (requires output_sigma and input_sigma)
residual_sigma, beta = pmx.R2D2M2CP(
    "r2d2",
    output_sigma=y.std(),
    input_sigma=X.std(axis=0),
    dims="features",
    r2=0.5,
)

# Laplace approximation for fast inference
idata = pmx.fit_laplace(model)
```

## Custom Distributions and Model Components

For extending PyMC beyond built-in distributions:

```python
import pymc as pm
import pytensor.tensor as pt

# Custom likelihood via DensityDist
def custom_logp(value, mu, sigma):
    return pm.logp(pm.Normal.dist(mu=mu, sigma=sigma), value)

with pm.Model() as model:
    mu = pm.Normal("mu", 0, 1)
    y = pm.DensityDist("y", mu, 1.0, logp=custom_logp, observed=y_obs)

# Soft constraints via Potential
with pm.Model() as model:
    alpha = pm.Normal("alpha", 0, 1, dims="group")
    pm.Potential("sum_to_zero", -100 * pt.sqr(alpha.sum()))
```

See [references/custom_models.md](references/custom_models.md) for:
- `pm.DensityDist` for custom likelihoods
- `pm.Potential` for soft constraints and Jacobian adjustments
- `pm.Simulator` for simulation-based inference (ABC)
- `pm.CustomDist` for custom prior distributions
