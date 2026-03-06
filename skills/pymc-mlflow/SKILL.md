---
name: pymc-mlflow
description: >
  MLflow integration for PyMC models. Use when tracking experiments, logging artifacts
  (InferenceData, diagnostics, posterior predictive checks), comparing model runs, or
  preparing models for deployment. Covers autologging with pymc_marketing.mlflow,
  metadata strategies (tags for test/production runs, sample counts, model complexity),
  artifact management, and domain-specific extensions (MMM, CLV). Emphasizes consistent
  naming and metric tracking to enable fast iteration and effective comparison.
---

# PyMC + MLflow Integration

MLflow experiment tracking and artifact management for PyMC Bayesian workflows.

## Purpose and Scope

This skill bridges PyMC modeling and production deployment through MLflow's tracking infrastructure. It focuses on the intersection of PyMC and MLflow—not general MLflow best practices (see separate MLflow skills) or detailed Bayesian modeling.

**For modeling guidance** (prior selection, model specification, parameterization, convergence diagnostics, posterior predictive checks), see the `pymc-modeling` skill. This skill assumes you know how to build and diagnose PyMC models and focuses exclusively on MLflow integration patterns.

**What this covers**:
- Autologging with `pymc_marketing.mlflow.autolog()`
- Metadata strategies: tags, parameters, and metrics for filtering/comparison
- Artifact management: InferenceData, plots, model serialization
- Domain-specific extensions: MMM and CLV workflows

**Why MLflow for PyMC**:
- **Fast iteration**: Tag test vs production runs, mock fits vs full sampling
- **Comparison**: Standardize metrics across experiments for apples-to-apples evaluation
- **Artifact management**: Consistent naming and storage for InferenceData, plots, diagnostics
- **Deployment bridge**: Serialization-ready artifacts for production systems

**Key insight**: Consistent logging conventions (tags, metrics, artifact names) enable effective experiment tracking whether you're running 2 models or 50, locally or in the cloud.

## Setup and Configuration

### Tracking URI

**Local SQLite** (for solo work, 2-50 models):
```python
import mlflow

# Simple local tracking with SQLite database
mlflow.set_tracking_uri("sqlite:///mlruns.db")
```

Benefits: Simple file-based tracking, no infrastructure needed, easy version control. The `mlruns.db` file stores all experiment metadata and can be committed to version control (though artifacts should typically be `.gitignore`d).

**Remote tracking** (Databricks, Azure ML, dedicated MLflow server):
```python
mlflow.set_tracking_uri("databricks")  # or Azure ML URI
```

Benefits: Team collaboration, centralized artifact storage, production-ready infrastructure.

### Experiment Organization

Separate test iterations from production runs:
```python
# For fast iteration and testing
mlflow.set_experiment("PyMC Models - Test")

# For production-ready models
mlflow.set_experiment("PyMC Models - Production")
```

**Pattern**: Use experiment names to organize by project, model family, or deployment stage.

### Artifact Naming Conventions

Use consistent names across all runs to enable side-by-side comparison:

| Artifact | Standard Name | Purpose |
|----------|---------------|---------|
| InferenceData | `idata.nc` | Full posterior, diagnostics, metadata |
| Posterior predictive check | `posterior_predictive_check.png` | Model calibration visual |
| Trace plot | `trace_plot.png` | Convergence diagnostic |
| Model graph | `model_graph.pdf` | Model structure visualization |
| ArviZ summary | `arviz_summary.csv` | Parameter estimates and diagnostics |

**Why consistency matters**: MLflow UI and programmatic searches work best when artifacts have predictable names across runs.

## Core Logging Patterns

### Autolog (Quickstart)

`pymc_marketing.mlflow.autolog()` patches PyMC's sampling functions to automatically log:

```python
import mlflow
import pymc as pm
import pymc_marketing.mlflow

# Enable autologging
pymc_marketing.mlflow.autolog()

mlflow.set_experiment("PyMC Experiment")

# Define model outside run context to reduce indentation
with pm.Model() as model:
    mu = pm.Normal("mu", mu=0, sigma=1)
    obs = pm.Normal("obs", mu=mu, sigma=1, observed=[1, 2, 3])

with mlflow.start_run(log_system_metrics=True):
    # Tag for filtering
    mlflow.set_tag("run_type", "test")
    
    # Log random seed for reproducibility
    RANDOM_SEED = 42
    mlflow.log_param("random_seed", RANDOM_SEED)
    
    # Use model= parameter to avoid nested context
    idata = pm.sample(model=model, nuts_sampler="nutpie", draws=1000, chains=4, random_seed=RANDOM_SEED)
```

**What gets logged automatically by `pymc_marketing.mlflow.autolog()`**:

1. **`pymc_marketing.mlflow.log_versions()`**: Package versions (PyMC, ArviZ, PyMC-Marketing)
2. **`pymc_marketing.mlflow.log_model_derived_info()`**: 
   - Parameter types (continuous, discrete, deterministic)
   - Coordinates and dimensions
   - Likelihood family
   - Graph complexity (number of nodes, distributions)
3. **`pymc_marketing.mlflow.log_sample_diagnostics()`**: 
   - Divergence count
   - ESS bulk and tail (total across chains)
   - r_hat statistics
   - Tree depth
4. **`pymc_marketing.mlflow.log_arviz_summary()`**: Parameter summary table (mean, sd, HDI, ESS, r_hat)
5. **`pymc_marketing.mlflow.log_metadata()`**: Data shapes, coordinate metadata
6. **`pymc_marketing.mlflow.log_inference_data()`**: Full InferenceData as `idata.nc` artifact

**Note**: These are all functions from the `pymc_marketing.mlflow` module. You can also call them manually if not using autolog (see Manual Logging section).

**When to use autolog**: Default starting point for all PyMC workflows. Captures essential diagnostics and metadata with zero manual effort.

**When to extend autolog**: Add custom metrics, domain-specific plots, or additional artifacts (see Manual Logging section).

**Important**: This autologging functionality is specific to the `pymc_marketing` package. For vanilla PyMC projects without PyMC-Marketing, you'll need to use manual logging (see Manual Logging section) or implement similar helpers yourself.

### Essential Metadata and Tags

Standardize tags and parameters across runs to enable effective filtering and comparison.

**Key principle**: Log metadata **before** sampling when possible. This allows you to filter and search runs even if sampling fails or is still in progress—particularly valuable for long-running or expensive models. This is a core design principle of the `pymc_marketing.mlflow` module.

```python
import mlflow
import pymc as pm
import pymc.testing  # For mock_sample

# Define model outside MLflow run context
with pm.Model(coords={"group": groups, "obs": obs_idx}) as model:
    # ... model specification ...
    pass

with mlflow.start_run():
    # ============================================
    # RUN CLASSIFICATION (log before sampling)
    # ============================================
    # Filter production-ready models vs test iterations
    mlflow.set_tag("run_type", "test")  # or "production"
    
    # Mark fast iterations vs full sampling
    mock_fit = True  # Set based on your iteration strategy
    mlflow.set_tag("mock_fit", str(mock_fit))
    
    # ============================================
    # SAMPLING CONFIGURATION (log before sampling)
    # ============================================
    draws = 1000
    chains = 4
    sampler = "nutpie"
    random_seed = 42
    
    mlflow.log_param("sampler", sampler)
    mlflow.log_param("draws", draws)
    mlflow.log_param("chains", chains)
    mlflow.log_param("total_samples", draws * chains)  # Critical for filtering
    mlflow.log_param("random_seed", random_seed)  # For reproducibility
    
    # ============================================
    # MODEL CONFIGURATION (log before sampling)
    # ============================================
    # Log likelihood family
    mlflow.log_param("likelihood", "Normal")  # or "Poisson", "Bernoulli", etc.
    
    # Log model complexity
    mlflow.log_param("n_parameters", len(model.free_RVs))
    mlflow.log_param("n_distributions", len(model.basic_RVs))
    
    # Log coordinate structure (if using coords/dims)
    if model.coords:
        mlflow.log_param("coords", list(model.coords.keys()))
    
    # ============================================
    # SPECIAL FLAGS (log before sampling)
    # ============================================
    # Tag prior sensitivity experiments
    mlflow.set_tag("prior_sensitivity_test", "false")
    
    # ============================================
    # SAMPLING (now execute after metadata is logged)
    # ============================================
    if mock_fit:
        # Fast iteration: use mock_sample to skip MCMC
        pm.sample = pymc.testing.mock_sample
    
    # Full sampling with model= parameter to avoid nested context
    idata = pm.sample(model=model, nuts_sampler=sampler, draws=draws, chains=chains, random_seed=random_seed)
```

**Why these tags matter**:

| Tag/Parameter | Use Case | Example Filter |
|---------------|----------|----------------|
| `run_type` | Separate test from production | `tags.run_type = "production"` |
| `mock_fit` | Exclude fast iterations | `tags.mock_fit = "false"` |
| `total_samples` | Find well-sampled models | `params.total_samples >= 2000` |
| `likelihood` | Compare model families | `params.likelihood = "Poisson"` |
| `sampler` | Compare sampler performance | `params.sampler IN ("nutpie", "numpyro")` |
| `n_parameters` | Track model complexity | Sort by `params.n_parameters` |
| `prior_sensitivity_test` | Isolate prior exploration | `tags.prior_sensitivity_test = "true"` |

**Common filtering patterns**:
```python
# Production models with sufficient sampling
filter_string = 'tags.run_type = "production" AND tags.mock_fit = "false" AND params.total_samples >= 2000'

# Compare specific likelihood families
filter_string = 'params.likelihood = "NegativeBinomial" AND tags.run_type = "production"'

# Find models by complexity threshold
filter_string = 'params.n_parameters <= 20'
```

### Fast Iteration with Mock Sampling

For rapid prototyping and testing, use PyMC's `mock_sample` to skip actual MCMC and return synthetic draws instantly:

```python
import mlflow
import pymc as pm
import pymc.testing
import pymc_marketing.mlflow

pymc_marketing.mlflow.autolog()

mock_fit = True  # Toggle based on your workflow

# Define model before MLflow run
with pm.Model() as model:
    mu = pm.Normal("mu", mu=0, sigma=1)
    obs = pm.Normal("obs", mu=mu, sigma=1, observed=data)

with mlflow.start_run():
    mlflow.set_tag("mock_fit", str(mock_fit))
    mlflow.log_param("draws", 1000)
    mlflow.log_param("chains", 4)
    mlflow.log_param("random_seed", 42)
    
    if mock_fit:
        # Replace pm.sample with mock sampler for instant results
        # Returns synthetic InferenceData with correct structure but random values
        pm.sample = pymc.testing.mock_sample
    
    # Same call regardless of mock_fit—clean interface
    idata = pm.sample(model=model, draws=1000, chains=4, random_seed=42)
```

**What `mock_sample` does**:
- Returns InferenceData with the correct structure (posterior, sample_stats groups)
- Fills arrays with random values matching expected shapes
- Completes instantly (no actual MCMC)
- Useful for testing data pipelines, MLflow logging, and visualization code

**Use cases for mock sampling**:
- **Model structure debugging**: Verify model builds and coord/dims are correct without waiting for MCMC
- **Experiment setup**: Test entire MLflow logging pipeline before committing to full runs
- **CI/CD**: Fast model validation in automated tests
- **Iteration**: Quickly iterate on model specification, plotting code, or post-processing

**Important**: 
- Always tag `mock_fit = "true"` so you can filter these runs out when comparing production models
- Mock samples have no statistical validity—only use for testing infrastructure
- See [PyMC testing documentation](https://www.pymc.io/projects/docs/en/stable/api/testing.html) and `pymc-testing` skill for more testing utilities

### System Metrics Monitoring

MLflow can automatically log system resource usage (CPU, GPU, memory, network, disk) during sampling—particularly useful when comparing different samplers (nutpie on CPU vs numpyro on GPU) or monitoring long-running models.

**Installation requirements**:
```bash
# Required for system metrics
pip install psutil

# Optional: for NVIDIA GPU metrics
pip install nvidia-ml-py

# Optional: for AMD/HIP GPU metrics
pip install pyrsmi
```

**Enable for a specific run** (recommended for selective monitoring):
```python
import mlflow
import pymc as pm

with pm.Model() as model:
    mu = pm.Normal("mu", 0, 1)
    obs = pm.Normal("obs", mu=mu, sigma=1, observed=data)

# Enable system metrics for this run only
with mlflow.start_run(log_system_metrics=True):
    mlflow.set_tag("sampler_backend", "nutpie")
    idata = pm.sample(model=model, nuts_sampler="nutpie", draws=2000, chains=4, random_seed=42)
```

**Enable globally** (for all runs in session):
```python
import mlflow

# Option 1: Environment variable (set before starting Python)
# export MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING=true

# Option 2: Programmatic
mlflow.enable_system_metrics_logging()

# Now all runs will log system metrics
with mlflow.start_run():
    idata = pm.sample(model=model, draws=2000, chains=4, random_seed=42)
```

**System metrics logged by default**:
- `system/cpu_utilization_percentage`
- `system/system_memory_usage_megabytes` and `system/system_memory_usage_percentage`
- `system/gpu_utilization_percentage`, `system/gpu_memory_usage_megabytes` (if GPU available)
- `system/gpu_power_usage_watts` and `system/gpu_power_usage_percentage`
- `system/network_receive_megabytes` and `system/network_transmit_megabytes`
- `system/disk_usage_megabytes` and `system/disk_available_megabytes`

**Customize sampling frequency**:
```python
# Sample every 5 seconds, aggregate 2 samples before logging (10s window)
mlflow.set_system_metrics_sampling_interval(5)
mlflow.set_system_metrics_samples_before_logging(2)

with mlflow.start_run(log_system_metrics=True):
    idata = pm.sample(model=model, nuts_sampler="nutpie", draws=2000, chains=4, random_seed=42)
```

**When to use**:
- **Comparing sampler backends**: Track GPU utilization for numpyro (JAX backend) vs CPU for nutpie (PyTensor/Rust backend)
- **Long-running models**: Monitor resource usage over hours-long sampling runs
- **Production deployment**: Benchmark resource requirements for scaling decisions
- **Debugging**: Identify memory leaks or unexpected resource consumption

**Reference**: See [MLflow System Metrics documentation](https://mlflow.org/docs/latest/ml/tracking/system-metrics/) for advanced configuration.

### Live Sampling Callbacks (PyMC Default NUTS Only)

Monitor parameter evolution during sampling with `create_log_callback`:

```python
import mlflow
import pymc as pm
import pymc_marketing.mlflow

pymc_marketing.mlflow.autolog()

# Define model before MLflow run
with pm.Model() as model:
    mu = pm.Normal("mu", mu=0, sigma=1)
    sigma = pm.HalfNormal("sigma", sigma=1)
    obs = pm.Normal("obs", mu=mu, sigma=sigma, observed=data)

with mlflow.start_run():
    # Create callback to log stats and parameters during sampling
    callback = pymc_marketing.mlflow.create_log_callback(
        stats=["energy", "model_logp", "step_size"],
        parameters=["mu", "sigma_log__"],
        take_every=100,  # Log every 100 draws
    )
    
    # Callback only works with PyMC's default NUTS
    # Use model= parameter to avoid nested context
    idata = pm.sample(
        model=model,
        draws=1000,
        chains=4,
        random_seed=42,
        callback=callback,  # Log during sampling
    )
```

**What gets logged**: Time-series of sampler statistics and parameter values throughout the MCMC run, visible in MLflow UI metrics tab.

**Limitation**: `create_log_callback` only works with PyMC's default NUTS sampler (not nutpie or numpyro). For production models, prefer nutpie/numpyro for speed and skip live callbacks—post-sampling diagnostics via autolog are sufficient.

### Manual Logging for Custom Artifacts and Metrics

Extend autolog with domain-specific artifacts and agreed-upon comparison metrics:

```python
import mlflow
import pymc as pm
import arviz as az
import numpy as np

pymc_marketing.mlflow.autolog()  # Still captures defaults

# Define model before MLflow run
with pm.Model() as model:
    # ... model specification ...
    pass

with mlflow.start_run():
    mlflow.set_tag("run_type", "production")
    mlflow.set_tag("mock_fit", "false")
    mlflow.log_param("random_seed", 42)
    
    # Use model= parameter to avoid nested context
    idata = pm.sample(model=model, nuts_sampler="nutpie", draws=1000, chains=4, random_seed=42)
    
    # ============================================
    # CHECK FOR SAMPLING ISSUES (log immediately)
    # ============================================
    n_divergences = idata.sample_stats["diverging"].sum().item()
    mlflow.log_metric("n_divergences", n_divergences)
    
    if n_divergences > 0:
        mlflow.set_tag("warning", f"Found {n_divergences} divergences")
    
    # ============================================
    # CUSTOM ARTIFACTS
    # ============================================
    
    # Posterior predictive check
    pm.sample_posterior_predictive(idata, model=model, extend_inferencedata=True)
    
    fig = az.plot_ppc(idata, kind="cumulative")
    mlflow.log_figure(fig, "posterior_predictive_check.png")
    
    # Trace plot
    fig = az.plot_trace(idata, compact=True)
    mlflow.log_figure(fig, "trace_plot.png")
    
    # Parameter posterior plots
    fig = az.plot_posterior(idata, var_names=["mu", "sigma"])
    mlflow.log_figure(fig, "posterior_distributions.png")
    
    # ============================================
    # AGREED-UPON METRICS
    # ============================================
    # Log the same metrics across all runs for effective comparison
    
    # Bayesian model comparison (if log_likelihood available)
    try:
        loo = az.loo(idata, pointwise=True)
        mlflow.log_metric("elpd_loo", loo.elpd_loo)
        mlflow.log_metric("p_loo", loo.p_loo)
        mlflow.log_metric("loo_se", loo.se)
        
        # Flag problematic Pareto k values
        bad_k = (loo.pareto_k > 0.7).sum().item()
        mlflow.log_metric("bad_k_count", bad_k)
    except Exception as e:
        # nutpie doesn't store log_likelihood automatically
        mlflow.log_param("loo_available", "false")
    
    # Posterior predictive metrics (if you have held-out data)
    if y_test is not None:
        y_pred = idata.posterior_predictive["y"].mean(dim=["chain", "draw"])
        mae = np.abs(y_test - y_pred).mean()
        rmse = np.sqrt(((y_test - y_pred) ** 2).mean())
        
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("rmse", rmse)
    
    # Convergence metrics (already logged by autolog, but can access directly)
    summary = az.summary(idata)
    mlflow.log_metric("min_ess_bulk", summary["ess_bulk"].min())
    mlflow.log_metric("min_ess_tail", summary["ess_tail"].min())
    mlflow.log_metric("max_rhat", summary["r_hat"].max())
```

**Key principle**: Choose metrics that align with your project's standards for comparing models. Log the *same metrics* across all production runs to enable sorting and filtering in the MLflow UI.

**Examples of domain-specific metrics**:
- **Bayesian model comparison**: ELPD (LOO, WAIC), stacking weights
- **Predictive accuracy**: MAE, RMSE, coverage probability
- **Domain-specific**: AUC (classification), MAPE (forecasting), lift (causal inference)

**Note**: This skill doesn't prescribe which metrics to use—see `pymc-modeling` skill for model assessment details. The focus here is *how* to log them consistently with MLflow.

## InferenceData Management

### What's in InferenceData

ArviZ's `InferenceData` is the standard container for Bayesian workflow artifacts:

```python
import arviz as az

# Inspect groups
print(idata)
```

**Common groups**:
- `posterior`: MCMC samples (parameters × draws × chains)
- `prior`: Prior samples (if `sample_prior_predictive` called)
- `posterior_predictive`: Predictions from posterior (if `sample_posterior_predictive` called)
- `observed_data`: Observed data used in likelihood
- `constant_data`: Predictors, coordinates, other non-observed data
- `sample_stats`: Diagnostics (divergences, tree depth, energy, accept probability)
- `log_likelihood`: Pointwise log-likelihood (required for LOO-CV, WAIC)—**not computed automatically**, must call `pm.compute_log_likelihood()` after sampling
- `warmup`: Warmup/tuning samples (not shown by default in `print(idata)`, but can be as large as posterior)

**Important notes**:
- `log_likelihood` is **not** computed automatically by any sampler—you must explicitly call `pm.compute_log_likelihood(idata, model=model)` if you need LOO-CV or WAIC
- Some samplers (e.g., nutpie) may include **unconstrained variables** in the posterior (e.g., `sigma_log__` instead of `sigma`). These are internal sampler representations and often not needed for analysis—check what's in your posterior before storing everything
- `warmup` groups are hidden by default but can double your storage size. Inspect with `idata.warmup` to check if present

### Size Considerations

InferenceData files can be large (multi-GB) depending on:
- Number of parameters × draws × chains
- Size of `posterior_predictive` (predictions for all observations × draws × chains)
- Whether `prior` and `prior_predictive` are included
- Hidden `warmup` groups (can double storage size)
- Unconstrained variables from certain samplers

**Decision framework**:
- **Small models** (< 50 parameters, < 10K draws): Store everything including warmup for diagnostics
- **Medium models** (50-200 parameters): Drop `prior`, `prior_predictive`, and `warmup`; thin if needed
- **Large models** (> 200 parameters, large predictions): Aggressively thin, drop unconstrained variables

**Strategies for managing size**:

```python
# 1. Drop unnecessary groups before logging
idata_lite = idata.copy()
del idata_lite.prior
del idata_lite.prior_predictive

# 2. Check for and remove warmup groups (hidden but can be large)
if hasattr(idata_lite, 'warmup'):
    del idata_lite.warmup

# 3. Drop unconstrained variables (e.g., from nutpie sampler)
# Inspect what's in posterior first
print(idata.posterior.data_vars)

# If you see variables like 'sigma_log__', these are unconstrained representations
# Drop them if you only need the constrained parameters
if 'sigma_log__' in idata_lite.posterior:
    idata_lite.posterior = idata_lite.posterior.drop_vars(['sigma_log__'])

mlflow.log_artifact(idata_lite, "idata_lite.nc")

# 4. Thin posterior for storage using arviz_stats (recommended over xarray slicing)
from arviz_stats import thin

# Thin to target ~2000-4000 effective samples across chains
# thin() automatically calculates optimal thinning factor based on ESS
idata_thinned = thin(idata, target_draws=2000)

# Alternative: Manual thinning with xarray (less intelligent)
# idata_thinned = idata.sel(draw=slice(None, None, 5))  # Keep every 5th draw
```

**Reference**: See [arviz-stats.thin documentation](https://python.arviz.org/projects/stats/en/stable/api/generated/arviz_stats.thin.html) for intelligent thinning based on effective sample size.

**Warning**: InferenceData already stores `observed_data` and `constant_data`. Don't log raw data files separately unless needed for other tools—this duplicates storage.

### Loading Artifacts Programmatically

```python
import mlflow
import arviz as az

# Load from specific run
client = mlflow.tracking.MlflowClient()
artifact_path = client.download_artifacts(run_id, "idata.nc")
idata = az.from_netcdf(artifact_path)

# Load from current run (within mlflow.start_run context)
artifact_uri = mlflow.get_artifact_uri("idata.nc")
idata = az.from_netcdf(artifact_uri)
```

### Computing log_likelihood for Model Comparison

**Important**: No PyMC sampler automatically computes `log_likelihood`—it's computationally expensive and not always needed. This applies to **all samplers** (nutpie, numpyro, default NUTS, etc.). 

If you need LOO-CV or WAIC for model comparison, you must explicitly compute it after sampling:

```python
import pymc as pm

idata = pm.sample(model=model, nuts_sampler="nutpie", draws=1000, chains=4, random_seed=42)

# Compute log_likelihood after sampling (required for LOO-CV/WAIC)
pm.compute_log_likelihood(idata, model=model)

# Now LOO-CV will work
loo = az.loo(idata)
mlflow.log_metric("elpd_loo", loo.elpd_loo)
```

This applies to all samplers (nutpie, numpyro, default NUTS, etc.).

## Domain-Specific Extensions

### PyMC-Marketing MMM

Marketing Mix Models (MMM) have specialized logging support via `pymc_marketing.mlflow`:

```python
import pandas as pd
import mlflow
from pymc_marketing.mmm import MMM, GeometricAdstock, LogisticSaturation
import pymc_marketing.mlflow

# Enable MMM-specific autologging
pymc_marketing.mlflow.autolog(log_mmm=True)

# Load data
data = pd.read_csv("mmm_data.csv", parse_dates=["date_week"])
X = data.drop("y", axis=1)
y = data["y"]

# Define model
mmm = MMM(
    adstock=GeometricAdstock(l_max=8),
    saturation=LogisticSaturation(),
    date_column="date_week",
    channel_columns=["tv", "radio", "digital"],
    control_columns=["holiday", "trend"],
    yearly_seasonality=2,
)

mlflow.set_experiment("MMM - Production")

with mlflow.start_run():
    # ============================================
    # MMM-SPECIFIC TAGS
    # ============================================
    mlflow.set_tag("model_type", "MMM")
    mlflow.set_tag("run_type", "production")
    
    # Log MMM configuration
    mlflow.log_param("adstock", "GeometricAdstock")
    mlflow.log_param("saturation", "LogisticSaturation")
    mlflow.log_param("adstock_l_max", 8)
    mlflow.log_param("n_channels", len(["tv", "radio", "digital"]))
    mlflow.log_param("yearly_seasonality", 2)
    
    # ============================================
    # FIT MODEL
    # ============================================
    # autolog captures sampling diagnostics, InferenceData, etc.
    idata = mmm.fit(X, y, draws=1000, chains=4, nuts_sampler="nutpie", random_seed=42)
    
    # Check for sampling issues
    n_divergences = idata.sample_stats["diverging"].sum().item()
    mlflow.log_metric("n_divergences", n_divergences)
    
    # ============================================
    # MMM-SPECIFIC ARTIFACTS
    # ============================================
    # Components contribution plot
    fig = mmm.plot_components_contributions()
    mlflow.log_figure(fig, "mmm_components.png")
    
    # Channel contribution waterfall
    fig = mmm.plot_channel_contribution_share_hdi()
    mlflow.log_figure(fig, "channel_contributions.png")
    
    # Transformation curves (adstock, saturation, seasonality)
    for transform in [mmm.adstock, mmm.saturation, mmm.yearly_fourier]:
        curve = transform.sample_curve(idata.posterior)
        fig, _ = transform.plot_curve(curve)
        mlflow.log_figure(fig, f"{transform.prefix}_curve.png")
    
    # ============================================
    # MMM EVALUATION METRICS
    # ============================================
    # In-sample and out-of-sample metrics with helper function
    import pymc_marketing.mlflow
    
    # In-sample predictions
    in_sample_predictions = mmm.sample_posterior_predictive(X_pred=X)
    
    pymc_marketing.mlflow.log_mmm_evaluation_metrics(
        y_true=y,
        y_pred=in_sample_predictions.y,
        prefix="in-sample",
        metrics_to_calculate=["r_squared", "rmse", "mae", "mape"],
    )
    
    # Out-of-sample predictions (if you have test data)
    if X_test is not None:
        out_sample_predictions = mmm.sample_posterior_predictive(
            X_pred=X_test,
            include_last_observations=True,
        )
        
        pymc_marketing.mlflow.log_mmm_evaluation_metrics(
            y_true=y_test,
            y_pred=out_sample_predictions.y,
            prefix="out-sample",
            metrics_to_calculate=["r_squared", "rmse", "mae", "mape"],
        )
    
    # ============================================
    # MODEL REGISTRATION FOR DEPLOYMENT
    # ============================================
    # PyMC-Marketing models support serialization for production
    pymc_marketing.mlflow.log_mmm(mmm, artifact_path="mmm_model")
```

**What `log_mmm` enables**: Stores model in PyMC-Marketing's serialization format, preserving:
- Model specification (adstock, saturation, priors)
- Fitted parameters (InferenceData)
- Preprocessing transformers (scalers, validators)

**Loading for deployment**:
```python
from pymc_marketing.mlflow import load_mmm

# Load from specific run
mmm_deployed = load_mmm(run_id="abc123")

# Make predictions
y_pred = mmm_deployed.predict(X_new)
```

**Key tags for MMM workflows**:
- `model_type = "MMM"`
- `adstock`, `saturation` (transformation choices)
- `n_channels`, `yearly_seasonality` (model configuration)

### PyMC-Marketing CLV

Customer Lifetime Value (CLV) models also have specialized autologging:

```python
import pandas as pd
import mlflow
from pymc_marketing.clv import BetaGeoModel
import pymc_marketing.mlflow

# Enable CLV-specific autologging
pymc_marketing.mlflow.autolog(log_clv=True)

# Load data
data = pd.read_csv("clv_data.csv")
data["customer_id"] = data.index

model = BetaGeoModel(data=data)

mlflow.set_experiment("CLV - Production")

with mlflow.start_run():
    # ============================================
    # CLV-SPECIFIC TAGS
    # ============================================
    mlflow.set_tag("model_type", "CLV")
    mlflow.set_tag("clv_model", "BetaGeo")
    mlflow.set_tag("run_type", "production")
    
    # ============================================
    # FIT MODEL
    # ============================================
    # Log fit method (MCMC vs MAP)
    fit_method = "MCMC"  # or "MAP"
    mlflow.set_tag("fit_method", fit_method)
    
    if fit_method == "MCMC":
        model.fit(draws=1000, chains=4, nuts_sampler="nutpie", random_seed=42)
        mlflow.log_param("draws", 1000)
        mlflow.log_param("chains", 4)
        mlflow.log_param("random_seed", 42)
    else:
        model.fit(fit_method="map")
    
    # ============================================
    # CLV-SPECIFIC ARTIFACTS
    # ============================================
    # Probability alive matrix
    fig = model.plot_probability_alive_matrix()
    mlflow.log_figure(fig, "probability_alive_matrix.png")
    
    # Frequency-recency matrix
    fig = model.plot_frequency_recency_matrix()
    mlflow.log_figure(fig, "frequency_recency_matrix.png")
```

**Key tags for CLV workflows**:
- `model_type = "CLV"`
- `clv_model` (BetaGeo, ParetoNBD, etc.)
- `fit_method` (MCMC vs MAP)

### Serialization and Deployment

PyMC-Marketing models (MMM, CLV) use a serialization-enabled infrastructure that preserves full model state for deployment:

**What gets serialized**:
- Model specification (distributions, priors, transformations)
- InferenceData (posterior samples)
- Preprocessing pipelines (scalers, validators)
- Metadata (column names, coordinates)

**Loading in production**:
```python
from pymc_marketing.mlflow import load_mmm

# Load model from registry
mmm = load_mmm(run_id="abc123", keep_idata=True)

# Model is ready for prediction
predictions = mmm.predict(new_data)
```

**See**: [PyMC-Marketing MLflow docs](https://www.pymc-marketing.io/en/latest/api/generated/pymc_marketing.mlflow.html) for full serialization details.

## Comparing Runs

### MLflow UI Workflow

**Filter production runs with sufficient sampling**:
1. Navigate to experiment: "PyMC Models - Production"
2. Filter: `tags.run_type = "production" AND tags.mock_fit = "false" AND params.total_samples >= 2000`
3. Sort by: `metrics.elpd_loo DESC` (or your agreed-upon metric)
4. Select runs for comparison
5. View side-by-side: metrics, parameters, artifacts

**Compare specific model families**:
```
params.likelihood = "NegativeBinomial" AND tags.run_type = "production"
```

**Find models by complexity threshold**:
```
params.n_parameters <= 20 AND metrics.max_rhat < 1.01
```

### Programmatic Comparison

```python
import mlflow
import pandas as pd

# Search for production runs
runs = mlflow.search_runs(
    experiment_names=["PyMC Models - Production"],
    filter_string='tags.run_type = "production" AND tags.mock_fit = "false" AND params.total_samples >= 2000',
    order_by=["metrics.elpd_loo DESC"]
)

# View top models
print(runs[["run_id", "metrics.elpd_loo", "params.likelihood", "params.total_samples"]].head())

# Load best model's InferenceData
best_run_id = runs.iloc[0]["run_id"]
artifact_path = mlflow.artifacts.download_artifacts(
    run_id=best_run_id,
    artifact_path="idata.nc"
)

import arviz as az
idata_best = az.from_netcdf(artifact_path)

# If you need LOO-CV for model comparison, compute log_likelihood first
# (no sampler computes it automatically)
# pm.compute_log_likelihood(idata_best, model=model)

# Compare specific artifacts across runs
for idx, row in runs.head(3).iterrows():
    run_id = row["run_id"]
    ppc_path = mlflow.artifacts.download_artifacts(
        run_id=run_id,
        artifact_path="posterior_predictive_check.png"
    )
    print(f"PPC for run {run_id}: {ppc_path}")
```

### Switching Between Experiments

```python
# Compare test vs production runs
test_runs = mlflow.search_runs(experiment_names=["PyMC Models - Test"])
prod_runs = mlflow.search_runs(experiment_names=["PyMC Models - Production"])

# Identify models promoted from test to production
# (match by parameters, compare metrics)
```

## Troubleshooting

### Large InferenceData Files

**Problem**: `idata.nc` exceeds 1-5 GB, slowing uploads/downloads.

**Solutions**:
```python
# Drop unnecessary groups
idata_lite = idata.copy()
del idata_lite.prior
del idata_lite.prior_predictive

# Thin posterior draws
idata_thinned = idata.sel(draw=slice(None, None, 5))  # Every 5th draw

# Log lite version
import tempfile
import os

with tempfile.TemporaryDirectory() as tmpdir:
    path = os.path.join(tmpdir, "idata_lite.nc")
    idata_lite.to_netcdf(path)
    mlflow.log_artifact(path, "idata_lite.nc")
```

### Missing log_likelihood for LOO-CV/WAIC

**Problem**: LOO-CV or WAIC fails because `log_likelihood` group is missing.

**Cause**: No PyMC sampler computes `log_likelihood` automatically—it's computationally expensive and not always needed. This applies to all samplers (nutpie, numpyro, default NUTS).

**Solution**: Explicitly compute log_likelihood after sampling if you need it for model comparison:
```python
import pymc as pm

idata = pm.sample(model=model, nuts_sampler="nutpie", draws=1000, chains=4, random_seed=42)

# Compute log_likelihood only if needed for LOO/WAIC
pm.compute_log_likelihood(idata, model=model)

# Now LOO works
import arviz as az
loo = az.loo(idata)
mlflow.log_metric("elpd_loo", loo.elpd_loo)
```

**Note**: This applies to all samplers (nutpie, numpyro, default NUTS), not just nutpie.

### Autolog Conflicts with Custom Callbacks

**Problem**: Custom sampling callbacks interfere with autolog.

**Solution**: Disable autolog and use manual logging helpers:
```python
# Don't call autolog()
import pymc_marketing.mlflow as pm_mlflow

with mlflow.start_run():
    idata = pm.sample(...)
    
    # Manually log components
    pm_mlflow.log_versions()
    pm_mlflow.log_model_derived_info(model)
    pm_mlflow.log_sample_diagnostics(idata)
    pm_mlflow.log_inference_data(idata)
```

### Duplicate Data in Artifacts

**Problem**: Raw data logged as CSV *and* stored in InferenceData's `observed_data` group.

**Solution**: InferenceData already preserves data—only log raw files if needed for other tools:
```python
# InferenceData includes observed_data and constant_data
print(idata.observed_data)  # Your y values (as stored in PyMC data containers)
print(idata.constant_data)  # Your X values, coords (as stored in PyMC data containers)

# Don't also log data.csv unless required for non-PyMC tools
```

**Important distinction**: The data in InferenceData (`observed_data`, `constant_data`) is what was stored in PyMC's data containers during model definition—this may differ from your original raw data if you applied transformations (scaling, log transforms, encoding, filtering, etc.) before passing to PyMC.

**When to log additional data artifacts**:
- **Raw data**: If you transformed data before modeling and need to track the original input
- **Transformed data**: If preprocessing steps aren't captured in the model definition
- **Data schema**: Use MLflow's [dataset logging](https://mlflow.org/docs/latest/ml/dataset/) to track data versions, schemas, and transformations—this is complementary to PyMC's InferenceData and helps document the full data pipeline

```python
import mlflow
import pandas as pd

# Example: Log both transformed data (in idata) and raw data provenance
with mlflow.start_run():
    # Log dataset schema/provenance if data was transformed
    raw_data = pd.read_csv("raw_sales.csv")
    transformed_data = preprocess(raw_data)  # Your transformations
    
    # Use MLflow's dataset tracking for the raw→transformed pipeline
    mlflow.log_input(
        mlflow.data.from_pandas(raw_data),
        context="raw_data"
    )
    
    # Model uses transformed_data
    with pm.Model() as model:
        obs = pm.Normal("obs", mu=0, sigma=1, observed=transformed_data["y"])
    
    idata = pm.sample(model=model)
    # idata.observed_data now contains transformed_data["y"], not raw values
```

## Code Organization Tips

### Separate Model Creation, Prior Sampling, and Inference

A clean pattern separates three concerns:
1. **Creating the model** (definition of priors, likelihood, structure)
2. **Sampling the prior** (if doing prior predictive checks)
3. **Doing inference** (MCMC sampling, posterior predictive checks)

You can achieve this by:
- Defining the model **before** the MLflow run context and using the `model=` parameter
- **Or equally useful**: Extracting model definition to a `create_model()` or `define_model()` function

Both approaches keep your MLflow tracking code clean and separate from model specification.

### Option 1: Define Before MLflow Context

```python
# Good: Clean separation, minimal indentation
with pm.Model() as model:
    mu = pm.Normal("mu", 0, 1)
    obs = pm.Normal("obs", mu=mu, sigma=1, observed=data)

with mlflow.start_run():
    mlflow.set_tag("run_type", "production")
    mlflow.log_param("random_seed", 42)
    idata = pm.sample(model=model, nuts_sampler="nutpie", random_seed=42)
    pm.sample_posterior_predictive(idata, model=model, extend_inferencedata=True)
```

```python
# Avoid: Nested contexts create deep indentation
with mlflow.start_run():
    with pm.Model() as model:
        mu = pm.Normal("mu", 0, 1)
        obs = pm.Normal("obs", mu=mu, sigma=1, observed=data)
        
        # Don't need another 'with model:' - just use model= parameter!
        idata = pm.sample(model=model, nuts_sampler="nutpie", random_seed=42)
```

### Option 2: Extract Model Definition to Functions

For complex models or multiple configurations, use helper functions:

```python
def define_normal_model(data):
    """Define a simple Normal model."""
    with pm.Model() as model:
        mu = pm.Normal("mu", mu=0, sigma=1)
        sigma = pm.HalfNormal("sigma", sigma=1)
        obs = pm.Normal("obs", mu=mu, sigma=sigma, observed=data)
    return model

def define_student_t_model(data):
    """Define a robust Student-t model."""
    with pm.Model() as model:
        mu = pm.Normal("mu", mu=0, sigma=1)
        sigma = pm.HalfNormal("sigma", sigma=1)
        nu = pm.Exponential("nu", lam=1/30)
        obs = pm.StudentT("obs", mu=mu, sigma=sigma, nu=nu, observed=data)
    return model

# Clean MLflow workflow
pymc_marketing.mlflow.autolog()
mlflow.set_tracking_uri("sqlite:///mlruns.db")
mlflow.set_experiment("Model Comparison")

for likelihood in ["normal", "student_t"]:
    model_fn = define_normal_model if likelihood == "normal" else define_student_t_model
    model = model_fn(data)
    
    with mlflow.start_run(run_name=f"{likelihood}_model"):
        mlflow.log_param("likelihood", likelihood)
        mlflow.log_param("random_seed", 42)
        idata = pm.sample(model=model, nuts_sampler="nutpie", random_seed=42)
```

**Benefits**:
- Cleaner code with less indentation
- Easier to test model specifications independently
- Simpler to run multiple model configurations
- Better separation of concerns (modeling vs tracking)

## Quick Reference Checklist

### Before Every Run

- [ ] Set tracking URI (`mlflow.set_tracking_uri(...)`)
- [ ] Set experiment name (`mlflow.set_experiment(...)`)
- [ ] Tag `run_type` (`"test"` or `"production"`)
- [ ] Tag `mock_fit` if using fast iteration
- [ ] Log sample counts: `draws`, `chains`, `total_samples`
- [ ] Log `random_seed` for reproducibility
- [ ] Log model configuration: `likelihood`, `sampler`, `n_parameters`
- [ ] Consider enabling system metrics (`log_system_metrics=True`) for long runs or GPU samplers

### After Sampling

- [ ] Check and log divergence count (`idata.sample_stats["diverging"].sum()`)
- [ ] Log agreed-upon comparison metrics (LOO, MAE, domain-specific)
- [ ] Log key diagnostic plots (PPC, trace, posterior distributions)
- [ ] Save InferenceData with consistent name (`idata.nc`)
- [ ] Verify autolog captured versions, diagnostics, metadata

### For Production Models

- [ ] Ensure `total_samples >= 2000` (or your threshold)
- [ ] Verify `mock_fit = "false"`
- [ ] Check convergence: `r_hat < 1.01`, `ess_bulk > 400`, `ess_tail > 400` (total across chains)
- [ ] Verify `n_divergences = 0` (or near zero)
- [ ] Include model serialization artifacts if using PyMC-Marketing
- [ ] Tag with domain-specific metadata (`model_type`, `adstock`, etc.)

## References

- [pymc_marketing.mlflow API documentation](https://www.pymc-marketing.io/en/latest/api/generated/pymc_marketing.mlflow.html)
- [PyMC testing utilities (mock_sample, etc.)](https://www.pymc.io/projects/docs/en/stable/api/testing.html)
- [PyMC-Marketing serialization and deployment patterns](https://www.pymc-marketing.io/)
- [arviz-stats.thin for intelligent posterior thinning](https://python.arviz.org/projects/stats/en/stable/api/generated/arviz_stats.thin.html)
- [MLflow system metrics monitoring](https://mlflow.org/docs/latest/ml/tracking/system-metrics/) - Track CPU, GPU, memory, network, and disk usage during sampling
- [Example repository: PyMC + MLflow workflows](https://github.com/williambdean/pymc-mlflow-example) - Progressive examples from basic logging to MMM autologging
- See `pymc-modeling` skill for Bayesian modeling fundamentals
- See `pymc-testing` skill for testing PyMC models and using mock utilities
- For general MLflow best practices (experiment lifecycle, model serving, registries), see dedicated MLflow skills
