# Inference Methods

## Table of Contents
- [Method Selection Guide](#method-selection-guide)
- [MCMC Samplers](#mcmc-samplers)
  - [nutpie (Default)](#nutpie-default)
  - [NumPyro/JAX Backend](#numpyrojax-backend)
  - [PyMC NUTS](#pymc-nuts)
- [Approximate Inference](#approximate-inference)
  - [Variational Inference (ADVI, DADVI)](#variational-inference)
  - [Pathfinder](#pathfinder)
- [Combining Methods](#combining-methods)

## Method Selection Guide

### MCMC Samplers (Exact Inference)

| Method | Best For | Speed | GPU | Notes |
|--------|----------|-------|-----|-------|
| **nutpie** | Default choice | 2-5x faster than PyMC | No | Rust-based, excellent adaptation |
| **NumPyro** | Large models, GPU | Fast | Yes | JAX-based, vectorized chains |
| **PyMC NUTS** | Compatibility | Baseline | No | Most tested, fallback option |

### Approximate Inference (Fast but Inexact)

| Method | Best For | Speed | GPU | Notes |
|--------|----------|-------|-----|-------|
| **ADVI** | Quick approximations | Fast | No | Mean-field or full-rank Gaussian |
| **DADVI** | Stable VI | Very fast | Yes | Deterministic gradients |
| **Pathfinder** | Initialization, screening | Very fast | Yes | Quasi-Newton optimization paths |

**When to use approximate inference:**
- Model screening before committing to full MCMC
- Very large datasets where MCMC is prohibitively slow
- Finding good initial values for MCMC
- Posteriors that are approximately Gaussian

**Caution**: Approximate methods underestimate posterior uncertainty and may miss multimodality. Always validate with MCMC when possible.

---

## Initialization and Common Failures

### "Initial evaluation of model at starting point failed"

This error occurs when the log-probability is `-inf` or `NaN` at initial parameter values.

**Common causes and fixes**:

| Cause | Fix |
|-------|-----|
| Data outside distribution support | Verify observed data matches likelihood bounds |
| Jitter pushes parameters outside constraints | Use `init="adapt_diag"` (no jitter) |
| Invalid default starting values | Specify `initvals={"param": value}` |
| Constant response variable | Ensure target variable has variance |

```python
# Fix 1: Reduce/eliminate initialization jitter
idata = pm.sample(init="adapt_diag")

# Fix 2: Specify valid starting values
idata = pm.sample(initvals={"sigma": 1.0, "beta": np.zeros(p)})

# Fix 3: Use ADVI for more robust initialization
idata = pm.sample(init="advi+adapt_diag")

# Debugging: check which variables have invalid log-probabilities
model.point_logps()
model.debug()
```

### The MCMC Prior Sampling Fallacy

**Common mistake**: Using `pm.sample()` to sample from the prior distribution.

```python
# BAD: pm.sample() uses MCMC even without observations
with prior_model:
    prior = pm.sample(draws=1000)  # slow, poor convergence for discrete vars

# GOOD: Use ancestral sampling for priors
with prior_model:
    prior = pm.sample_prior_predictive(draws=1000)  # instant, exact
```

`pm.sample_prior_predictive()` performs ancestral sampling (drawing directly from distributions in dependency order), which is instant and avoids all MCMC convergence issues.

---

## MCMC Samplers

### nutpie (Default)

Rust-based sampler with excellent mass matrix adaptation. Use as default.

#### Basic Usage

```python
import pymc as pm

with pm.Model() as model:
    # ... model specification ...
    pass

# Sample with nutpie backend
with model:
    idata = pm.sample(
        draws=1000,
        tune=1000,
        chains=4,
        nuts_sampler="nutpie",
        random_seed=42,
    )
```

#### Configuration Options

```python
with model:
    idata = pm.sample(
        draws=1000,
        tune=1000,
        chains=4,
        nuts_sampler="nutpie",
        random_seed=42,
        progressbar=True,
        target_accept=0.8,  # increase for difficult posteriors
        cores=4,            # number of parallel chains
    )
```

#### When to Use PyMC NUTS Instead

- Model uses features not yet supported by nutpie
- Need specific PyMC sampling features (step methods, compound steps)
- Debugging model specification issues

### NumPyro/JAX Backend

JAX-based sampling with GPU support and vectorized chains.

#### Setup

```python
import pymc as pm

# Optional: configure JAX for GPU
import jax
jax.config.update("jax_platform_name", "gpu")  # or "cpu"
```

#### Basic Usage

```python
with pm.Model() as model:
    # ... model specification ...
    pass

# Sample with NumPyro NUTS
with model:
    idata = pm.sample(
        draws=1000,
        tune=1000,
        chains=4,
        nuts_sampler="numpyro",
        random_seed=42,
    )
```

#### Vectorized Chains (GPU Efficient)

```python
# Run all chains in parallel on GPU
with model:
    idata = pm.sample(
        draws=1000,
        tune=1000,
        chains=4,
        nuts_sampler="numpyro",
        nuts_sampler_kwargs={"chain_method": "vectorized"},
    )
```

#### Configuration

```python
with model:
    idata = pm.sample(
        draws=1000,
        tune=1000,
        chains=4,
        nuts_sampler="numpyro",
        target_accept=0.9,
        nuts_sampler_kwargs={"max_tree_depth": 12},
        progressbar=True,
    )
```

#### When to Use NumPyro

- Large models that benefit from GPU
- Many chains needed (vectorization efficient)
- Already in JAX ecosystem

### PyMC NUTS

The default PyMC sampler. Use as fallback when nutpie is unavailable.

```python
with model:
    idata = pm.sample(
        draws=1000,
        tune=1000,
        chains=4,
        random_seed=42,
        target_accept=0.8,
    )
```

---

## Approximate Inference

### Variational Inference

#### ADVI (Automatic Differentiation Variational Inference)

Approximates the posterior with a Gaussian distribution.

```python
with model:
    approx = pm.fit(
        n=30000,
        method="advi",
        callbacks=[pm.callbacks.CheckParametersConvergence()],
    )

# Draw samples from the approximation
idata = approx.sample(1000)
```

#### Full-Rank ADVI

Better posterior approximation (captures correlations) at higher cost:

```python
with model:
    approx = pm.fit(
        n=50000,
        method="fullrank_advi",
    )
```

#### DADVI (Deterministic ADVI)

More stable variational inference from pymc-extras:

```python
import pymc_extras as pmx

with model:
    idata = pmx.fit(
        method="dadvi",
        num_steps=10000,
        random_seed=42,
    )
```

DADVI advantages:
- Deterministic gradients (no Monte Carlo noise)
- Faster convergence
- More stable optimization

### Pathfinder

Quasi-Newton variational method that follows optimization paths. Very fast for quick approximations or initialization.

```python
with model:
    idata = pm.fit(method="pathfinder")
```

#### Multi-Path Pathfinder

```python
with model:
    idata = pm.fit(
        method="pathfinder",
        num_paths=8,  # multiple optimization paths
        maxcor=10,    # L-BFGS history
    )
```

#### When to Use Pathfinder

- Quick posterior approximation (seconds vs minutes)
- Finding good initial values for MCMC
- Model screening before full inference
- When posterior is approximately Gaussian

---

## Combining Methods

### Pathfinder Initialization + MCMC

Use Pathfinder to find good starting points, then run MCMC for accurate inference:

```python
with model:
    # Quick pathfinder approximation
    pathfinder_idata = pm.fit(method="pathfinder")

    # Extract initial values
    init_vals = {
        var.name: pathfinder_idata.posterior[var.name].mean(dim=["chain", "draw"]).values
        for var in model.free_RVs
    }

    # Run MCMC with good initialization
    idata = pm.sample(initvals=init_vals)
```

### VI for Screening, MCMC for Final Inference

```python
# Screen model with VI (fast)
with model:
    vi_approx = pm.fit(n=30000)

# If model looks reasonable, run full MCMC
with model:
    idata = pm.sample()
```

### VI for Large Data, MCMC for Validation

```python
# Full data: use VI for speed
with full_model:
    vi_approx = pm.fit(n=30000)

# Validation subset: use MCMC for accurate uncertainty
with subset_model:
    idata = pm.sample()
```
