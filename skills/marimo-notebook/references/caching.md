# Marimo Caching Reference

## Built-in Caching Decorators

### In-Memory Cache (Session Only)
Fastest option, cleared when notebook restarts.

```python
@mo.cache
def expensive_function(x, y):
    return compute(x, y)
```

### Persistent Cache (Survives Restarts)
Cached to disk, persists across sessions.

```python
@mo.persistent_cache(name="embeddings")
def compute_embeddings(text):
    return model.encode(text)

# Block-style for multiple statements
with mo.persistent_cache(name="data_load"):
    df = load_large_dataset()
    processed = transform(df)
```

### LRU Cache with Size Limit
Bounded cache that evicts oldest entries.

```python
@mo.lru_cache(maxsize=128)
def bounded_cache(x):
    return compute(x)
```

## Model Output Caching for Data Science

For expensive MCMC sampling and Bayesian model fits, use this pattern to cache outputs and avoid unnecessary re-runs during development. This supplements marimo's built-in caching with data/parameter-aware invalidation.

### Configuration Cell

```python
@app.cell
def _():
    from pathlib import Path

    ENABLE_MODEL_CACHE = True  # Set False to force re-fitting
    CACHE_DIR = Path(".model_cache")
    CACHE_DIR.mkdir(exist_ok=True)
    return CACHE_DIR, ENABLE_MODEL_CACHE
```

### Cache Utilities with Hashing

```python
@app.cell
def _(CACHE_DIR):
    import hashlib
    from pathlib import Path

    def get_cache_path(name: str, data=None, **params) -> Path:
        """Generate cache path from name, data hash, and parameters."""
        components = [name]
        if data is not None:
            # Hash data shape and sample values for change detection
            data_repr = f"{type(data).__name__}_{getattr(data, 'shape', len(data))}"
            components.append(hashlib.md5(data_repr.encode()).hexdigest()[:6])
        if params:
            param_str = str(sorted(params.items()))
            components.append(hashlib.md5(param_str.encode()).hexdigest()[:6])
        return CACHE_DIR / f"{'_'.join(components)}.nc"

    return (get_cache_path,)
```

### PyMC Caching Pattern

```python
@app.cell
def _(mo, pm, az, X, y, ENABLE_MODEL_CACHE, get_cache_path):
    # Cache key includes data shape and sampling parameters
    cache_path = get_cache_path(
        "linear_model",
        data=y,
        n_samples=2000,
        seed=42
    )

    trace = None
    if ENABLE_MODEL_CACHE and cache_path.exists():
        trace = az.from_netcdf(cache_path)
        mo.output.append(mo.callout("Loaded trace from cache", kind="info"))

    if trace is None:
        with pm.Model() as model:
            alpha = pm.Normal("alpha", 0, 10)
            beta = pm.Normal("beta", 0, 10, shape=X.shape[1])
            sigma = pm.HalfNormal("sigma", 1)
            mu = alpha + pm.math.dot(X, beta)
            pm.Normal("y", mu, sigma, observed=y)
            trace = pm.sample(2000, nuts_sampler="nutpie", random_seed=42)

        if ENABLE_MODEL_CACHE:
            az.to_netcdf(trace, cache_path)

    return (trace,)
```

### When to Use Model Caching

**Use this pattern for:**
- MCMC sampling (>30 seconds runtime)
- Model comparison workflows
- Iterative model development where you're modifying downstream analysis

**Not needed for:**
- Quick computations (<5 seconds)—use `@mo.cache` instead
- Computations with automatic dependency tracking—use `@mo.persistent_cache`

Toggle `ENABLE_MODEL_CACHE = False` when you need fresh fits (e.g., after model specification changes not captured in parameters).
