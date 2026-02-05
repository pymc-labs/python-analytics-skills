# Mixture Models

## Table of Contents
- [Finite Mixture Models](#finite-mixture-models)
- [Label Switching Problem](#label-switching-problem)
- [Marginalized Mixtures](#marginalized-mixtures)
- [Diagnostics for Mixtures](#diagnostics-for-mixtures)

---

## Finite Mixture Models

Mixture models assume data comes from multiple subpopulations, each described by its own distribution. Use when:
- Data shows multimodality
- Subgroups exist but group membership is unknown
- You need to cluster observations probabilistically

### Gaussian Mixture

For simple univariate Gaussian mixtures, use `pm.NormalMixture`:

```python
import pymc as pm
import numpy as np

coords = {"component": range(K)}

with pm.Model(coords=coords) as gmm:
    # Mixture weights (Dirichlet prior)
    w = pm.Dirichlet("w", a=np.ones(K), dims="component")

    # Component means (with ordering constraint to avoid label switching)
    mu = pm.Normal("mu", mu=0, sigma=10, dims="component",
                   transform=pm.distributions.transforms.ordered)

    # Component standard deviations
    sigma = pm.HalfNormal("sigma", sigma=2, dims="component")

    # Mixture likelihood
    y = pm.NormalMixture("y", w=w, mu=mu, sigma=sigma, observed=y_obs)
```

### General Mixtures with pm.Mixture

For mixtures of arbitrary distributions, use `pm.Mixture`:

```python
with pm.Model(coords=coords) as general_mixture:
    # Weights
    w = pm.Dirichlet("w", a=np.ones(K))

    # Define component distributions
    components = [
        pm.Normal.dist(mu=pm.Normal("mu_0", 0, 5), sigma=pm.HalfNormal("sigma_0", 2)),
        pm.StudentT.dist(nu=3, mu=pm.Normal("mu_1", 0, 5), sigma=pm.HalfNormal("sigma_1", 2)),
    ]

    # Mixture
    y = pm.Mixture("y", w=w, comp_dists=components, observed=y_obs)
```

### Mixture of Regressions

When different subgroups follow different regression relationships:

```python
with pm.Model(coords={"component": range(K), "obs": range(N)}) as mixture_regression:
    # Mixture weights
    w = pm.Dirichlet("w", a=np.ones(K))

    # Component-specific regression coefficients
    alpha = pm.Normal("alpha", mu=0, sigma=5, dims="component")
    beta = pm.Normal("beta", mu=0, sigma=2, dims="component")
    sigma = pm.HalfNormal("sigma", sigma=1, dims="component")

    # Component distributions (one regression per component)
    components = [
        pm.Normal.dist(mu=alpha[k] + beta[k] * x, sigma=sigma[k])
        for k in range(K)
    ]

    y = pm.Mixture("y", w=w, comp_dists=components, observed=y_obs, dims="obs")
```

---

## Label Switching Problem

### The Problem

In mixture models, the likelihood is invariant to permutations of component labels. If you swap "component 1" and "component 2", the joint probability is unchanged. This creates:
- **Multimodal posterior**: K! equivalent modes
- **Meaningless component-wise summaries**: Averaging across modes mixes components
- **Failed diagnostics**: R-hat appears bad because chains find different modes

### Detecting Label Switching

```python
# Trace plots show "switching" between modes
az.plot_trace(idata, var_names=["mu"])

# Pair plots show symmetric clusters
az.plot_pair(idata, var_names=["mu"], coords={"component": [0, 1]})
```

### Solution 1: Ordering Constraints (Recommended)

Impose an ordering on component parameters to break symmetry:

```python
import pytensor.tensor as pt

with pm.Model(coords=coords) as gmm_ordered:
    w = pm.Dirichlet("w", a=np.ones(K))

    # Unordered means on unconstrained space
    mu_raw = pm.Normal("mu_raw", mu=0, sigma=10, dims="component")

    # Apply ordering constraint: mu[0] < mu[1] < ... < mu[K-1]
    mu = pm.Deterministic("mu", pt.sort(mu_raw), dims="component")

    sigma = pm.HalfNormal("sigma", sigma=2, dims="component")
    y = pm.NormalMixture("y", w=w, mu=mu, sigma=sigma, observed=y_obs)
```

Or use PyMC's built-in ordered transform:

```python
# This applies the ordered transform directly
mu = pm.Normal("mu", mu=0, sigma=10, dims="component",
               transform=pm.distributions.transforms.ordered)
```

**Note**: Ordering constraints only work when the ordered parameter differs meaningfully across components. For equal component means, use other identifiability strategies.

### Solution 2: Post-Processing (Relabeling)

When ordering constraints aren't natural, relabel samples post-hoc:

```python
# Simple relabeling based on component means
def relabel_samples(idata):
    """Relabel mixture components by sorting means within each draw."""
    mu = idata.posterior["mu"].values  # (chain, draw, component)

    # Get sort indices for each draw
    sort_idx = np.argsort(mu, axis=-1)

    # Apply to all component-indexed variables
    for var in ["mu", "sigma", "w"]:
        if var in idata.posterior:
            vals = idata.posterior[var].values
            # Gather along component axis using sort indices
            relabeled = np.take_along_axis(vals, sort_idx, axis=-1)
            idata.posterior[var].values = relabeled

    return idata
```

For more sophisticated relabeling, see the `label.switching` R package or implement the Stephens algorithm.

### When Label Switching Doesn't Matter

If you only care about **predictions** (not component interpretation), label switching is harmless:

```python
# Posterior predictive is invariant to label permutations
with model:
    pm.sample_posterior_predictive(idata, extend_inferencedata=True)

# This is unaffected by label switching
az.plot_ppc(idata)
```

---

## Marginalized Mixtures

### Why Marginalize

Standard mixture models sample discrete component assignments, which:
- Requires specialized samplers (not NUTS)
- Often mixes poorly
- Scales badly with data size

**Marginalization** integrates out the discrete assignments analytically, enabling efficient NUTS sampling.

### Using pm.Mixture (Automatic Marginalization)

`pm.Mixture` and `pm.NormalMixture` automatically marginalize:

```python
# This is already marginalized - no discrete latent variables
y = pm.NormalMixture("y", w=w, mu=mu, sigma=sigma, observed=y_obs)
```

### pymc-extras MarginalMixture

For more complex marginalizations:

```python
import pymc_extras as pmx

with pm.Model() as marginal_model:
    # Discrete latent variable (will be marginalized)
    z = pm.Categorical("z", p=w)  # Not sampled directly

    # Conditional distributions
    y = pmx.MarginalMixture(
        "y",
        dist=[
            pm.Normal.dist(mu[0], sigma[0]),
            pm.Normal.dist(mu[1], sigma[1]),
        ],
        support_idxs=z,
        observed=y_obs,
    )
```

### When to Use Standard vs Marginalized

| Scenario | Recommendation |
|----------|----------------|
| Continuous components, want efficient sampling | Marginalized (`pm.Mixture`) |
| Need posterior on component assignments | Standard with Gibbs sampling |
| Large dataset | Marginalized (much faster) |
| Few observations per component | Either works |

---

## Diagnostics for Mixtures

### Checking for Label Switching

```python
# 1. Trace plots should NOT show "switching" patterns
az.plot_trace(idata, var_names=["mu", "w"])

# 2. Rank plots should be uniform (not bimodal)
az.plot_rank(idata, var_names=["mu"])

# 3. R-hat should be < 1.01 (won't be if label switching occurs)
summary = az.summary(idata, var_names=["mu", "sigma", "w"])
print(summary[["r_hat"]])
```

### Posterior Predictive Checks

```python
with model:
    pm.sample_posterior_predictive(idata, extend_inferencedata=True)

# Check if mixture captures data distribution shape
az.plot_ppc(idata, kind="kde")

# For multimodal data, cumulative is often clearer
az.plot_ppc(idata, kind="cumulative")
```

### Model Selection for Number of Components

Compare models with different K using LOO-CV:

```python
# Fit models with K=2, 3, 4 components
models = {}
for K in [2, 3, 4]:
    with build_mixture_model(K) as model:
        idata = pm.sample(nuts_sampler="nutpie")
        models[f"K={K}"] = idata

# Compare
comparison = az.compare(models, ic="loo")
print(comparison[["rank", "elpd_loo", "d_loo", "weight"]])
az.plot_compare(comparison)
```

**Caution**: LOO can be unreliable for mixture models due to high Pareto k values. Consider:
- K-fold cross-validation when LOO diagnostics fail
- WAIC as a secondary check
- Domain knowledge about plausible number of components

### Assessing Component Separation

```python
# Posterior distribution of component means
az.plot_posterior(idata, var_names=["mu"])

# Check overlap between components
# Well-separated components have non-overlapping HDIs
summary = az.summary(idata, var_names=["mu"], hdi_prob=0.94)
print(summary[["mean", "hdi_3%", "hdi_97%"]])
```

---

## See Also

- [priors.md](priors.md) - Prior selection for mixture components
- [diagnostics.md](diagnostics.md) - General convergence diagnostics
- [gotchas.md](gotchas.md) - Common modeling pitfalls
