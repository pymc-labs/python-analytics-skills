# Specialized Likelihoods

## Table of Contents
- [Zero-Inflated Models](#zero-inflated-models)
- [Hurdle Models](#hurdle-models)
- [Censored Data](#censored-data)
- [Truncated Distributions](#truncated-distributions)
- [Ordinal Regression](#ordinal-regression)
- [Robust Regression](#robust-regression)

---

## Zero-Inflated Models

### When to Use

Use zero-inflated models when data has more zeros than the base distribution predicts. Common in:
- Count data with structural zeros (species never present at a site)
- Medical data (many healthy patients with zero symptoms)
- Insurance claims (many policies with no claims)

### Zero-Inflated Poisson

```python
import pymc as pm

with pm.Model() as zip_model:
    # Zero-inflation probability
    psi = pm.Beta("psi", alpha=2, beta=2)  # P(structural zero)

    # Poisson rate for non-zero process
    mu = pm.Exponential("mu", lam=1)

    # Zero-Inflated Poisson likelihood
    y = pm.ZeroInflatedPoisson("y", psi=psi, mu=mu, observed=y_obs)
```

**Interpretation**:
- `psi`: Probability of a structural zero (zero from a separate process)
- `mu`: Mean of the Poisson process for non-structural observations
- Observed zeros come from both sources: structural (with prob `psi`) and sampling zeros from Poisson

### Zero-Inflated Poisson Regression

```python
with pm.Model() as zip_regression:
    # Regression on zero-inflation (logit link)
    alpha_psi = pm.Normal("alpha_psi", 0, 2)
    beta_psi = pm.Normal("beta_psi", 0, 1, dims="features")
    logit_psi = alpha_psi + pm.math.dot(X, beta_psi)
    psi = pm.math.sigmoid(logit_psi)

    # Regression on Poisson rate (log link)
    alpha_mu = pm.Normal("alpha_mu", 0, 2)
    beta_mu = pm.Normal("beta_mu", 0, 1, dims="features")
    log_mu = alpha_mu + pm.math.dot(X, beta_mu)
    mu = pm.math.exp(log_mu)

    # Zero-Inflated Poisson
    y = pm.ZeroInflatedPoisson("y", psi=psi, mu=mu, observed=y_obs)
```

### Zero-Inflated Negative Binomial

When data is both zero-inflated AND overdispersed:

```python
with pm.Model() as zinb_model:
    psi = pm.Beta("psi", alpha=2, beta=2)
    mu = pm.Exponential("mu", lam=0.5)
    alpha = pm.Exponential("alpha", lam=1)  # overdispersion

    y = pm.ZeroInflatedNegativeBinomial(
        "y", psi=psi, mu=mu, alpha=alpha, observed=y_obs
    )
```

### Zero-Inflated Binomial

For bounded count data with excess zeros:

```python
with pm.Model() as zib_model:
    psi = pm.Beta("psi", alpha=2, beta=2)
    p = pm.Beta("p", alpha=2, beta=2)  # success probability

    y = pm.ZeroInflatedBinomial("y", psi=psi, n=n_trials, p=p, observed=y_obs)
```

---

## Hurdle Models

### Hurdle vs Zero-Inflated

| Aspect | Zero-Inflated | Hurdle |
|--------|---------------|--------|
| Conceptual | Two sources of zeros | One process for zero/nonzero, another for positive counts |
| Zeros | From both processes | Only from the "hurdle" process |
| Positive values | From count process (which can also produce zeros) | From truncated count process |

Use **hurdle** when:
- Zero represents "no event occurred" (binary decision)
- Positive counts represent "how many given event occurred"

Use **zero-inflated** when:
- Some zeros are "structural" (impossible for event to occur)
- Other zeros are sampling zeros from a count process

### Hurdle Poisson

```python
import pytensor.tensor as pt

with pm.Model() as hurdle_poisson:
    # Probability of crossing the hurdle (having any count)
    theta = pm.Beta("theta", alpha=2, beta=2)

    # Poisson rate for positive counts
    mu = pm.Exponential("mu", lam=1)

    # Custom likelihood for hurdle model
    def hurdle_logp(y, theta, mu):
        # P(y=0) = 1 - theta
        # P(y=k | k>0) = theta * Poisson(k|mu) / (1 - Poisson(0|mu))
        zero_logp = pt.log(1 - theta)
        pos_logp = (
            pt.log(theta)
            + pm.logp(pm.Poisson.dist(mu=mu), y)
            - pt.log(1 - pt.exp(-mu))  # truncation adjustment
        )
        return pt.where(pt.eq(y, 0), zero_logp, pos_logp)

    y = pm.CustomDist("y", theta, mu, logp=hurdle_logp, observed=y_obs)
```

### Hurdle Negative Binomial

```python
with pm.Model() as hurdle_negbin:
    theta = pm.Beta("theta", alpha=2, beta=2)
    mu = pm.Exponential("mu", lam=0.5)
    alpha = pm.Exponential("alpha", lam=1)

    def hurdle_nb_logp(y, theta, mu, alpha):
        nb_dist = pm.NegativeBinomial.dist(mu=mu, alpha=alpha)
        p_zero_nb = pt.exp(pm.logp(nb_dist, 0))

        zero_logp = pt.log(1 - theta)
        pos_logp = (
            pt.log(theta)
            + pm.logp(nb_dist, y)
            - pt.log(1 - p_zero_nb)
        )
        return pt.where(pt.eq(y, 0), zero_logp, pos_logp)

    y = pm.CustomDist("y", theta, mu, alpha, logp=hurdle_nb_logp, observed=y_obs)
```

---

## Censored Data

### Types of Censoring

- **Right censoring**: Value known to be above some threshold (e.g., survival time > study end)
- **Left censoring**: Value known to be below some threshold (e.g., concentration < detection limit)
- **Interval censoring**: Value known to be within an interval

### pm.Censored

PyMC's `pm.Censored` wraps any distribution to handle censoring:

```python
with pm.Model() as censored_model:
    mu = pm.Normal("mu", mu=0, sigma=10)
    sigma = pm.HalfNormal("sigma", sigma=5)

    # Right-censored at upper_bound
    y = pm.Censored(
        "y",
        dist=pm.Normal.dist(mu=mu, sigma=sigma),
        lower=None,           # no left censoring
        upper=upper_bound,    # right censoring threshold
        observed=y_obs,
    )
```

### Right-Censored Data

Common in survival analysis where follow-up ends before events occur:

```python
# y_obs: observed times (events) or censoring times
# censored: boolean indicator (True if censored)

with pm.Model() as survival_model:
    mu = pm.Normal("mu", mu=3, sigma=2)
    sigma = pm.HalfNormal("sigma", sigma=1)

    # Upper bound is observed time for censored observations, None otherwise
    upper = np.where(censored, y_obs, np.inf)

    y = pm.Censored(
        "y",
        dist=pm.LogNormal.dist(mu=mu, sigma=sigma),
        lower=None,
        upper=upper,
        observed=y_obs,
    )
```

### Left-Censored Data (Detection Limits)

When measurements below a threshold are recorded as the threshold:

```python
# y_obs: observed values (detection_limit for censored observations)
# below_limit: boolean indicator

with pm.Model() as detection_limit_model:
    mu = pm.Normal("mu", mu=0, sigma=5)
    sigma = pm.HalfNormal("sigma", sigma=2)

    lower = np.where(below_limit, -np.inf, y_obs)  # actual lower for censored
    upper = np.where(below_limit, detection_limit, y_obs)

    # For left-censored: value is somewhere in (-inf, detection_limit]
    y = pm.Censored(
        "y",
        dist=pm.Normal.dist(mu=mu, sigma=sigma),
        lower=lower,
        upper=upper,
        observed=y_obs,
    )
```

### Tobit Regression (Censored Regression)

```python
with pm.Model() as tobit:
    alpha = pm.Normal("alpha", 0, 10)
    beta = pm.Normal("beta", 0, 2, dims="features")
    sigma = pm.HalfNormal("sigma", sigma=2)

    mu = alpha + pm.math.dot(X, beta)

    # Left-censored at 0 (common for expenditure, hours worked)
    y = pm.Censored(
        "y",
        dist=pm.Normal.dist(mu=mu, sigma=sigma),
        lower=0,
        upper=None,
        observed=y_obs,
    )
```

---

## Truncated Distributions

### pm.Truncated

Unlike censoring (where we know value exceeds a bound), truncation means **data outside bounds is never observed**. Use `pm.Truncated`:

```python
with pm.Model() as truncated_model:
    mu = pm.Normal("mu", mu=0, sigma=10)
    sigma = pm.HalfNormal("sigma", sigma=5)

    # Only observe values in [lower, upper]
    y = pm.Truncated(
        "y",
        dist=pm.Normal.dist(mu=mu, sigma=sigma),
        lower=0,      # no negative values observed
        upper=100,    # no values above 100 observed
        observed=y_obs,
    )
```

### Censored vs Truncated

| Aspect | Censored | Truncated |
|--------|----------|-----------|
| Data | Censored values recorded at bound | Values outside bounds not in dataset |
| Sample size | Fixed, includes censored obs | Varies, excludes out-of-bound obs |
| Example | Survival time > study end recorded as "censored at T" | Only customers who bought are in purchase dataset |

### Example: Truncated at Zero

```python
# Data: Only positive observations (e.g., income for employed)
with pm.Model() as positive_only:
    mu = pm.Normal("mu", mu=10, sigma=5)
    sigma = pm.HalfNormal("sigma", sigma=3)

    y = pm.Truncated(
        "y",
        dist=pm.Normal.dist(mu=mu, sigma=sigma),
        lower=0,
        upper=None,
        observed=y_obs,
    )
```

---

## Ordinal Regression

### When to Use

For ordered categorical outcomes: survey responses (1-5 stars), education levels, disease severity grades.

### pm.OrderedLogistic

```python
with pm.Model() as ordinal:
    # Regression coefficients
    beta = pm.Normal("beta", mu=0, sigma=2, dims="features")

    # Cutpoints (thresholds between categories)
    # K-1 cutpoints for K categories
    cutpoints = pm.Normal(
        "cutpoints",
        mu=np.linspace(-2, 2, n_categories - 1),
        sigma=1,
        transform=pm.distributions.transforms.ordered,
        shape=n_categories - 1,
    )

    # Linear predictor
    eta = pm.math.dot(X, beta)

    # Ordered logistic likelihood
    y = pm.OrderedLogistic("y", eta=eta, cutpoints=cutpoints, observed=y_obs)
```

### Interpreting Cutpoints

Cutpoints define boundaries between adjacent categories on the latent scale:
- `P(Y <= k) = logistic(cutpoints[k] - eta)`
- More positive `eta` → higher probability of higher categories

```python
# Posterior predictive probabilities for each category
with model:
    pm.sample_posterior_predictive(idata, extend_inferencedata=True)

# Examine predicted category probabilities
pred_probs = idata.posterior_predictive["y"]
```

### Priors for Cutpoints

The ordering constraint is critical:

```python
# Option 1: Ordered transform (recommended)
cutpoints = pm.Normal(
    "cutpoints", mu=0, sigma=2,
    transform=pm.distributions.transforms.ordered,
    shape=n_categories - 1,
)

# Option 2: Induced from differences
diffs = pm.HalfNormal("diffs", sigma=1, shape=n_categories - 1)
cutpoints = pm.Deterministic("cutpoints", pt.cumsum(diffs) - diffs.sum() / 2)
```

### Ordered Probit Alternative

```python
with pm.Model() as ordered_probit:
    beta = pm.Normal("beta", mu=0, sigma=2, dims="features")
    cutpoints = pm.Normal(
        "cutpoints", mu=0, sigma=2,
        transform=pm.distributions.transforms.ordered,
        shape=n_categories - 1,
    )

    eta = pm.math.dot(X, beta)

    # Probit link via cumulative normal
    y = pm.OrderedProbit("y", eta=eta, cutpoints=cutpoints, observed=y_obs)
```

---

## Robust Regression

### Student-t Likelihood

Replace Normal with Student-t for outlier-robust regression:

```python
with pm.Model() as robust_regression:
    alpha = pm.Normal("alpha", mu=0, sigma=10)
    beta = pm.Normal("beta", mu=0, sigma=2, dims="features")
    sigma = pm.HalfNormal("sigma", sigma=2)

    # Degrees of freedom: lower = heavier tails = more robust
    # nu=1 is Cauchy (very heavy), nu>30 ≈ Normal
    nu = pm.Exponential("nu", lam=1/30) + 1  # ensures nu > 1

    mu = alpha + pm.math.dot(X, beta)

    # Student-t downweights outliers
    y = pm.StudentT("y", nu=nu, mu=mu, sigma=sigma, observed=y_obs)
```

### Prior on Degrees of Freedom

The `nu` parameter controls robustness:

```python
# Weakly informative, allows both robust and normal-like behavior
nu = pm.Gamma("nu", alpha=2, beta=0.1)  # mode around 10

# Prior expecting heavy tails (more robust)
nu = pm.Exponential("nu", lam=1/10) + 1  # concentrated at small values

# Fixed (if you have strong prior knowledge)
nu = 4  # moderately heavy tails
```

### Comparing to Normal Likelihood

```python
# Fit both models
with normal_model:
    idata_normal = pm.sample()

with robust_model:
    idata_robust = pm.sample()

# Compare via LOO
comparison = az.compare({
    "normal": idata_normal,
    "robust": idata_robust,
}, ic="loo")
```

### Quantile Regression

For median or other quantile regression using asymmetric Laplace:

```python
import pytensor.tensor as pt

with pm.Model() as quantile_regression:
    alpha = pm.Normal("alpha", mu=0, sigma=10)
    beta = pm.Normal("beta", mu=0, sigma=2, dims="features")
    sigma = pm.HalfNormal("sigma", sigma=2)

    mu = alpha + pm.math.dot(X, beta)

    # Quantile (0.5 = median, 0.9 = 90th percentile)
    tau = 0.5

    # Asymmetric Laplace log-probability
    def asymmetric_laplace_logp(y, mu, sigma, tau):
        z = (y - mu) / sigma
        return pt.log(tau * (1 - tau) / sigma) - z * (tau - (z < 0))

    y = pm.CustomDist(
        "y", mu, sigma, tau,
        logp=asymmetric_laplace_logp,
        observed=y_obs,
    )
```

---

## See Also

- [priors.md](priors.md) - Prior selection guidance
- [diagnostics.md](diagnostics.md) - Convergence diagnostics
- [mixtures.md](mixtures.md) - Mixture models (related to zero-inflated)
- [gotchas.md](gotchas.md) - Common pitfalls
