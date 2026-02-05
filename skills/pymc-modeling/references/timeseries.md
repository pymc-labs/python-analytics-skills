# Time Series Models

## Table of Contents
- [Built-in Time Series Distributions](#built-in-time-series-distributions)
- [Autoregressive Models](#autoregressive-models)
- [Random Walk Models](#random-walk-models)
- [Structural Time Series](#structural-time-series)
- [GPs for Time Series](#gps-for-time-series)
- [Handling Seasonality](#handling-seasonality)
- [Forecasting](#forecasting)

## Built-in Time Series Distributions

PyMC provides specialized distributions for time series:

| Distribution | Use Case |
|-------------|----------|
| `pm.GaussianRandomWalk` | Random walk with Gaussian innovations |
| `pm.AR` | Autoregressive process of any order |
| `pm.GARCH11` | GARCH(1,1) volatility model |

## Autoregressive Models

### AR(1)

```python
import pymc as pm

with pm.Model(coords={"time": range(T)}) as ar1_model:
    # AR coefficient (stationary if |rho| < 1)
    rho = pm.Uniform("rho", -1, 1)
    sigma = pm.HalfNormal("sigma", sigma=1)

    # AR(1) process using built-in distribution
    y = pm.AR(
        "y",
        rho=[rho],
        sigma=sigma,
        init_dist=pm.Normal.dist(0, sigma),
        observed=y_obs,
        dims="time",
    )
```

### AR(p)

```python
with pm.Model(coords={"time": range(T), "lag": range(p)}) as arp_model:
    rho = pm.Normal("rho", 0, 0.5, dims="lag")
    sigma = pm.HalfNormal("sigma", sigma=1)

    y = pm.AR(
        "y",
        rho=rho,
        sigma=sigma,
        init_dist=pm.Normal.dist(0, sigma),
        observed=y_obs,
        dims="time",
    )
```

### AR with Constant/Intercept

To include a constant (intercept), set `constant=True` and include the intercept as the first element of `rho`.

```python
with pm.Model() as ar_intercept:
    # Intercept
    c = pm.Normal("c", 0, 10)
    # AR coefficient
    rho = pm.Uniform("rho", -1, 1)
    sigma = pm.HalfNormal("sigma", 1)

    # When constant=True, the first element of rho is the intercept
    y = pm.AR(
        "y",
        rho=[c, rho],
        constant=True,
        init_dist=pm.Normal.dist(c / (1 - rho), sigma),
        sigma=sigma,
        observed=y_obs,
    )
```

## Random Walk Models

### Gaussian Random Walk

```python
with pm.Model(coords={"time": range(T)}) as rw_model:
    sigma = pm.HalfNormal("sigma", sigma=1)

    # Built-in GaussianRandomWalk
    y = pm.GaussianRandomWalk(
        "y",
        sigma=sigma,
        init_dist=pm.Normal.dist(0, sigma),
        steps=T - 1,  # number of steps after initial value
        observed=y_obs,
        dims="time",
    )
```

### Random Walk with Drift

```python
with pm.Model(coords={"time": range(T)}) as rw_drift:
    drift = pm.Normal("drift", 0, 0.1)
    sigma = pm.HalfNormal("sigma", sigma=1)

    y = pm.GaussianRandomWalk(
        "y",
        mu=drift,  # drift parameter
        sigma=sigma,
        init_dist=pm.Normal.dist(0, sigma),
        steps=T - 1,
        observed=y_obs,
        dims="time",
    )
```

### Local Level Model (Random Walk + Observation Noise)

```python
with pm.Model(coords={"time": range(T)}) as local_level:
    # State noise
    sigma_state = pm.HalfNormal("sigma_state", sigma=1)
    # Observation noise
    sigma_obs = pm.HalfNormal("sigma_obs", sigma=1)

    # Latent random walk (state)
    state = pm.GaussianRandomWalk(
        "state",
        sigma=sigma_state,
        init_dist=pm.Normal.dist(y_obs[0], sigma_state),
        steps=T - 1,
        dims="time",
    )

    # Observations
    y = pm.Normal("y", mu=state, sigma=sigma_obs, observed=y_obs, dims="time")
```

### Local Linear Trend

```python
import pytensor.tensor as pt

with pm.Model(coords={"time": range(T)}) as local_linear:
    sigma_level = pm.HalfNormal("sigma_level", 0.1)
    sigma_slope = pm.HalfNormal("sigma_slope", 0.01)
    sigma_obs = pm.HalfNormal("sigma_obs", 0.5)

    # Slope follows random walk
    slope = pm.GaussianRandomWalk(
        "slope",
        sigma=sigma_slope,
        init_dist=pm.Normal.dist(0, 0.1),
        steps=T - 1,
        dims="time",
    )

    # Level follows random walk with drift = slope
    # Need to build manually since drift varies
    level_init = pm.Normal("level_init", y_obs[0], 1)
    level_innovations = pm.Normal("level_innov", 0, sigma_level, shape=T - 1)

    level = pm.Deterministic(
        "level",
        pt.concatenate([[level_init], level_init + pt.cumsum(slope[:-1] + level_innovations)]),
        dims="time",
    )

    y = pm.Normal("y", mu=level, sigma=sigma_obs, observed=y_obs, dims="time")
```

## GARCH Models

### GARCH(1,1)

```python
with pm.Model(coords={"time": range(T)}) as garch_model:
    # GARCH parameters
    omega = pm.HalfNormal("omega", sigma=0.1)
    alpha = pm.Beta("alpha", alpha=2, beta=5)  # ARCH term
    beta = pm.Beta("beta", alpha=5, beta=2)    # GARCH term

    # Initial volatility
    initial_vol = pm.HalfNormal("initial_vol", sigma=1)

    y = pm.GARCH11(
        "y",
        omega=omega,
        alpha_1=alpha,
        beta_1=beta,
        initial_vol=initial_vol,
        observed=returns,
        dims="time",
    )
```

## Structural Time Series

### Trend + Seasonality (Fourier)

```python
def make_fourier_features(t, period, n_terms):
    """Create Fourier basis for seasonality."""
    features = []
    for k in range(1, n_terms + 1):
        features.append(np.sin(2 * np.pi * k * t / period))
        features.append(np.cos(2 * np.pi * k * t / period))
    return np.column_stack(features)

with pm.Model(coords={"time": range(T)}) as structural_ts:
    # Trend: random walk with drift
    drift = pm.Normal("drift", 0, 0.1)
    sigma_trend = pm.HalfNormal("sigma_trend", 0.1)

    trend = pm.GaussianRandomWalk(
        "trend",
        mu=drift,
        sigma=sigma_trend,
        init_dist=pm.Normal.dist(y_obs[0], 1),
        steps=T - 1,
        dims="time",
    )

    # Seasonality: Fourier terms
    X_seasonal = make_fourier_features(np.arange(T), period=12, n_terms=3)
    beta_seasonal = pm.Normal("beta_seasonal", 0, 1, shape=6)
    seasonal = pm.Deterministic("seasonal", X_seasonal @ beta_seasonal, dims="time")

    # Observation
    sigma_obs = pm.HalfNormal("sigma_obs", 0.5)
    y = pm.Normal("y", mu=trend + seasonal, sigma=sigma_obs, observed=y_obs, dims="time")
```

### Multiple Seasonalities

```python
with pm.Model(coords={"time": range(T)}) as multi_seasonal:
    # Weekly (period=7)
    X_weekly = make_fourier_features(np.arange(T), period=7, n_terms=3)
    beta_weekly = pm.Normal("beta_weekly", 0, 0.5, shape=6)

    # Yearly (period=365.25)
    X_yearly = make_fourier_features(np.arange(T), period=365.25, n_terms=6)
    beta_yearly = pm.Normal("beta_yearly", 0, 0.5, shape=12)

    seasonal = X_weekly @ beta_weekly + X_yearly @ beta_yearly

    # Trend
    sigma_trend = pm.HalfNormal("sigma_trend", 0.1)
    trend = pm.GaussianRandomWalk("trend", sigma=sigma_trend, steps=T - 1, dims="time")

    mu = trend + seasonal
    sigma_obs = pm.HalfNormal("sigma_obs", 0.5)
    y = pm.Normal("y", mu=mu, sigma=sigma_obs, observed=y_obs, dims="time")
```

## GPs for Time Series

### GP with Temporal Kernel

```python
with pm.Model(coords={"time": range(T)}) as gp_ts:
    t = np.arange(T)[:, None]

    ell = pm.InverseGamma("ell", alpha=5, beta=5)
    eta = pm.HalfNormal("eta", sigma=2)

    cov = eta**2 * pm.gp.cov.Matern52(1, ls=ell)

    # Use HSGP for T > 500
    gp = pm.gp.Latent(cov_func=cov)
    f = gp.prior("f", X=t)

    sigma = pm.HalfNormal("sigma", 0.5)
    y = pm.Normal("y", mu=f, sigma=sigma, observed=y_obs, dims="time")
```

### GP with Trend + Periodic

```python
with pm.Model() as gp_decomposed:
    t = np.arange(T)[:, None]

    # Long-term trend
    ell_trend = pm.InverseGamma("ell_trend", 5, 50)
    eta_trend = pm.HalfNormal("eta_trend", 2)
    cov_trend = eta_trend**2 * pm.gp.cov.Matern52(1, ls=ell_trend)

    # Periodic (locally periodic = periodic * decay)
    period = 365
    ell_periodic = pm.InverseGamma("ell_periodic", 5, 5)
    eta_periodic = pm.HalfNormal("eta_periodic", 1)
    ell_decay = pm.InverseGamma("ell_decay", 5, 100)

    cov_periodic = (
        eta_periodic**2
        * pm.gp.cov.Periodic(1, period=period, ls=ell_periodic)
        * pm.gp.cov.Matern52(1, ls=ell_decay)
    )

    cov_total = cov_trend + cov_periodic

    # HSGP for efficiency
    gp = pm.gp.HSGP(m=[30], c=1.5, cov_func=cov_total)
    f = gp.prior("f", X=t)

    sigma = pm.HalfNormal("sigma", 0.5)
    y = pm.Normal("y", mu=f, sigma=sigma, observed=y_obs)
```

## Handling Seasonality

### Fourier Terms (Efficient)

```python
def make_fourier_features(t, period, n_terms):
    features = []
    for k in range(1, n_terms + 1):
        features.append(np.sin(2 * np.pi * k * t / period))
        features.append(np.cos(2 * np.pi * k * t / period))
    return np.column_stack(features)

# Number of terms controls flexibility
# More terms = more flexible but risk overfitting
# Rule of thumb: n_terms = period / 2 is max, usually much less needed
```

### Dummy Variables (Interpretable)

```python
with pm.Model(coords={"dow": range(7)}) as dummy_seasonal:
    # Day-of-week effects (sum-to-zero)
    dow_raw = pm.Normal("dow_raw", 0, 1, shape=6)
    dow_effect = pm.Deterministic(
        "dow_effect",
        pt.concatenate([dow_raw, -dow_raw.sum(keepdims=True)]),
        dims="dow",
    )

    seasonal = dow_effect[day_of_week_idx]
```

## Forecasting

### Using pm.Data for Out-of-Sample

```python
with pm.Model(coords={"time": range(T_train)}) as forecast_model:
    t_data = pm.Data("t", np.arange(T_train))
    X_data = pm.Data("X_seasonal", X_seasonal_train)

    sigma_trend = pm.HalfNormal("sigma_trend", 0.1)
    trend = pm.GaussianRandomWalk("trend", sigma=sigma_trend, steps=T_train - 1)

    beta = pm.Normal("beta", 0, 1, shape=X_seasonal_train.shape[1])
    seasonal = X_data @ beta

    sigma_obs = pm.HalfNormal("sigma_obs", 0.5)
    y = pm.Normal("y", mu=trend + seasonal, sigma=sigma_obs, observed=y_train)

    idata = pm.sample()

# For forecasting, need to extend the random walk
# This requires careful handling - see PyMC docs on forecasting
```

### GP Conditional Predictions

```python
with gp_model:
    t_future = np.arange(T, T + horizon)[:, None]
    f_future = gp.conditional("f_future", Xnew=t_future)
    y_future = pm.Normal("y_future", mu=f_future, sigma=sigma)

    forecast = pm.sample_posterior_predictive(idata, var_names=["f_future", "y_future"])
```
