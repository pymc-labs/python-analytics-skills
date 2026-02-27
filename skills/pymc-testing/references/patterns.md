# Testing Patterns

Real-world examples:
- **pymc-marketing tests**: https://github.com/pymc-labs/pymc-marketing/tree/main/tests

## Basic Model Structure Test

```python
# test_model.py
import pytest
import pymc as pm
from pymc.testing import mock_sample_setup_and_teardown

mock_pymc_sample = pytest.fixture(scope="function")(mock_sample_setup_and_teardown)

def test_linear_regression_model_runs(mock_pymc_sample):
    """Test that a linear regression model runs without errors."""
    import numpy as np
    
    # Fake data
    np.random.seed(42)
    X = np.random.randn(100, 3)
    y = X @ [1.0, 2.0, -0.5] + np.random.randn(100) * 0.5
    
    with pm.Model(coords={"feature": ["a", "b", "c"]}) as model:
        beta = pm.Normal("beta", mu=0, sigma=1, dims="feature")
        sigma = pm.HalfNormal("sigma", sigma=1)
        mu = X @ beta
        y = pm.Normal("y", mu=mu, sigma=sigma, observed=y)
        
        idata = pm.sample(draws=10)
    
    # Verify structure (not values!)
    assert "beta" in idata.posterior
    assert "sigma" in idata.posterior
    assert idata.posterior["beta"].shape == (1, 10, 3)  # chains, draws, dims
```

## Testing Downstream Code

When testing code that consumes InferenceData (plotting, serialization):

```python
# conftest.py
import pytest
from functools import partial
import numpy as np
from pymc.testing import mock_sample

def mock_diverging(size):
    return np.zeros(size, dtype=int)

mock_sample_fast = partial(
    mock_sample,
    draws=10,
    sample_stats={"diverging": mock_diverging},
)

@pytest.fixture(scope="function")
def mock_pymc_sample():
    import pymc as pm
    original = pm.sample
    pm.sample = mock_sample_fast
    yield
    pm.sample = original
```

## Testing Plotting Functions

```python
# test_plotting.py
def test_plot_posterior_works(mock_pymc_sample):
    import arviz as az
    
    with pm.Model() as model:
        mu = pm.Normal("mu", 0, 1)
        sigma = pm.HalfNormal("sigma", 1)
        y = pm.Normal("y", mu, sigma, observed=[1.0, 2.0, 3.0])
        
        idata = pm.sample()
    
    # Test plotting - just check it doesn't error
    az.plot_posterior(idata, var_names=["mu"])  # No error = pass
```

## Testing Serialization

```python
# test_serialization.py
def test_model_save_load(mock_pymc_sample):
    import tempfile
    import arviz as az
    
    with pm.Model() as model:
        pm.Normal("x", 0, 1)
        idata = pm.sample()
    
    with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as f:
        fname = f.name
    
    idata.to_netcdf(fname)
    loaded = az.from_netcdf(fname)
    
    assert "x" in loaded.posterior
```

## CI/CD Integration

For GitHub Actions or other CI systems, use mock sampling to keep tests fast:

```yaml
# .github/workflows/test.yml
- name: Test with mock sampling
  run: |
    pytest tests/ -m "not slow"  # Only fast mock tests
    
- name: Full integration tests
  run: |
    pytest tests/ -m "slow"  # Real sampling, scheduled nightly
```

## Marking Tests

```python
import pytest

@pytest.mark.slow
def test_posterior_convergence():
    """Real sampling test - only run on main branch."""
    import pymc as pm
    
    with pm.Model() as model:
        pm.Normal("x", 0, 1)
        idata = pm.sample(nuts_sampler="nutpie", draws=1000)
    
    # Check real convergence
    assert az.summary(idata)["ess_bulk"].min() > 400
```

## When NOT to Use Mocking

These require real sampling:
- Checking posterior means are close to true values
- Testing ESS, r_hat, divergences
- LOO-CV or WAIC model comparison
- Prior/posterior predictive checks (calibration)
- Testing sampler-specific behavior (e.g., nutpie warmup)

For real sampling tests, see the [pymc-modeling skill](../pymc-modeling/SKILL.md).
