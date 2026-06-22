# az.compare() — Detailed Model Comparison Reference (ArviZ 1.1)

## Function Signature

```python
az.compare(
    compare_dict,       # Dict[str, DataTree] — named models
    method="stacking",  # "stacking", "BB-pseudo-BMA", or "pseudo-BMA"
    var_name=None,      # observed variable when multiple log_likelihood variables exist
    reference=None,     # reference model for elpd_diff; default is best model
    round_to="auto",
)
```

CRITICAL: PyMC 6 returns DataTree by default, and ArviZ 1.1 is DataTree-first while still accepting idata-like inputs. `az.waic` is removed and `az.compare` only supports LOO — the old `ic=` and `scale=` arguments have been dropped.

## Basic Usage

```python
import arviz as az

# dt_* are DataTree objects from pm.sample()
comparison = az.compare({
    "linear": dt_linear,
    "quadratic": dt_quad,
    "gp": dt_gp,
})
print(comparison)
```

## Output Table Columns

| Column | Description |
|---|---|
| `rank` | Model rank (0 = best predictive accuracy) |
| `elpd` | Expected log pointwise predictive density (higher = better) |
| `p` | Effective number of parameters |
| `elpd_diff` | Difference in ELPD from best model (0 for best) |
| `weight` | Model weight (stacking or pseudo-BMA, sums to 1) |
| `se` | Standard error of ELPD estimate |
| `dse` | Standard error of ELPD difference from best model |
| `diag_elpd` | Pareto-k diagnostic issues for each model's ELPD |
| `diag_diff` | Small-data or practically-equivalent-difference diagnostics |

## Interpreting Results

### ELPD Differences

The key comparison metric is `elpd_diff` with its standard error `dse`:

```python
# Rule of thumb:
# |elpd_diff| < 4: models are practically equivalent
# |elpd_diff| > 4 and |elpd_diff/dse| > 2: meaningful difference
# non-empty diag_elpd: LOO unreliable, investigate Pareto k
```

### Example Interpretation

```
              rank  elpd  p  elpd_diff  weight    se   dse  diag_elpd
quadratic        0 -145.3 4.2        0.0    0.72 12.3   0.0
gp               1 -148.1 8.7       -2.8    0.28 13.1   3.2
linear           2 -162.5 2.1      -17.2    0.00 11.8   7.5
```

Reading this:
- Quadratic model has best predictive accuracy (rank 0)
- GP is close (elpd_diff=-2.8, dse=3.2) — difference is within 1 SE, models are comparable
- Linear is clearly worse (elpd_diff=-17.2, dse=7.5) — difference is ~2.3 SE
- Stacking assigns 72% weight to quadratic, 28% to GP, 0% to linear

## Visualization

### Comparison Plot

```python
az.plot_compare(comparison)
```

Shows ELPD values with standard errors as a forest plot. Quick visual check of model ranking and uncertainty.

### ELPD Pointwise Differences

```python
az.plot_elpd({"linear": dt_linear, "quadratic": dt_quad})
```

Shows observation-level ELPD differences between models. Identifies which observations drive the model comparison. Useful for understanding where models disagree.

## Weighting Methods

### Stacking (Default)

```python
comparison = az.compare(model_dict, method="stacking")
```

Stacking minimizes the KL divergence from the mixture predictive to the true predictive. Properties:
- Optimal for prediction when the true model is not in the candidate set
- Weights can be exactly 0 for clearly inferior models
- Does not require models to be nested
- Default and recommended for most use cases

### Pseudo-BMA+ (Bayesian Bootstrap)

```python
comparison = az.compare(model_dict, method="BB-pseudo-BMA")
```

Uses Bayesian bootstrap to estimate uncertainty in the weights themselves. Properties:
- Provides a distribution over weights, not just point estimates
- More robust to outlier ELPD values
- Weights are always positive (no exact zeros)
- Useful when you want uncertainty quantification on the weights

## Handling Warnings

When `diag_elpd` is non-empty for a model:

```python
# 1. Compute LOO separately to inspect Pareto k
loo = az.loo(dt_problematic, pointwise=True)
az.plot_khat(loo)

# 2. Try moment matching
loo_mm = az.loo_moment_match(dt_problematic)

# 3. If still problematic, use k-fold
kfold = az.loo_kfold(dt_problematic, K=10)
```

Do not trust comparisons with non-empty `diag_elpd` until you have addressed the Pareto k issues.

## Comparing Nested Models

For nested models (e.g., with/without a predictor), the comparison is straightforward:

```python
comparison = az.compare({
    "full": dt_full,
    "reduced": dt_reduced,
})
# If elpd_diff is small relative to dse, the additional complexity is not justified
```

## Comparing Non-Nested Models

Works identically — LOO does not require nesting:

```python
comparison = az.compare({
    "linear": dt_linear,
    "gp": dt_gp,
    "tree": dt_tree,
})
```

## Multiple Comparisons

When comparing many models, be cautious:
- The best model by LOO may be best by chance (selection bias)
- Use stacking weights rather than picking the single best model
- Consider model averaging for predictions

## Common Pitfalls

1. **Different data**: All models must be fit to the same observations. LOO cannot compare models fit to different datasets.
2. **Missing log_likelihood**: Ensure all DataTree objects contain `log_likelihood`. In PyMC 6, call `pm.sample(idata_kwargs={"log_likelihood": True})` or `pm.compute_log_likelihood(idata, model=model)` after sampling.
3. **Ignoring diagnostics**: A model with non-empty `diag_elpd` has unreliable ELPD. Fix the Pareto k issues before comparing.
4. **Over-interpreting small differences**: If `abs(elpd_diff) < 4`, models are practically equivalent. Prefer the simpler model.
5. **Confusing ELPD with evidence**: LOO measures predictive accuracy, not posterior probability of the model being true. They answer different questions.
