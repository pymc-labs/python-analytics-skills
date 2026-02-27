"""Generate/prepare datasets for bayesian-workflow benchmark tasks.

Datasets:
  W1: Minnesota radon (srrs2 from radon package / bundled)
  W2: Simulated dose-response (known ED50=5.0, Hill slope=2)
  W3: Simulated changepoint time series (break at day 365)
  W4: Simulated observational data with confounding (true ATE=2.0)
  W5: Simulated sparse regression (n=200, p=50, 5 active predictors)
"""

from pathlib import Path

import numpy as np
import polars as pl

DATA_DIR = Path(__file__).parent.parent / "data"
RNG = np.random.default_rng(20260219)


def prepare_radon():
    """W1: Minnesota radon dataset.

    Simulates data matching the classic srrs2 structure:
    ~900 observations across 85 counties, with floor and uranium predictors.
    """
    n_counties = 85
    county_names = [f"county_{i:02d}" for i in range(n_counties)]

    # County-level uranium (log scale)
    log_uranium = RNG.normal(0.0, 0.8, n_counties)

    # County-level intercepts driven by uranium
    county_intercepts = 1.5 + 0.7 * log_uranium + RNG.normal(0, 0.3, n_counties)

    # Generate observations
    records = []
    for i, county in enumerate(county_names):
        n_obs = RNG.integers(5, 20)
        for _ in range(n_obs):
            floor = RNG.choice([0, 1], p=[0.65, 0.35])
            # Floor effect: basement is ~0.7 higher on log scale
            floor_effect = -0.7 * floor
            log_radon = county_intercepts[i] + floor_effect + RNG.normal(0, 0.6)
            records.append({
                "county": county,
                "floor": floor,
                "log_radon": round(log_radon, 4),
                "log_uranium": round(log_uranium[i], 4),
            })

    df = pl.DataFrame(records)
    df.write_csv(DATA_DIR / "radon.csv")
    print(f"  radon.csv: {len(df)} observations, {n_counties} counties")


def prepare_dose_response():
    """W2: Simulated dose-response data.

    5 dose levels, 30 subjects per dose, binary outcome.
    True ED50=5.0, Hill slope=2.0.
    """
    doses = [0.0, 2.5, 5.0, 10.0, 20.0]
    n_per_dose = 30

    # 4-parameter Hill equation: P(response) = bottom + (top - bottom) / (1 + (ED50/dose)^slope)
    ed50 = 5.0
    hill_slope = 2.0
    bottom = 0.05  # baseline response rate
    top = 0.90  # max response rate

    records = []
    for dose in doses:
        if dose == 0:
            prob = bottom
        else:
            prob = bottom + (top - bottom) / (1.0 + (ed50 / dose) ** hill_slope)

        responses = RNG.binomial(1, prob, n_per_dose)
        for r in responses:
            records.append({"dose": dose, "response": int(r)})

    df = pl.DataFrame(records)
    df.write_csv(DATA_DIR / "dose_response.csv")
    print(f"  dose_response.csv: {len(df)} observations, {len(doses)} dose levels")


def prepare_timeseries():
    """W3: Simulated changepoint time series.

    730 days (2 years), changepoint at day 365.
    Before: level=10, slope=0.01, noise_sd=1.5
    After: level=15, slope=-0.005, noise_sd=2.5
    """
    n_days = 730
    changepoint = 365

    days = np.arange(1, n_days + 1)
    y = np.zeros(n_days)

    # Before changepoint
    before = days[:changepoint]
    y[:changepoint] = 10.0 + 0.01 * before + RNG.normal(0, 1.5, changepoint)

    # After changepoint (level shift + slope change + variance change)
    after = days[changepoint:] - changepoint
    y[changepoint:] = 15.0 - 0.005 * after + RNG.normal(0, 2.5, n_days - changepoint)

    df = pl.DataFrame({"day": days.tolist(), "y": [round(v, 4) for v in y]})
    df.write_csv(DATA_DIR / "timeseries.csv")
    print(f"  timeseries.csv: {n_days} days, changepoint at day {changepoint}")


def prepare_observational():
    """W4: Simulated observational data with known confounding structure.

    DAG: age -> treatment, age -> outcome
         income -> treatment, income -> outcome
         treatment -> outcome (true ATE = 2.0)
         health_score is a mediator: treatment -> health_score -> outcome
         education and prior_condition are additional confounders.

    N=500 observations.
    """
    n = 500

    # Confounders
    age = RNG.normal(50, 12, n)
    income = RNG.normal(50000, 15000, n)
    education = RNG.choice([1, 2, 3, 4], n, p=[0.15, 0.30, 0.35, 0.20])
    prior_condition = RNG.binomial(1, 0.3, n)

    # Treatment assignment (confounded by age, income, education)
    logit_treat = (
        -2.0
        + 0.03 * (age - 50)
        + 0.00002 * (income - 50000)
        + 0.3 * (education - 2)
        - 0.5 * prior_condition
    )
    prob_treat = 1.0 / (1.0 + np.exp(-logit_treat))
    treatment = RNG.binomial(1, prob_treat)

    # Health score (mediator: affected by treatment)
    health_score = 70 + 5 * treatment + 0.1 * age + RNG.normal(0, 8, n)

    # Outcome (true ATE = 2.0)
    outcome = (
        10.0
        + 2.0 * treatment  # true causal effect
        + 0.15 * (age - 50)
        + 0.0001 * (income - 50000)
        + 0.5 * (education - 2)
        - 1.5 * prior_condition
        + RNG.normal(0, 3, n)
    )

    df = pl.DataFrame({
        "treatment": treatment.tolist(),
        "outcome": [round(v, 4) for v in outcome],
        "age": [round(v, 2) for v in age],
        "income": [round(v, 2) for v in income],
        "health_score": [round(v, 2) for v in health_score],
        "education": education.tolist(),
        "prior_condition": prior_condition.tolist(),
    })
    df.write_csv(DATA_DIR / "observational.csv")
    print(f"  observational.csv: {n} observations, true ATE=2.0")


def prepare_sparse_regression():
    """W5: Simulated sparse regression data.

    N=200, P=50 predictors. Only x1-x5 are truly active.
    True coefficients: [3.0, -2.0, 1.5, -1.0, 0.5, 0, 0, ..., 0]
    """
    n = 200
    p = 50

    X = RNG.normal(0, 1, (n, p))

    # True coefficients: only first 5 are nonzero
    true_beta = np.zeros(p)
    true_beta[:5] = [3.0, -2.0, 1.5, -1.0, 0.5]

    y = X @ true_beta + RNG.normal(0, 2.0, n)

    data = {"y": [round(v, 4) for v in y]}
    for j in range(p):
        data[f"x{j+1}"] = [round(v, 4) for v in X[:, j]]

    df = pl.DataFrame(data)
    df.write_csv(DATA_DIR / "sparse_regression.csv")
    print(f"  sparse_regression.csv: {n} observations, {p} predictors, 5 active")


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print("Preparing datasets for bayesian-workflow benchmark:")
    prepare_radon()
    prepare_dose_response()
    prepare_timeseries()
    prepare_observational()
    prepare_sparse_regression()
    print("Done.")


if __name__ == "__main__":
    main()
