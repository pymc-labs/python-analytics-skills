"""Prepare benchmark datasets from raw data files.

Produces cleaned/subsampled versions suitable for fast MCMC sampling:
- GSS 2022: select relevant columns, drop nulls (~487 rows)
- S&P 500: subsample to ~750 rows of log returns
- Mixture: synthetic 3-component Gaussian mixture (500 obs)
"""

from pathlib import Path

import numpy as np
import polars as pl

DATA_DIR = Path(__file__).parent.parent / "data"
SP500_SOURCE = Path.home() / "repos" / "pymc-examples" / "examples" / "data" / "SP500.csv"
SEED = 42


def prepare_gss():
    """Clean GSS 2022 data for T2 (ordinal) and T5 (horseshoe) tasks."""
    raw = pl.read_csv(DATA_DIR / "gss_2022.csv")
    columns = [
        "age", "sex", "satjob", "hlthdep", "stress", "feelnerv",
        "worry", "wrkmeangfl", "richwork", "satfin", "realrinc",
        "anxiety", "hours_worked",
    ]
    clean = (
        raw.select(columns)
        .drop_nulls()
        .sort("age")
    )
    out = DATA_DIR / "gss_2022_clean.csv"
    clean.write_csv(out)
    print(f"GSS: {len(raw)} -> {len(clean)} rows -> {out}")


def prepare_sp500():
    """Subsample S&P 500 returns for T3 (stochastic volatility) task."""
    raw = pl.read_csv(SP500_SOURCE)
    # Subsample to ~750 rows (roughly 3 years of trading days)
    sub = raw.head(750).select([
        pl.col("Date").alias("date"),
        pl.col("change").alias("returns"),
    ])
    out = DATA_DIR / "sp500_returns.csv"
    sub.write_csv(out)
    print(f"S&P 500: {len(raw)} -> {len(sub)} rows -> {out}")


def prepare_mixture():
    """Generate synthetic 3-component Gaussian mixture for T4 (mixture) task."""
    rng = np.random.default_rng(SEED)
    centers = [-5.0, 0.0, 5.0]
    sds = [0.5, 2.0, 0.75]
    n = 500
    # Uniform component assignment
    components = rng.integers(0, len(centers), size=n)
    x = np.array([rng.normal(centers[c], sds[c]) for c in components])
    out = DATA_DIR / "synthetic" / "mixture_data.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"x": x}).write_csv(out)
    print(f"Mixture: {n} observations (3 components) -> {out}")


def main():
    print("Preparing benchmark datasets...")
    prepare_gss()
    prepare_sp500()
    prepare_mixture()
    print("Done.")


if __name__ == "__main__":
    main()
