"""Prepare benchmark datasets from raw data files.

Produces cleaned/subsampled versions suitable for fast MCMC sampling:
- GSS 2022: select relevant columns, drop nulls (~487 rows)
- Mauna Loa CO2: keep every other row (~396 rows)
- Regression comparison: copy as-is (150 rows)
"""

from pathlib import Path

import polars as pl

DATA_DIR = Path(__file__).parent.parent / "data"
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


def prepare_mauna_loa():
    """Subsample Mauna Loa CO2 data for T4 (GP) task."""
    raw = pl.read_csv(DATA_DIR / "mauna_loa_co2.csv")
    # Keep every other row
    sub = raw.gather_every(2)
    out = DATA_DIR / "mauna_loa_co2.csv"
    sub.write_csv(out)
    print(f"Mauna Loa: {len(raw)} -> {len(sub)} rows -> {out} (overwritten)")


def prepare_regression():
    """Copy regression comparison data as-is for T3 (model comparison) task."""
    raw = pl.read_csv(DATA_DIR / "synthetic" / "regression_comparison.csv")
    # Already small enough, just verify
    out = DATA_DIR / "synthetic" / "regression_comparison.csv"
    print(f"Regression: {len(raw)} rows (no change) -> {out}")


def main():
    print("Preparing benchmark datasets...")
    prepare_gss()
    prepare_mauna_loa()
    prepare_regression()
    print("Done.")


if __name__ == "__main__":
    main()
