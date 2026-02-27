"""Diagnostic comparison of no_skill vs with_skill benchmark runs."""

import arviz as az
import numpy as np
from pathlib import Path

RESULTS = Path("results/runs")

PAIRS = [
    ("T3_stochastic_volatility_no_skill_rep1", "T3_stochastic_volatility_with_skill_rep1"),
    ("T4_mixture_no_skill_rep0", "T4_mixture_with_skill_rep0"),
    ("T5_horseshoe_no_skill_rep0", "T5_horseshoe_with_skill_rep0"),
]


def diagnose(run_name: str) -> dict | None:
    nc_path = RESULTS / run_name / "results.nc"
    if not nc_path.exists():
        print(f"  [MISSING] {nc_path}")
        return None

    idata = az.from_netcdf(nc_path)

    # Divergences
    if hasattr(idata, "sample_stats") and "diverging" in idata.sample_stats:
        divs = int(idata.sample_stats["diverging"].values.sum())
    else:
        divs = "N/A"

    # Posterior variable names
    if not hasattr(idata, "posterior"):
        print(f"  [NO POSTERIOR] {run_name}")
        return None

    var_names = list(idata.posterior.data_vars)
    print(f"  Variables ({len(var_names)}): {var_names[:15]}{'...' if len(var_names) > 15 else ''}")
    print(f"  Chains: {idata.posterior.dims.get('chain', '?')}, Draws: {idata.posterior.dims.get('draw', '?')}")
    print(f"  Divergences: {divs}")

    # R-hat and ESS
    try:
        rhat = az.rhat(idata)
        ess_bulk = az.ess(idata, method="bulk")
        ess_tail = az.ess(idata, method="tail")
    except Exception as e:
        print(f"  [DIAG ERROR] {e}")
        return None

    # Collect per-variable max rhat, min ess
    worst_rhat = -1.0
    worst_rhat_var = None
    worst_rhat_val = None
    global_min_ess_bulk = float("inf")
    global_min_ess_tail = float("inf")

    for var in var_names:
        rh = rhat[var].values
        eb = ess_bulk[var].values
        et = ess_tail[var].values

        max_rh = float(np.nanmax(rh))
        min_eb = float(np.nanmin(eb))
        min_et = float(np.nanmin(et))

        if max_rh > worst_rhat:
            worst_rhat = max_rh
            worst_rhat_var = var
            idx = np.unravel_index(np.nanargmax(rh), rh.shape) if rh.ndim > 0 else ()
            worst_rhat_val = {
                "rhat": max_rh,
                "ess_bulk": float(eb[idx]) if eb.shape else float(eb),
                "ess_tail": float(et[idx]) if et.shape else float(et),
            }

        global_min_ess_bulk = min(global_min_ess_bulk, min_eb)
        global_min_ess_tail = min(global_min_ess_tail, min_et)

    print(f"  Max R-hat (global): {worst_rhat:.4f}")
    print(f"  Min ESS bulk (global): {global_min_ess_bulk:.1f}")
    print(f"  Min ESS tail (global): {global_min_ess_tail:.1f}")
    print(f"  Worst param: '{worst_rhat_var}' -- R-hat={worst_rhat_val['rhat']:.4f}, "
          f"ESS_bulk={worst_rhat_val['ess_bulk']:.1f}, ESS_tail={worst_rhat_val['ess_tail']:.1f}")

    return {
        "divergences": divs,
        "max_rhat": worst_rhat,
        "min_ess_bulk": global_min_ess_bulk,
        "min_ess_tail": global_min_ess_tail,
        "worst_var": worst_rhat_var,
        "worst_rhat_detail": worst_rhat_val,
        "n_vars": len(var_names),
        "var_names": var_names,
    }


def main():
    summary = []

    for no_skill, with_skill in PAIRS:
        task = no_skill.split("_no_skill")[0]
        print(f"\n{'='*70}")
        print(f"TASK: {task}")
        print(f"{'='*70}")

        print(f"\n--- NO SKILL: {no_skill} ---")
        d_no = diagnose(no_skill)

        print(f"\n--- WITH SKILL: {with_skill} ---")
        d_with = diagnose(with_skill)

        if d_no and d_with:
            rhat_gap = d_no["max_rhat"] - d_with["max_rhat"]
            ess_ratio = d_with["min_ess_bulk"] / max(d_no["min_ess_bulk"], 1.0)
            div_gap = (d_no["divergences"] if isinstance(d_no["divergences"], int) else 0) - \
                      (d_with["divergences"] if isinstance(d_with["divergences"], int) else 0)

            print(f"\n  >> CONTRAST: R-hat gap={rhat_gap:+.4f}, ESS ratio={ess_ratio:.2f}x, Div gap={div_gap:+d}")
            summary.append((task, rhat_gap, ess_ratio, div_gap, d_no, d_with))

    if summary:
        print(f"\n\n{'='*70}")
        print("SUMMARY -- Best task for trace plot contrast")
        print(f"{'='*70}")
        for task, rhat_gap, ess_ratio, div_gap, d_no, d_with in sorted(summary, key=lambda x: x[1] + x[2]/10, reverse=True):
            print(f"\n  {task}:")
            print(f"    no_skill   -- R-hat={d_no['max_rhat']:.4f}, ESS_bulk={d_no['min_ess_bulk']:.1f}, divs={d_no['divergences']}")
            print(f"    with_skill -- R-hat={d_with['max_rhat']:.4f}, ESS_bulk={d_with['min_ess_bulk']:.1f}, divs={d_with['divergences']}")
            print(f"    Gap: R-hat {rhat_gap:+.4f} | ESS {ess_ratio:.1f}x | Divs {div_gap:+d}")
            print(f"    Worst no_skill param: '{d_no['worst_var']}'")
            print(f"    Worst with_skill param: '{d_with['worst_var']}'")


if __name__ == "__main__":
    main()
