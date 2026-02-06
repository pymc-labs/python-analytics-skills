"""Phase 3: Extract real MCMC diagnostics from InferenceData files.

Loads actual .nc files produced by executed scripts and computes
ArviZ diagnostics — the core differentiator from the prior art
benchmark which used regex on generated code.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import arviz as az
import numpy as np

BENCHMARK_DIR = Path(__file__).resolve().parent.parent
EXEC_DIR = BENCHMARK_DIR / "results" / "execution"
DIAG_DIR = BENCHMARK_DIR / "results" / "diagnostics"


def compute_diagnostics(idata_path: Path) -> dict:
    """Compute MCMC diagnostics from an InferenceData file.

    Returns a dict with convergence, sampling, model structure,
    and statistical quality metrics.
    """
    diagnostics: dict = {
        "idata_path": str(idata_path),
        "load_success": False,
        "error": None,
    }

    try:
        idata = az.from_netcdf(str(idata_path))
        diagnostics["load_success"] = True
    except Exception as e:
        diagnostics["error"] = f"Failed to load InferenceData: {e}"
        return diagnostics

    # Model structure
    diagnostics["model_structure"] = _extract_structure(idata)

    # Convergence diagnostics
    if hasattr(idata, "posterior") and idata.posterior is not None:
        diagnostics["convergence"] = _compute_convergence(idata)
    else:
        diagnostics["convergence"] = {"error": "No posterior group"}

    # Sampling diagnostics
    if hasattr(idata, "sample_stats") and idata.sample_stats is not None:
        diagnostics["sampling"] = _compute_sampling_stats(idata)
    else:
        diagnostics["sampling"] = {"error": "No sample_stats group"}

    return diagnostics


def _extract_structure(idata) -> dict:
    """Extract model structure information from InferenceData."""
    structure = {
        "has_posterior": hasattr(idata, "posterior") and idata.posterior is not None,
        "has_prior": hasattr(idata, "prior") and idata.prior is not None,
        "has_prior_predictive": (
            hasattr(idata, "prior_predictive") and idata.prior_predictive is not None
        ),
        "has_posterior_predictive": (
            hasattr(idata, "posterior_predictive")
            and idata.posterior_predictive is not None
        ),
        "has_log_likelihood": (
            hasattr(idata, "log_likelihood") and idata.log_likelihood is not None
        ),
        "has_sample_stats": (
            hasattr(idata, "sample_stats") and idata.sample_stats is not None
        ),
        "has_observed_data": (
            hasattr(idata, "observed_data") and idata.observed_data is not None
        ),
    }

    if structure["has_posterior"]:
        posterior = idata.posterior
        structure["n_chains"] = int(posterior.dims.get("chain", 0))
        structure["n_draws"] = int(posterior.dims.get("draw", 0))
        structure["variables"] = list(posterior.data_vars)
        structure["n_variables"] = len(structure["variables"])

        # Count total scalar parameters
        n_params = 0
        for var in posterior.data_vars.values():
            n_params += int(
                np.prod(
                    [
                        d
                        for dim, d in zip(var.dims, var.shape)
                        if dim not in ("chain", "draw")
                    ]
                )
            )
        structure["n_parameters"] = n_params

    return structure


def _compute_convergence(idata) -> dict:
    """Compute convergence diagnostics from posterior."""
    convergence: dict = {}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            summary = az.summary(idata, kind="diagnostics")
        except Exception as e:
            return {"error": f"az.summary failed: {e}"}

    # R-hat
    if "r_hat" in summary.columns:
        rhat = summary["r_hat"].dropna()
        if len(rhat) > 0:
            convergence["r_hat_max"] = round(float(rhat.max()), 4)
            convergence["r_hat_mean"] = round(float(rhat.mean()), 4)
            convergence["r_hat_all_below_1_01"] = bool(rhat.max() < 1.01)
            convergence["r_hat_all_below_1_05"] = bool(rhat.max() < 1.05)

    # ESS bulk
    if "ess_bulk" in summary.columns:
        ess_bulk = summary["ess_bulk"].dropna()
        if len(ess_bulk) > 0:
            convergence["ess_bulk_min"] = round(float(ess_bulk.min()), 1)
            convergence["ess_bulk_median"] = round(float(ess_bulk.median()), 1)
            convergence["ess_bulk_all_above_400"] = bool(ess_bulk.min() > 400)

    # ESS tail
    if "ess_tail" in summary.columns:
        ess_tail = summary["ess_tail"].dropna()
        if len(ess_tail) > 0:
            convergence["ess_tail_min"] = round(float(ess_tail.min()), 1)
            convergence["ess_tail_median"] = round(float(ess_tail.median()), 1)
            convergence["ess_tail_all_above_400"] = bool(ess_tail.min() > 400)

    # Overall convergence assessment
    converged = (
        convergence.get("r_hat_all_below_1_01", False)
        and convergence.get("ess_bulk_all_above_400", False)
        and convergence.get("ess_tail_all_above_400", False)
    )
    convergence["converged"] = converged

    return convergence


def _compute_sampling_stats(idata) -> dict:
    """Compute sampling diagnostics from sample_stats."""
    stats: dict = {}
    sample_stats = idata.sample_stats

    # Divergences
    if "diverging" in sample_stats:
        diverging = sample_stats["diverging"].values
        n_div = int(diverging.sum())
        total = int(diverging.size)
        stats["n_divergences"] = n_div
        stats["pct_divergences"] = round(100 * n_div / total, 2) if total > 0 else 0.0

    # Tree depth
    if "tree_depth" in sample_stats:
        tree_depth = sample_stats["tree_depth"].values
        if "max_treedepth" in sample_stats.attrs:
            max_td = sample_stats.attrs["max_treedepth"]
        else:
            max_td = 10  # default
        n_maxed = int((tree_depth >= max_td).sum())
        total = int(tree_depth.size)
        stats["max_tree_depth_hits"] = n_maxed
        stats["pct_max_tree_depth"] = (
            round(100 * n_maxed / total, 2) if total > 0 else 0.0
        )

    # Energy (BFMI)
    if "energy" in sample_stats:
        try:
            bfmi = az.bfmi(idata)
            if hasattr(bfmi, "__iter__"):
                stats["energy_bfmi_min"] = round(float(min(bfmi)), 4)
                stats["energy_bfmi_mean"] = round(float(np.mean(bfmi)), 4)
                stats["energy_bfmi_all_above_0_3"] = bool(min(bfmi) > 0.3)
            else:
                stats["energy_bfmi_min"] = round(float(bfmi), 4)
                stats["energy_bfmi_all_above_0_3"] = bool(bfmi > 0.3)
        except Exception:
            pass

    # Step size
    if "step_size" in sample_stats:
        step_sizes = sample_stats["step_size"].values
        stats["mean_step_size"] = round(float(step_sizes.mean()), 6)

    return stats


def diagnose_result(
    task_id: str,
    condition: str,
    replication: int,
) -> dict:
    """Compute diagnostics for a single result."""
    DIAG_DIR.mkdir(parents=True, exist_ok=True)

    nc_name = f"{task_id}_{condition}_rep{replication}.nc"
    nc_path = EXEC_DIR / nc_name

    result = {
        "task_id": task_id,
        "condition": condition,
        "replication": replication,
    }

    if not nc_path.exists():
        result["error"] = f"InferenceData not found: {nc_path}"
        result["load_success"] = False
    else:
        diag = compute_diagnostics(nc_path)
        result.update(diag)

    # Save
    diag_name = f"{task_id}_{condition}_rep{replication}.json"
    diag_path = DIAG_DIR / diag_name
    with open(diag_path, "w") as f:
        json.dump(result, f, indent=2, default=str)

    return result


def diagnose_all() -> list[dict]:
    """Compute diagnostics for all execution results."""
    if not EXEC_DIR.exists():
        print("No execution results found. Run 'execute' first.")
        return []

    results = []
    nc_files = sorted(EXEC_DIR.glob("*.nc"))

    if not nc_files:
        print("No .nc files found in results/execution/")
        return []

    for nc_path in nc_files:
        stem = nc_path.stem
        parts = stem.split("_")
        task_id = parts[0]
        rep_idx = next(
            (j for j, p in enumerate(parts) if p.startswith("rep")), len(parts) - 1
        )
        condition = "_".join(parts[1:rep_idx])
        try:
            replication = int(parts[rep_idx].replace("rep", ""))
        except (ValueError, IndexError):
            replication = 1

        print(f"  Diagnosing {stem}...", end=" ")
        result = diagnose_result(task_id, condition, replication)

        if result.get("load_success"):
            conv = result.get("convergence", {})
            samp = result.get("sampling", {})
            status = "converged" if conv.get("converged") else "issues"
            n_div = samp.get("n_divergences", "?")
            rhat = conv.get("r_hat_max", "?")
            print(f"{status} | r_hat_max={rhat} | divergences={n_div}")
        else:
            print(f"FAIL: {result.get('error', 'unknown')}")

        results.append(result)

    success = sum(1 for r in results if r.get("load_success"))
    print(f"\nDiagnosed {success}/{len(results)} InferenceData files")
    return results
