"""
Generate a trace plot comparison for the blog post.

Shows label-switching in the no_skill mixture model (T4) vs clean traces
from the with_skill model that uses an ordered transform.
"""

import arviz as az
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

RESULTS = Path(__file__).resolve().parent.parent / "results" / "runs"
NO_SKILL = RESULTS / "T4_mixture_no_skill_rep0" / "results.nc"
WITH_SKILL = RESULTS / "T4_mixture_with_skill_rep0" / "results.nc"
OUT = Path(__file__).resolve().parent.parent / "figures"

# Chain colors — distinct but not garish
CHAIN_COLORS = ["#4878A8", "#E8783A", "#54A868", "#C75B7A"]


def extract_means(path, var_name):
    """Extract component-mean traces: returns array (chains, draws, K)."""
    idata = az.from_netcdf(path)
    post = idata.posterior[var_name]  # (chain, draw, component)
    return post.values


def main():
    OUT.mkdir(exist_ok=True)

    ns_means = extract_means(NO_SKILL, "means_raw")   # (4, 2000, 4)
    ws_means = extract_means(WITH_SKILL, "mu")         # (4, 1000, 3)

    n_chains_ns, n_draws_ns, K_ns = ns_means.shape
    n_chains_ws, n_draws_ws, K_ws = ws_means.shape

    # Show first 3 components for no_skill (to match with_skill's K=3)
    # Sort components by posterior mean so we compare analogous components
    ns_order = np.argsort(ns_means.mean(axis=(0, 1)))[:K_ws]
    ns_plot = ns_means[:, :, ns_order]

    fig, axes = plt.subplots(
        K_ws, 2, figsize=(10, 5.5),
        gridspec_kw={"wspace": 0.12, "hspace": 0.35},
    )
    fig.subplots_adjust(top=0.85, bottom=0.08)

    component_labels = [f"$\\mu_{{{k+1}}}$" for k in range(K_ws)]

    for k in range(K_ws):
        ax_ns = axes[k, 0]
        ax_ws = axes[k, 1]

        # No-skill traces
        for c in range(n_chains_ns):
            ax_ns.plot(
                ns_plot[c, :, k],
                color=CHAIN_COLORS[c], alpha=0.7, linewidth=0.3, rasterized=True,
            )

        # With-skill traces
        for c in range(n_chains_ws):
            ax_ws.plot(
                ws_means[c, :, k],
                color=CHAIN_COLORS[c], alpha=0.7, linewidth=0.3, rasterized=True,
            )

        # Y-axis label on left panel only
        ax_ns.set_ylabel(component_labels[k], fontsize=11, rotation=0, labelpad=15,
                         va="center")

        # Styling — each panel keeps its own y-limits so the with-skill
        # traces aren't compressed by the no-skill label-switching range
        for ax in (ax_ns, ax_ws):
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.tick_params(axis="both", labelsize=8)

        # Only bottom row gets x-label
        if k == K_ws - 1:
            ax_ns.set_xlabel("Draw", fontsize=9)
            ax_ws.set_xlabel("Draw", fontsize=9)
        else:
            ax_ns.set_xticklabels([])
            ax_ws.set_xticklabels([])

    # Column titles
    axes[0, 0].set_title("Without skill", fontsize=12, fontweight="bold", pad=8)
    axes[0, 1].set_title("With skill", fontsize=12, fontweight="bold", pad=8)

    fig.suptitle(
        "MCMC traces for mixture component means (T4)",
        fontsize=13, fontweight="bold", y=0.97,
    )

    out_path = OUT / "trace_comparison.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
