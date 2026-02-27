"""
Generate a coefficient comparison figure for the blog post.

Compares posterior coefficient estimates from the no_skill (Laplace prior)
and with_skill (regularized horseshoe) T5 benchmark runs.
"""

import arviz as az
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

RESULTS = Path(__file__).resolve().parent.parent / "results" / "runs"
NO_SKILL = RESULTS / "T5_horseshoe_no_skill_rep1" / "results.nc"
WITH_SKILL = RESULTS / "T5_horseshoe_with_skill_rep1" / "results.nc"
OUT = Path(__file__).resolve().parent.parent / "figures"

PREDICTORS = [
    "age", "sex", "hlthdep", "stress", "feelnerv", "worry",
    "wrkmeangfl", "richwork", "realrinc", "anxiety", "hours_worked",
]

# Nicer display names
LABELS = {
    "age": "Age",
    "sex": "Sex",
    "hlthdep": "Health limits\ndaily activities",
    "stress": "Stress",
    "feelnerv": "Feel nervous",
    "worry": "Worry",
    "wrkmeangfl": "Work\nmeaningful",
    "richwork": "Rich from\nwork",
    "realrinc": "Real income",
    "anxiety": "Anxiety",
    "hours_worked": "Hours worked",
}


def load_betas(path, standardize_y=False):
    """Load beta posteriors and return (samples, mean, hdi_lo, hdi_hi)."""
    idata = az.from_netcdf(path)
    # Shape: (chains, draws, features)
    beta = idata.posterior["beta"].values
    flat = beta.reshape(-1, beta.shape[-1])  # (n_samples, n_features)

    if standardize_y:
        # no_skill model didn't standardize y — divide by y_std to get
        # comparable effect sizes (change per SD of predictor, in SD of y)
        y_obs = idata.observed_data["y_obs"].values
        y_std = y_obs.std()
        flat = flat / y_std

    means = flat.mean(axis=0)
    hdi = np.array([az.hdi(flat[:, i], hdi_prob=0.95) for i in range(flat.shape[1])])
    return flat, means, hdi[:, 0], hdi[:, 1]


def main():
    OUT.mkdir(exist_ok=True)

    # Load both sets of betas — standardize no_skill to put on comparable scale
    ns_flat, ns_mean, ns_lo, ns_hi = load_betas(NO_SKILL, standardize_y=True)
    ws_flat, ws_mean, ws_lo, ws_hi = load_betas(WITH_SKILL, standardize_y=False)

    n = len(PREDICTORS)
    labels = [LABELS[p] for p in PREDICTORS]

    # Sort by with_skill absolute coefficient magnitude (largest at top)
    order = np.argsort(np.abs(ws_mean))  # ascending — plotted bottom to top
    labels = [labels[i] for i in order]

    ns_mean, ns_lo, ns_hi = ns_mean[order], ns_lo[order], ns_hi[order]
    ws_mean, ws_lo, ws_hi = ws_mean[order], ws_lo[order], ws_hi[order]

    # ── Figure ──
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(10, 5.2), sharey=True,
        gridspec_kw={"wspace": 0.08},
    )
    fig.subplots_adjust(top=0.86, bottom=0.12)

    y_pos = np.arange(n)
    dot_size = 45

    # Color scheme
    c_ns = "#8B7355"  # muted brown
    c_ws = "#2E6B8A"  # teal
    c_zero = "#D5D5D5"

    for ax, means, lo, hi, color, title, prior_label in [
        (ax1, ns_mean, ns_lo, ns_hi, c_ns, "Without skill", "Laplace (LASSO) prior"),
        (ax2, ws_mean, ws_lo, ws_hi, c_ws, "With skill", "Regularized horseshoe prior"),
    ]:
        # Zero reference line
        ax.axvline(0, color=c_zero, linewidth=0.8, zorder=0)

        # HDI bars
        for i in range(n):
            excludes_zero = lo[i] > 0 or hi[i] < 0
            alpha = 1.0 if excludes_zero else 0.35
            lw = 2.2 if excludes_zero else 1.0

            ax.plot([lo[i], hi[i]], [y_pos[i], y_pos[i]],
                    color=color, linewidth=lw, alpha=alpha, solid_capstyle="round")
            ax.scatter(means[i], y_pos[i], color=color, s=dot_size,
                       zorder=5, alpha=alpha, edgecolors="white", linewidths=0.5)

        # Title + prior label combined with proper spacing
        ax.set_title(f"{title}\n", fontsize=12, fontweight="bold", pad=4)
        ax.text(0.5, 1.005, prior_label, transform=ax.transAxes,
                fontsize=9, color="#888888", ha="center", va="bottom",
                fontstyle="italic")

        ax.set_xlabel("Standardized coefficient", fontsize=9.5, labelpad=6)
        ax.set_yticks(y_pos)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="both", labelsize=9)

    ax1.set_yticklabels(labels, fontsize=9)
    ax2.tick_params(axis="y", length=0)

    # Match x-axis limits across panels
    x_min = min(ax1.get_xlim()[0], ax2.get_xlim()[0])
    x_max = max(ax1.get_xlim()[1], ax2.get_xlim()[1])
    margin = (x_max - x_min) * 0.05
    for ax in (ax1, ax2):
        ax.set_xlim(x_min - margin, x_max + margin)

    fig.suptitle(
        "Posterior coefficient estimates — sparse variable selection (T5)",
        fontsize=13, fontweight="bold", y=0.97,
    )
    fig.text(
        0.5, 0.02,
        "Points = posterior means; bars = 95% HDI. "
        "Full opacity = HDI excludes zero (selected). "
        "Coefficients standardized to effect per SD of predictor.",
        ha="center", fontsize=8, color="#999999",
    )

    out_path = OUT / "coefficient_comparison.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Saved to {out_path}")

    # Print summary for blog narrative
    print("\n── Summary ──")
    for name, means, lo, hi, label in [
        ("no_skill", ns_mean, ns_lo, ns_hi, "Laplace"),
        ("with_skill", ws_mean, ws_lo, ws_hi, "Horseshoe"),
    ]:
        selected = sum(1 for i in range(n) if lo[i] > 0 or hi[i] < 0)
        print(f"\n{label}: {selected}/{n} predictors selected (HDI excludes zero)")
        for i in range(n):
            pred = PREDICTORS[order[i]]
            sel = "*" if lo[i] > 0 or hi[i] < 0 else " "
            print(f"  {sel} {pred:15s}  mean={means[i]:+.3f}  95% HDI [{lo[i]:+.3f}, {hi[i]:+.3f}]")


if __name__ == "__main__":
    main()
