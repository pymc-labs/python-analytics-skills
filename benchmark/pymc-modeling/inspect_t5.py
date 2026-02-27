"""Inspect T5 horseshoe InferenceData files."""
import arviz as az
import numpy as np

BASE = "results/runs"
files = {
    "no_skill": f"{BASE}/T5_horseshoe_no_skill_rep1/results.nc",
    "with_skill": f"{BASE}/T5_horseshoe_with_skill_rep1/results.nc",
}

for label, path in files.items():
    print("=" * 70)
    print(f"  {label}: {path}")
    print("=" * 70)

    idata = az.from_netcdf(path)

    # 1. Groups
    groups = list(idata.groups())
    print(f"\nGroups: {groups}")

    # 2. Posterior variables
    if hasattr(idata, "posterior"):
        post = idata.posterior
        var_names = list(post.data_vars)
        print(f"\nPosterior variables: {var_names}")

        # 3. Shapes of all variables
        print("\nPosterior variable shapes:")
        for v in var_names:
            print(f"  {v}: {dict(post[v].sizes)}")

        # 4. Coords
        print(f"\nPosterior coords:")
        for c, vals in post.coords.items():
            v = vals.values
            if v.size <= 30:
                print(f"  {c}: {v}")
            else:
                print(f"  {c}: shape={v.shape}, first 10={v[:10]}")

        # 5. Beta coefficients summary
        beta_candidates = [v for v in var_names if "beta" in v.lower() or "coef" in v.lower() or "b" == v.lower()]
        if not beta_candidates:
            beta_candidates = [v for v in var_names if post[v].ndim > 2]
            if not beta_candidates:
                beta_candidates = var_names[:3]

        print(f"\nSummary for coefficient variables: {beta_candidates}")
        try:
            summary = az.summary(idata, var_names=beta_candidates, kind="stats",
                                 hdi_prob=0.94)
            print(summary.to_string())
        except Exception as e:
            print(f"  Summary failed: {e}")
            for v in beta_candidates:
                arr = post[v].values
                flat = arr.reshape(-1, *arr.shape[2:]) if arr.ndim > 2 else arr.reshape(-1)
                print(f"\n  {v}: mean={np.mean(flat, axis=0)}, sd={np.std(flat, axis=0)}")
    else:
        print("\n  NO posterior group!")

    # 6. Observed data
    if hasattr(idata, "observed_data"):
        obs = idata.observed_data
        print(f"\nObserved data variables: {list(obs.data_vars)}")
        for v in obs.data_vars:
            arr = obs[v].values
            print(f"  {v}: shape={arr.shape}, dtype={arr.dtype}, "
                  f"min={np.nanmin(arr):.4f}, max={np.nanmax(arr):.4f}, "
                  f"mean={np.nanmean(arr):.4f}")
    else:
        print("\n  NO observed_data group.")

    # 7. Log likelihood
    if hasattr(idata, "log_likelihood"):
        ll = idata.log_likelihood
        print(f"\nLog-likelihood variables: {list(ll.data_vars)}")
    else:
        print("\n  NO log_likelihood group.")

    # 8. Sample stats
    if hasattr(idata, "sample_stats"):
        ss = idata.sample_stats
        print(f"\nSample stats variables: {list(ss.data_vars)}")
        if "diverging" in ss:
            div = ss["diverging"].values.sum()
            total = ss["diverging"].values.size
            print(f"  Divergences: {div}/{total}")

    print("\n")
