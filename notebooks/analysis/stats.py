import numpy as np
import pandas as pd


def _p_stars(p):
    if pd.isna(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def _paired_test_pvalue(x, y):
    """
    Paired test p-value for x vs y.
    Tries scipy t-test; if scipy unavailable, exact sign-flip permutation test
    on paired differences.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 2:
        return np.nan

    try:
        from scipy.stats import ttest_rel
        return float(ttest_rel(x, y, nan_policy="omit").pvalue)
    except Exception:
        d = x - y
        n = len(d)
        obs = abs(d.mean())
        extreme = 0
        total = 0
        for bits in range(1 << n):
            signs = np.ones(n)
            for i in range(n):
                if (bits >> i) & 1:
                    signs[i] = -1.0
            m = abs((signs * d).mean())
            total += 1
            if m >= obs - 1e-12:
                extreme += 1
        return extreme / total


def make_combined_latex_table(
    df_long: pd.DataFrame,
    n_list=(1000, 2500, 5000, 10000),
    metrics=("NDCG", "RCTR"),
    prop_order=("frequency-based", "MLPregression", "true_propensity"),
    model_order=("dr", "dm", "ips"),
    policy_strength=1.0,
    digits=4,
    include_distance_models=False,
):
    """
    Produces a single combined LaTeX table with rows grouped by n_sessions and metric,
    and columns by ips_model.
    """

    if include_distance_models:
        prop_order = tuple(list(prop_order) + ["cosine", "knn", "kmeans"])

    # --- keep only requested IPS models/settings ---
    wanted = (
        ((df_long["ips_model"].isin(["dm", "dr"])) & (df_long["plot_filter"] == False)) |
        ((df_long["ips_model"] == "ips") & (df_long["plot_filter"] == "merged"))
    )
    df = df_long.loc[wanted].copy()
    df = df[df["propensity_model"].isin(prop_order)].copy()
    df = df[df["n_sessions"].isin(n_list)].copy()
    df = df[df["metric"].isin(metrics)].copy()
    df = df[np.isclose(df["policy_strength"].astype(float), float(policy_strength))].copy()

    # --- compute paired p-values for MLP vs frequency-based (pool over tmp+random_state) ---
    wide = (
        df.pivot_table(
            index=["metric", "n_sessions", "ips_model", "tmp", "random_state"],
            columns="propensity_model",
            values="value",
            aggfunc="mean",
        )
        .reset_index()
    )

    records = []
    for (m, n, ips), g in wide.groupby(["metric", "n_sessions", "ips_model"]):
        base_vals = g.get("frequency-based", [])
        for prop_mod in prop_order:
            if prop_mod == "frequency-based":
                continue
            cmp_vals = g.get(prop_mod, [])
            p = _paired_test_pvalue(cmp_vals, base_vals)
            records.append({
                "metric": m,
                "n_sessions": n,
                "ips_model": ips,
                "propensity_model": prop_mod,
                "p_val": p
            })
    pvals = pd.DataFrame(records, columns=["metric", "n_sessions", "ips_model", "propensity_model", "p_val"])
    pvals["stars"] = pvals["p_val"].map(_p_stars)

    # --- average over tmp and random_state for the displayed means ---
    means = (
        df.groupby(["metric", "n_sessions", "ips_model", "propensity_model"], as_index=False)
        .agg(
            mean_value=("value", "mean"),
            std_value=("value", "std"),
        )
    )
    means["std_value"] = means["std_value"].fillna(0.0)

    # attach stars
    means = means.merge(pvals[["metric", "n_sessions", "ips_model", "propensity_model", "stars"]],
                        on=["metric", "n_sessions", "ips_model", "propensity_model"], how="left")
    means["cell"] = means.apply(
        lambda row: (
            ""
            if pd.isna(row["mean_value"])
            else f'{row["mean_value"]:.{digits}f} ({row["std_value"]:.{digits}f})'
        ),
        axis=1,
    )
    
    mask = means["propensity_model"] != "frequency-based"
    means.loc[mask, "cell"] = (
        means.loc[mask, "cell"] + means.loc[mask, "stars"].fillna("")
    )

    # --- build one combined LaTeX table ---
    piv = (
        means.pivot_table(
            index=["n_sessions", "metric", "propensity_model"],
            columns="ips_model",
            values="cell",
            aggfunc="first",
        )
    )

    # ensure full row grid exists & correct order
    idx = pd.MultiIndex.from_product(
        [list(n_list), list(metrics), list(prop_order)], 
        names=["n_sessions", "metric", "propensity_model"]
    )
    piv = piv.reindex(index=idx)
    
    # reindex columns to ensure order
    piv = piv.reindex(columns=list(model_order))

    # LaTeX multi-level output
    latex = piv.to_latex(
        escape=False,
        multirow=True,
        caption=(
            "Combined Table (averaged over temperatures and random seeds). "
            "Cells show mean (SD). "
            "Stars indicate paired significance vs frequency-based "
            "within each (metric, model, n)."
        ),
        label="tab:combined_means",
        na_rep="",
    )

    return latex
