from typing import Optional
from pathlib import Path
import json
import re
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter
from matplotlib.lines import Line2D
from result_parsing.analysis.plot_style import apply_result_comparison_plot_style

apply_result_comparison_plot_style(plt)

# Ensure output directory exists (repo root)
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PLOTS_DIR = os.path.join(REPO_ROOT, "result_parsing/result_plots")
os.makedirs(PLOTS_DIR, exist_ok=True)


def ensure_2d_axes(axes, n_rows: int, n_cols: int):
    if n_rows == 1 and n_cols == 1:
        return np.array([[axes]])
    if n_rows == 1:
        return np.expand_dims(axes, axis=0)
    if n_cols == 1:
        return np.expand_dims(axes, axis=1)
    return axes


def get_interval_bounds(df: pd.DataFrame, interval_mode: str = "std"):
    if interval_mode not in {"std", "ci", "none"}:
        raise ValueError(f"Unsupported interval_mode: {interval_mode}")
    if interval_mode == "none":
        mean = df["mean_value"].to_numpy()
        return mean, mean
    if interval_mode == "ci" and {"ci_low_value", "ci_high_value"}.issubset(df.columns):
        low = df["ci_low_value"].fillna(df["mean_value"]).to_numpy()
        high = df["ci_high_value"].fillna(df["mean_value"]).to_numpy()
        return low, high

    mean = df["mean_value"].to_numpy()
    std = df["std_value"].fillna(0).to_numpy()
    return mean - std, mean + std


def get_baseline_interval(row: pd.Series, interval_mode: str = "std"):
    if interval_mode not in {"std", "ci", "none"}:
        raise ValueError(f"Unsupported interval_mode: {interval_mode}")
    if interval_mode == "none":
        mean = float(row["mean_value"])
        return mean, mean
    if interval_mode == "ci" and "ci_low_value" in row and pd.notna(row["ci_low_value"]):
        low = float(row["ci_low_value"])
    else:
        sd = float(row["std_value"]) if pd.notna(row["std_value"]) else 0.0
        low = float(row["mean_value"]) - sd

    if interval_mode == "ci" and "ci_high_value" in row and pd.notna(row["ci_high_value"]):
        high = float(row["ci_high_value"])
    else:
        sd = float(row["std_value"]) if pd.notna(row["std_value"]) else 0.0
        high = float(row["mean_value"]) + sd

    return low, high


def slugify_plot_part(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.strip().lower()).strip("-")
    return slug or "plot"


def latex_bold(text: str) -> str:
    escaped = (
        str(text)
        .replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("$", r"\$")
        .replace("#", r"\#")
        .replace("_", r"\_")
        .replace("{", r"\{")
        .replace("}", r"\}")
    )
    return rf"\textbf{{{escaped}}}"


def format_max_two_decimals(value, _position=None) -> str:
    if not np.isfinite(value):
        return ""
    if np.isclose(value, round(value), atol=1e-10):
        return str(int(round(value)))
    return f"{value:.2f}".rstrip("0").rstrip(".")


def format_fixed_decimals(decimals: int):
    def formatter(value, _position=None) -> str:
        if not np.isfinite(value):
            return ""
        return f"{value:.{decimals}f}"

    return formatter


def build_plot_filename(
    *,
    filename_namespace: Optional[str],
    plot_kind: str,
    model_name: Optional[str] = None,
    include_distance_models: bool = False,
    filter_val: Optional[bool] = None,
    legacy_prefix: str = "",
    legacy_suffix: str = "",
) -> str:
    if filename_namespace:
        model_aliases = {
            "mul-two-tower": "mul",
            "add-two-tower": "add",
        }
        kind_aliases = {
            "propensity": "prop",
            "methods": "methods",
        }

        parts = [slugify_plot_part(filename_namespace), kind_aliases.get(plot_kind, slugify_plot_part(plot_kind))]
        if model_name:
            parts.append(slugify_plot_part(model_aliases.get(model_name, model_name)))
        if include_distance_models:
            parts.append("all")
        if filter_val:
            parts.append("filtered")
        return f"{'-'.join(parts)}.pdf"

    if plot_kind == "propensity":
        return f"{legacy_prefix}Propensity_estimator_comparison_{model_name}{legacy_suffix}.pdf"
    if plot_kind == "methods":
        return f"{legacy_prefix}Method_Comparison_filter_{filter_val}{legacy_suffix}.pdf"
    raise ValueError(f"Unsupported plot_kind: {plot_kind}")


DATASET_TITLE_MAP = {
    "istella": "Istella-S",
    "mslr30k": "MSLR-WEB30K",
    "yahoo": "Yahoo!",
}

PROPENSITY_DISPLAY_ALIASES = {
    "MLPregression": "MLP Propensity",
    "frequency-based": "Frequency-based Propensity",
    "true_propensity": "Oracle Propensity",
    "max-score": "Label-Trained (Skyline)",
    "logging-policy": "Logging Policy (Baseline)",
    "cosine": "Cosine",
    "knn": "KNN",
    "kmeans": "K-Means",
}

PROPENSITY_DISPLAY_COLORS = {
    "MLPregression": "tab:blue",
    "frequency-based": "tab:orange",
    "true_propensity": "tab:green",
}


def propensity_display_color(model: str, index: int, colors) -> object:
    return PROPENSITY_DISPLAY_COLORS.get(model, colors[(index + 2) % len(colors)])


IPS_MODEL_DISPLAY_ALIASES = {
    "ips": "IPS",
    "dm": "DM",
    "dr": "DR",
}

DATASET_Y_DECIMALS = {
    "istella": 1,
    "mslr30k": 2,
    "yahoo": 2,
}


def plot_grid(
    agg: pd.DataFrame,
    baselines: pd.DataFrame,
    *,
    metrics_to_plot=("RCTR", "NDCG"),
    figsize_per_cell=(6.5, 4.8),
    float_tol=1e-12,
    include_distance_models=False,
    filename_suffix="",
    include_std_bars=True,
    dataset_col: Optional[str] = None,
    dataset_order=None,
    filename_prefix="",
    filename_namespace: Optional[str] = None,
    x_col: str = "n_sessions",
    x_label: str = "Number of Sessions",
    x_scale: str = "log",
    facet_by_temperature: bool = True,
    x_ticks=None,
    x_ticklabels=None,
    x_limits=None,
    sharex: bool = True,
    interval_mode: str = "std",
):
    baseline_models = {"max-score", "logging-policy"}
    ips_models = [m for m in sorted(agg["ips_model"].unique()) if m not in baseline_models]

    temps = sorted(agg["tmp"].unique())
    strengths = sorted(agg["policy_strength"].unique())

    if dataset_col is not None and dataset_col not in agg.columns:
        raise ValueError(f"dataset_col '{dataset_col}' not found in aggregated dataframe.")
    if x_col not in agg.columns:
        raise ValueError(f"x_col '{x_col}' not found in aggregated dataframe.")

    if dataset_col is None:
        datasets = [None]
    else:
        available = set(agg[dataset_col].dropna().unique())
        if dataset_order is None:
            datasets = sorted(available)
        else:
            datasets = list(dataset_order)
            if not datasets:
                datasets = sorted(available)

    temp_facets = len(temps) if facet_by_temperature else 1
    if dataset_col is None:
        n_rows = len(strengths)
        n_cols = temp_facets * len(metrics_to_plot)
    else:
        # Combined-dataset view: keep strengths as rows and place datasets side-by-side in columns.
        n_rows = len(strengths)
        n_cols = len(datasets) * temp_facets * len(metrics_to_plot)

    colors = plt.cm.tab10.colors

    style_map = {
        "max-score": dict(color="black", linestyle="--", linewidth=2),
        "logging-policy": dict(color="black", linestyle=":", linewidth=2),
    }
    band_alpha = {"max-score": 0.25, "logging-policy": 0.15}

    def plot_axis(ax, df_metric: pd.DataFrame, metric: str, prop_models, dataset_value=None):
        # curves (only those present in this figure)
        for i, pm in enumerate(prop_models):
            s = df_metric[df_metric["propensity_model"] == pm].sort_values(x_col)
            if s.empty:
                continue
            x = s[x_col].to_numpy()
            y = s["mean_value"].to_numpy()
            low, high = get_interval_bounds(s, interval_mode=interval_mode)
            c = propensity_display_color(pm, i, colors)
            linestyle = "-." if pm == "true_propensity" else "-"
            ax.plot(x, y, marker="o", color=c, linestyle=linestyle, zorder=2)
            if include_std_bars:
                ax.fill_between(x, low, high, color=c, alpha=0.2, zorder=1)

        # baseline horizontal bands (metric-only)
        if baselines is not None and not baselines.empty:
            b = baselines[baselines["metric"] == metric]
            if dataset_col is not None and dataset_col in baselines.columns and dataset_value is not None:
                b = b[b[dataset_col] == dataset_value]
            for _, row in b.iterrows():
                name = row["baseline_name"]
                mu = float(row["mean_value"])
                low, high = get_baseline_interval(row, interval_mode=interval_mode)
                ax.axhline(mu, zorder=4, **style_map.get(name, {}))
                ax.axhspan(low, high, color="grey", alpha=band_alpha.get(name, 0.2), zorder=3)

        if x_scale:
            ax.set_xscale(x_scale)
        if x_ticks is not None:
            ax.set_xticks(x_ticks)
        if x_ticklabels is not None:
            ax.set_xticklabels(x_ticklabels)
        if x_limits is not None:
            ax.set_xlim(*x_limits)
        ax.grid(True, which="both", linestyle="--", alpha=0.5)

    for ips_model in ips_models:
        # dm/dr use per-filter plots; ips/naive/naive-ho are merged; mul-two-tower/add-two-tower also use per-filter plots
        allowed_filters = [True, False] if ips_model in ("dm", "dr", "mul-two-tower", "add-two-tower") else ["merged"]

        for filt in allowed_filters:
            sub = agg[(agg["ips_model"] == ips_model) & (agg["plot_filter"] == filt)]
            if sub.empty:
                continue

            # ONLY propensity models actually present in this figure
            prop_models = sorted(sub["propensity_model"].unique())
            if not include_distance_models:
                prop_models = [pm for pm in prop_models if pm not in {"cosine", "knn", "kmeans"}]

            if not prop_models:
                continue

            print(
                f"IPS model: {ips_model} | filter_single_display_pairs={filt} | "
                f"Mean with {interval_mode} band over random_state"
            )
            
            fig, axes = plt.subplots(
                n_rows, n_cols,
                figsize=(figsize_per_cell[0] * n_cols, figsize_per_cell[1] * n_rows),
                sharex=sharex,
                sharey=False,
            )
            axes = ensure_2d_axes(axes, n_rows, n_cols)

            if dataset_col is None:
                for r, strength in enumerate(strengths):
                    s_df = sub[np.isclose(sub["policy_strength"], strength, atol=float_tol)]
                    if facet_by_temperature:
                        for t_idx, temp in enumerate(temps):
                            t_df = s_df[np.isclose(s_df["tmp"], temp, atol=float_tol)]
                            for m_idx, metric in enumerate(metrics_to_plot):
                                c = t_idx * len(metrics_to_plot) + m_idx
                                ax = axes[r, c]
                                m_df = t_df[t_df["metric"] == metric]
                                plot_axis(ax, m_df, metric, prop_models)

                                ax.set_xlabel(x_label)
                                ax.set_ylabel(metric)
                                print(f"Subplot: {metric} | T={temp} | strength={strength}")
                    else:
                        for m_idx, metric in enumerate(metrics_to_plot):
                            ax = axes[r, m_idx]
                            m_df = s_df[s_df["metric"] == metric]
                            plot_axis(ax, m_df, metric, prop_models)

                            ax.set_xlabel(x_label)
                            ax.set_ylabel(metric)
                            print(f"Subplot: {metric} | strength={strength}")
            else:
                cols_per_dataset = temp_facets * len(metrics_to_plot)
                for r, strength in enumerate(strengths):
                    for d_idx, dataset_value in enumerate(datasets):
                        ds_df = sub[sub[dataset_col] == dataset_value]
                        s_df = ds_df[np.isclose(ds_df["policy_strength"], strength, atol=float_tol)] if not ds_df.empty else ds_df
                        if facet_by_temperature:
                            for t_idx, temp in enumerate(temps):
                                t_df = s_df[np.isclose(s_df["tmp"], temp, atol=float_tol)] if not s_df.empty else s_df
                                for m_idx, metric in enumerate(metrics_to_plot):
                                    c = d_idx * cols_per_dataset + t_idx * len(metrics_to_plot) + m_idx
                                    ax = axes[r, c]
                                    if r == 0:
                                        dataset_title = DATASET_TITLE_MAP.get(str(dataset_value), str(dataset_value))
                                        ax.set_title(dataset_title)
                                    if t_df.empty:
                                        ax.text(
                                            0.5, 0.5, "No data",
                                            transform=ax.transAxes, ha="center", va="center", color="gray"
                                        )
                                        ax.set_xlabel(x_label)
                                        ax.set_ylabel(metric)
                                        continue
                                    m_df = t_df[t_df["metric"] == metric]
                                    plot_axis(ax, m_df, metric, prop_models, dataset_value=dataset_value)

                                    ax.set_xlabel(x_label)
                                    ax.set_ylabel(metric)
                                    print(
                                        f"Subplot: {metric} | T={temp} | strength={strength} | dataset={dataset_value}"
                                    )
                        else:
                            for m_idx, metric in enumerate(metrics_to_plot):
                                c = d_idx * cols_per_dataset + m_idx
                                ax = axes[r, c]
                                if r == 0:
                                    dataset_title = DATASET_TITLE_MAP.get(str(dataset_value), str(dataset_value))
                                    ax.set_title(dataset_title)
                                if s_df.empty:
                                    ax.text(
                                        0.5, 0.5, "No data",
                                        transform=ax.transAxes, ha="center", va="center", color="gray"
                                    )
                                    ax.set_xlabel(x_label)
                                    ax.set_ylabel(metric)
                                    continue
                                m_df = s_df[s_df["metric"] == metric]
                                plot_axis(ax, m_df, metric, prop_models, dataset_value=dataset_value)

                                ax.set_xlabel(x_label)
                                ax.set_ylabel(metric)
                                print(
                                    f"Subplot: {metric} | strength={strength} | dataset={dataset_value}"
                                )

            # build legend dynamically: only what is plotted for this figure
            handles = []
            for i, label in enumerate(prop_models):
                display_label = PROPENSITY_DISPLAY_ALIASES.get(label, label)
                handles.append(
                    Line2D(
                        [0], [0],
                        color=propensity_display_color(label, i, colors),
                        marker="o",
                        linestyle="-." if label == "true_propensity" else "-",
                        label=display_label,
                    )
                )

            baseline_models_present = (
                set(baselines["baseline_name"].unique())
                if baselines is not None and not baselines.empty
                else set()
            )
            if "max-score" in baseline_models_present:
                handles.append(
                    Line2D([0], [0], color="black", linestyle="--", linewidth=2, label="Label-Trained (Skyline)")
                )
            if "logging-policy" in baseline_models_present:
                handles.append(
                    Line2D([0], [0], color="black", linestyle=":", linewidth=2, label="Logging Policy (Baseline)")
                )

            fig.legend(
                handles=handles,
                labels=[h.get_label() for h in handles],
                loc="lower center",
                ncol=min(len(handles), 15),
                bbox_to_anchor=(0.5, -0.07),
                frameon=False,
            )

            plt.tight_layout()
            output_name = build_plot_filename(
                filename_namespace=filename_namespace,
                plot_kind="propensity",
                model_name=ips_model,
                include_distance_models=include_distance_models,
                legacy_prefix=filename_prefix,
                legacy_suffix=filename_suffix,
            )
            plt.savefig(
                os.path.join(PLOTS_DIR, output_name),
                bbox_inches='tight'
            )
            plt.close(fig)


def plot_dm_dr_ips_naiveho_frequency_based(
    agg: pd.DataFrame,
    baselines: Optional[pd.DataFrame] = None,
    *,
    filter_val: bool = False,
    metrics_to_plot=("RCTR", "NDCG"),
    figsize_per_cell=(6.5, 4.8),
    float_tol=1e-12,
    filename_suffix="",
    dataset_col: Optional[str] = None,
    dataset_order=None,
    filename_prefix="",
    filename_namespace: Optional[str] = None,
    x_col: str = "n_sessions",
    x_label: str = "Number of Sessions",
    x_scale: str = "log",
    facet_by_temperature: bool = True,
    x_ticks=None,
    x_ticklabels=None,
    x_limits=None,
    sharex: bool = True,
    interval_mode: str = "std",
):
    # --- keep only wanted methods ---
    wanted_ips_models = {"dm", "dr", "ips", "naive", "naive-ho", "mul-two-tower", "add-two-tower"}
    sub = agg[agg["ips_model"].isin(wanted_ips_models)].copy()
    if sub.empty:
        raise ValueError("No rows found for dm/dr/ips/naive/naive-ho/mul-two-tower/add-two-tower in agg.")

    # --- keep only frequency-based propensity runs for methods that have propensity ---
    # dm/dr/ips/mul-two-tower/add-two-tower have propensity_model entries; naive/naive-ho does not (and should be kept).
    sub = sub[
        (sub["ips_model"].isin({"dm", "dr", "ips", "mul-two-tower", "add-two-tower"}) & (sub["propensity_model"] == "frequency-based"))
        | (sub["ips_model"].isin({"naive", "naive-ho"}))
    ].copy()

    # --- dm/dr/mul-two-tower/add-two-tower: keep both filter settings (based on filter_val) ---
    # ips & naive & naive-ho: merged
    sub = sub[
        (sub["ips_model"].isin({"dm", "dr", "mul-two-tower", "add-two-tower"}) & sub["plot_filter"].isin([filter_val]))
        | (sub["ips_model"].isin({"ips", "naive", "naive-ho"}) & (sub["plot_filter"] == "merged"))
    ].copy()

    # --- build a "method label" for plotting lines ---
    pretty_name = {
        "dm": "Direct Method",
        "dr": "Doubly Robust",
        "ips": "Inverse Propensity Scoring",
        "naive": "Naive",
        "naive-ho": "Naive Hold-Out",
        "mul-two-tower": "Multiplicative Two-Tower",
        "add-two-tower": "Additive Two-Tower",
    }

    sub["method"] = sub.apply(
        lambda r: f"{pretty_name[r['ips_model']]}"
        if r["ips_model"] in {"dm", "dr", "mul-two-tower", "add-two-tower"}
        else pretty_name[r["ips_model"]],
        axis=1,
    )

    if dataset_col is not None and dataset_col not in sub.columns:
        raise ValueError(f"dataset_col '{dataset_col}' not found in aggregated dataframe.")
    if x_col not in sub.columns:
        raise ValueError(f"x_col '{x_col}' not found in aggregated dataframe.")

    if dataset_col is None:
        datasets = [None]
    else:
        available = set(sub[dataset_col].dropna().unique())
        if dataset_order is None:
            datasets = sorted(available)
        else:
            datasets = list(dataset_order)
            if not datasets:
                datasets = sorted(available)

    temps = sorted(sub["tmp"].unique())
    strengths = sorted(sub["policy_strength"].unique())

    temp_facets = len(temps) if facet_by_temperature else 1
    if dataset_col is None:
        n_rows = len(strengths)
        n_cols = temp_facets * len(metrics_to_plot)
    else:
        # Combined-dataset view: keep strengths as rows and place datasets side-by-side in columns.
        n_rows = len(strengths)
        n_cols = len(datasets) * temp_facets * len(metrics_to_plot)

    colors = plt.cm.tab10.colors

    # ---- baseline styles ----
    style_map = {
        "max-score": dict(color="black", linestyle="--", linewidth=2),
        "logging-policy": dict(color="black", linestyle=":", linewidth=2),
    }
    band_alpha = {"max-score": 0.25, "logging-policy": 0.15}

    def plot_axis(ax, df_metric: pd.DataFrame, metric: str, dataset_value=None):
        # Enforce method order: dm, dr, ips, naive, naive-ho, mul-two-tower, add-two-tower
        method_order = [
            "Direct Method", "Doubly Robust", "Inverse Propensity Scoring",
            "Naive", "Naive Hold-Out", "Multiplicative Two-Tower", "Additive Two-Tower"
        ]
        present_methods = [m for m in method_order if m in df_metric["method"].values]
        
        # Add any other methods not in order
        remaining = [m for m in sorted(df_metric["method"].unique()) if m not in present_methods]
        methods = present_methods + remaining

        # curves
        for i, method in enumerate(methods):
            s = df_metric[df_metric["method"] == method].sort_values(x_col)
            if s.empty:
                continue
            x = s[x_col].to_numpy()
            y = s["mean_value"].to_numpy()
            low, high = get_interval_bounds(s, interval_mode=interval_mode)
            c = colors[i % len(colors)]
            ax.plot(x, y, marker="o", color=c, zorder=2, label=method)
            ax.fill_between(x, low, high, color=c, alpha=0.2, zorder=1)

        # baseline horizontal bands (metric-only)
        if baselines is not None and not baselines.empty:
            b = baselines[baselines["metric"] == metric]
            if dataset_col is not None and dataset_col in baselines.columns and dataset_value is not None:
                b = b[b[dataset_col] == dataset_value]
            for _, row in b.iterrows():
                name = row["baseline_name"]
                mu = float(row["mean_value"])
                low, high = get_baseline_interval(row, interval_mode=interval_mode)
                ax.axhline(mu, zorder=4, **style_map.get(name, {}))
                ax.axhspan(low, high, color="grey", alpha=band_alpha.get(name, 0.2), zorder=3)

        if x_scale:
            ax.set_xscale(x_scale)
        if x_ticks is not None:
            ax.set_xticks(x_ticks)
        if x_ticklabels is not None:
            ax.set_xticklabels(x_ticklabels)
        if x_limits is not None:
            ax.set_xlim(*x_limits)
        ax.grid(True, which="both", linestyle="--", alpha=0.5)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(figsize_per_cell[0] * n_cols, figsize_per_cell[1] * n_rows),
        sharex=sharex,
        sharey=False,
    )
    axes = ensure_2d_axes(axes, n_rows, n_cols)

    if dataset_col is None:
        for r, strength in enumerate(strengths):
            s_df = sub[np.isclose(sub["policy_strength"], strength, atol=float_tol)]
            if facet_by_temperature:
                for t_idx, temp in enumerate(temps):
                    t_df = s_df[np.isclose(s_df["tmp"], temp, atol=float_tol)]
                    for m_idx, metric in enumerate(metrics_to_plot):
                        c = t_idx * len(metrics_to_plot) + m_idx
                        ax = axes[r, c]
                        m_df = t_df[t_df["metric"] == metric]
                        plot_axis(ax, m_df, metric)

                        ax.set_xlabel(x_label)
                        ax.set_ylabel(metric)
            else:
                for m_idx, metric in enumerate(metrics_to_plot):
                    ax = axes[r, m_idx]
                    m_df = s_df[s_df["metric"] == metric]
                    plot_axis(ax, m_df, metric)

                    ax.set_xlabel(x_label)
                    ax.set_ylabel(metric)
    else:
        cols_per_dataset = temp_facets * len(metrics_to_plot)
        for r, strength in enumerate(strengths):
            for d_idx, dataset_value in enumerate(datasets):
                ds_df = sub[sub[dataset_col] == dataset_value]
                s_df = ds_df[np.isclose(ds_df["policy_strength"], strength, atol=float_tol)] if not ds_df.empty else ds_df
                if facet_by_temperature:
                    for t_idx, temp in enumerate(temps):
                        t_df = s_df[np.isclose(s_df["tmp"], temp, atol=float_tol)] if not s_df.empty else s_df
                        for m_idx, metric in enumerate(metrics_to_plot):
                            c = d_idx * cols_per_dataset + t_idx * len(metrics_to_plot) + m_idx
                            ax = axes[r, c]
                            if r == 0:
                                dataset_title = {
                                    "istella": "Istella-S",
                                    "mslr30k": "MSLR-WEB30K",
                                    "yahoo": "Yahoo!",
                                }.get(str(dataset_value), str(dataset_value))
                                ax.set_title(dataset_title)
                            if t_df.empty:
                                ax.text(
                                    0.5, 0.5, "No data",
                                    transform=ax.transAxes, ha="center", va="center", color="gray"
                                )
                                ax.set_xlabel(x_label)
                                ax.set_ylabel(metric)
                                continue
                            m_df = t_df[t_df["metric"] == metric]
                            plot_axis(ax, m_df, metric, dataset_value=dataset_value)

                            ax.set_xlabel(x_label)
                            ax.set_ylabel(metric)
                else:
                    for m_idx, metric in enumerate(metrics_to_plot):
                        c = d_idx * cols_per_dataset + m_idx
                        ax = axes[r, c]
                        if r == 0:
                            dataset_title = {
                                "istella": "Istella-S",
                                "mslr30k": "MSLR-WEB30K",
                                "yahoo": "Yahoo!",
                            }.get(str(dataset_value), str(dataset_value))
                            ax.set_title(dataset_title)
                        if s_df.empty:
                            ax.text(
                                0.5, 0.5, "No data",
                                transform=ax.transAxes, ha="center", va="center", color="gray"
                            )
                            ax.set_xlabel(x_label)
                            ax.set_ylabel(metric)
                            continue
                        m_df = s_df[s_df["metric"] == metric]
                        plot_axis(ax, m_df, metric, dataset_value=dataset_value)

                        ax.set_xlabel(x_label)
                        ax.set_ylabel(metric)

    # One global legend (unique methods + baselines)
    handles, labels = [], []
    for ax in fig.axes:
        h, l = ax.get_legend_handles_labels()
        for hh, ll in zip(h, l):
            if ll not in labels:
                handles.append(hh)
                labels.append(ll)

    # add baseline legend entries if present
    baseline_models_present = (
        set(baselines["baseline_name"].unique())
        if baselines is not None and not baselines.empty
        else set()
    )
    if "max-score" in baseline_models_present and "max-score" not in labels:
        handles.append(Line2D([0], [0], color="black", linestyle="--", linewidth=2, label="Label-Trained (Skyline)"))
        labels.append("Label-Trained (Skyline)")
    if "logging-policy" in baseline_models_present and "logging-policy" not in labels:
        handles.append(
            Line2D([0], [0], color="black", linestyle=":", linewidth=2, label="Logging Policy (Baseline)")
        )
        labels.append("Logging Policy (Baseline)")

    fig.legend(
        handles=handles,
        labels=labels,
        loc="lower center",
        ncol=min(len(labels), 6),
        bbox_to_anchor=(0.5, -0.07),
        frameon=False,
    )
    plt.tight_layout()
    output_name = build_plot_filename(
        filename_namespace=filename_namespace,
        plot_kind="methods",
        filter_val=filter_val,
        legacy_prefix=filename_prefix,
        legacy_suffix=filename_suffix,
    )
    plt.savefig(
        os.path.join(PLOTS_DIR, output_name),
        bbox_inches='tight'
    )
    plt.close(fig)


def plot_ips_dm_dr_stacked_propensity_grid(
    agg: pd.DataFrame,
    baselines: pd.DataFrame,
    *,
    metric: str = "NDCG",
    figsize_per_cell=(5.8, 3.1),
    float_tol=1e-12,
    include_distance_models=False,
    filename_namespace: Optional[str] = None,
    output_name: Optional[str] = None,
    dataset_col: Optional[str] = None,
    dataset_order=None,
    x_col: str = "n_sessions",
    x_label: str = "Number of Sessions",
    x_scale: str = "log",
    x_ticks=None,
    x_ticklabels=None,
    x_limits=None,
    sharex: bool = True,
    interval_mode: str = "std",
    filter_val: bool = False,
    wspace: float = 0.12,
):
    """Plot IPS, DM, and DR as vertically stacked rows with shared x-axis."""
    if dataset_col is not None and dataset_col not in agg.columns:
        raise ValueError(f"dataset_col '{dataset_col}' not found in aggregated dataframe.")
    if x_col not in agg.columns:
        raise ValueError(f"x_col '{x_col}' not found in aggregated dataframe.")

    row_models = ("ips", "dm", "dr")
    sub = agg[agg["ips_model"].isin(row_models) & (agg["metric"] == metric)].copy()
    sub = sub[
        ((sub["ips_model"] == "ips") & (sub["plot_filter"] == "merged"))
        | (sub["ips_model"].isin({"dm", "dr"}) & (sub["plot_filter"] == filter_val))
    ].copy()

    if sub.empty:
        raise ValueError("No rows found for stacked IPS/DM/DR propensity plot.")

    if dataset_col is None:
        datasets = [None]
    else:
        available = set(sub[dataset_col].dropna().unique())
        datasets = sorted(available) if dataset_order is None else list(dataset_order)
        if not datasets:
            datasets = sorted(available)

    strengths = sorted(sub["policy_strength"].unique())
    if len(strengths) != 1:
        raise ValueError(
            "Stacked IPS/DM/DR plot expects exactly one policy_strength; "
            f"found {strengths}."
        )
    strength = strengths[0]
    sub = sub[np.isclose(sub["policy_strength"], strength, atol=float_tol)]

    n_rows = len(row_models)
    n_cols = len(datasets)
    colors = plt.cm.tab10.colors

    prop_models = sorted(sub["propensity_model"].dropna().unique())
    if not include_distance_models:
        prop_models = [pm for pm in prop_models if pm not in {"cosine", "knn", "kmeans"}]
    if not prop_models:
        raise ValueError("No propensity models available for stacked IPS/DM/DR plot.")

    style_map = {
        "max-score": dict(color="black", linestyle="--", linewidth=2),
        "logging-policy": dict(color="black", linestyle=":", linewidth=2),
    }
    band_alpha = {"max-score": 0.25, "logging-policy": 0.15}

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(figsize_per_cell[0] * n_cols, figsize_per_cell[1] * n_rows),
        sharex=sharex,
        sharey=False,
    )
    axes = ensure_2d_axes(axes, n_rows, n_cols)

    for r, ips_model in enumerate(row_models):
        for c, dataset_value in enumerate(datasets):
            ax = axes[r, c]
            df = sub[sub["ips_model"] == ips_model]
            if dataset_col is not None and dataset_value is not None:
                df = df[df[dataset_col] == dataset_value]

            if r == 0 and dataset_col is not None:
                ax.set_title(latex_bold(DATASET_TITLE_MAP.get(str(dataset_value), str(dataset_value))))

            if c == 0:
                ax.text(
                    -0.22,
                    0.5,
                    latex_bold(IPS_MODEL_DISPLAY_ALIASES.get(ips_model, ips_model.upper())),
                    transform=ax.transAxes,
                    rotation=90,
                    ha="center",
                    va="center",
                    fontsize=plt.rcParams["axes.titlesize"],
                )
                ax.set_ylabel(metric)

            if df.empty:
                ax.text(
                    0.5,
                    0.5,
                    "No data",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    color="gray",
                )
            else:
                for i, pm in enumerate(prop_models):
                    s = df[df["propensity_model"] == pm].sort_values(x_col)
                    if s.empty:
                        continue
                    x = s[x_col].to_numpy()
                    y = s["mean_value"].to_numpy()
                    low, high = get_interval_bounds(s, interval_mode=interval_mode)
                    color = propensity_display_color(pm, i, colors)
                    linestyle = "-." if pm == "true_propensity" else "-"
                    ax.plot(x, y, marker="o", color=color, linestyle=linestyle, zorder=2)
                    ax.fill_between(x, low, high, color=color, alpha=0.2, zorder=1)

                if baselines is not None and not baselines.empty:
                    b = baselines[baselines["metric"] == metric]
                    if dataset_col is not None and dataset_col in baselines.columns and dataset_value is not None:
                        b = b[b[dataset_col] == dataset_value]
                    for _, row in b.iterrows():
                        name = row["baseline_name"]
                        mu = float(row["mean_value"])
                        low, high = get_baseline_interval(row, interval_mode=interval_mode)
                        ax.axhline(mu, zorder=4, **style_map.get(name, {}))
                        ax.axhspan(low, high, color="grey", alpha=band_alpha.get(name, 0.2), zorder=3)

            if x_scale:
                ax.set_xscale(x_scale)
            if x_ticks is not None:
                ax.set_xticks(x_ticks)
            if x_ticklabels is not None:
                ax.set_xticklabels(x_ticklabels)
            if x_limits is not None:
                ax.set_xlim(*x_limits)
            if r == n_rows - 1:
                ax.set_xlabel(x_label)
            else:
                ax.set_xlabel("")
            if x_ticklabels is None and x_scale != "log":
                ax.xaxis.set_major_formatter(FuncFormatter(format_max_two_decimals))
            y_decimals = DATASET_Y_DECIMALS.get(str(dataset_value), 2)
            ax.yaxis.set_major_formatter(FuncFormatter(format_fixed_decimals(y_decimals)))
            ax.grid(True, which="both", linestyle="--", alpha=0.5)

    handles = []
    for i, label in enumerate(prop_models):
        handles.append(
            Line2D(
                [0],
                [0],
                color=propensity_display_color(label, i, colors),
                marker="o",
                linestyle="-." if label == "true_propensity" else "-",
                label=PROPENSITY_DISPLAY_ALIASES.get(label, label),
            )
        )

    baseline_models_present = (
        set(baselines["baseline_name"].unique())
        if baselines is not None and not baselines.empty
        else set()
    )
    if "max-score" in baseline_models_present:
        handles.append(Line2D([0], [0], color="black", linestyle="--", linewidth=2, label="Label-Trained (Skyline)"))
    if "logging-policy" in baseline_models_present:
        handles.append(
            Line2D([0], [0], color="black", linestyle=":", linewidth=2, label="Logging Policy (Baseline)")
        )

    fig.legend(
        handles=handles,
        labels=[h.get_label() for h in handles],
        loc="lower center",
        ncol=min(len(handles), 5),
        bbox_to_anchor=(0.5, -0.04),
        frameon=False,
    )
    fig.align_ylabels()
    plt.tight_layout()
    fig.subplots_adjust(wspace=wspace)
    for upper_row in range(n_rows - 1):
        upper_axes = axes[upper_row, :]
        lower_axes = axes[upper_row + 1, :]
        first_pos = axes[upper_row, 0].get_position()
        x0 = max(0.0, first_pos.x0 - 0.30 * first_pos.width)
        x1 = max(ax.get_position().x1 for ax in upper_axes)
        y = (
            min(ax.get_position().y0 for ax in upper_axes)
            + max(ax.get_position().y1 for ax in lower_axes)
        ) / 2
        fig.add_artist(
            Line2D(
                [x0, x1],
                [y, y],
                transform=fig.transFigure,
                color="black",
                linewidth=0.8,
                alpha=0.75,
            )
        )

    if output_name is None:
        if not filename_namespace:
            raise ValueError("Either output_name or filename_namespace must be provided.")
        output_name = f"{slugify_plot_part(filename_namespace)}-prop-stacked.pdf"

    plt.savefig(os.path.join(PLOTS_DIR, output_name), bbox_inches="tight")
    plt.close(fig)

def plot_policy_models_train_histograms_ps1p0(
    results_dir: str = "results/real_targets",
    output_csv: str = "result_parsing/result_plots/training_histograms_real_targets_summary.csv",
):
    results_root = Path(REPO_ROOT) / results_dir
    pattern = re.compile(
        r"data=(?P<dataset>[^,]+).*?ips\.n_sessions=(?P<n_sessions>\d+).*?random_state=(?P<random_state>\d+)"
    )
    dataset_order = ["istella", "mslr30k", "yahoo"]
    dataset_labels = {
        "istella": "Istella-S",
        "mslr30k": "MSLR-WEB30K",
        "yahoo": "Yahoo!",
    }

    rows = []
    seen_keys = set()

    for histogram_path in sorted(results_root.glob("*/display_histograms/display_histogram_train.json")):
        match = pattern.search(str(histogram_path))
        if match is None:
            continue

        key = (
            match.group("dataset"),
            int(match.group("n_sessions")),
            int(match.group("random_state")),
        )
        if key in seen_keys:
            continue
        seen_keys.add(key)

        with histogram_path.open() as f:
            payload = json.load(f)

        histogram = {int(k): int(v) for k, v in payload["histogram"].items()}
        total_docs = int(payload["total_docs"])
        observed_docs = sum(count for obs, count in histogram.items() if obs >= 1)
        single_observation_docs = histogram.get(1, 0)
        weighted_observation_sum = sum(obs * count for obs, count in histogram.items() if obs >= 1)

        single_observation_proportion = (
            single_observation_docs / observed_docs if observed_docs else float("nan")
        )
        observed_at_least_once_proportion = (
            observed_docs / total_docs if total_docs else float("nan")
        )
        avg_observations_per_doc = (
            weighted_observation_sum / total_docs if total_docs else float("nan")
        )

        rows.append(
            {
                "dataset": key[0],
                "n_sessions": key[1],
                "random_state": key[2],
                "total_docs": total_docs,
                "observed_docs": observed_docs,
                "single_observation_docs": single_observation_docs,
                "weighted_observation_sum": weighted_observation_sum,
                "single_observation_proportion": single_observation_proportion,
                "observed_at_least_once_proportion": observed_at_least_once_proportion,
                "avg_observations_per_doc": avg_observations_per_doc,
                "source_path": os.path.relpath(histogram_path, REPO_ROOT),
            }
        )

    if not rows:
        raise RuntimeError(f"No training histogram files found under {results_root}.")

    df = pd.DataFrame(rows).sort_values(["dataset", "n_sessions", "random_state"]).reset_index(drop=True)
    output_csv_path = Path(REPO_ROOT) / output_csv
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv_path, index=False)

    summary = (
        df.groupby(["dataset", "n_sessions"], as_index=False)
        .agg(
            single_observation_proportion_mean=("single_observation_proportion", "mean"),
            single_observation_proportion_std=("single_observation_proportion", "std"),
            single_observation_proportion_p05=("single_observation_proportion", lambda values: values.quantile(0.05)),
            single_observation_proportion_p95=("single_observation_proportion", lambda values: values.quantile(0.95)),
            observed_at_least_once_proportion_mean=("observed_at_least_once_proportion", "mean"),
            observed_at_least_once_proportion_std=("observed_at_least_once_proportion", "std"),
            observed_at_least_once_proportion_p05=("observed_at_least_once_proportion", lambda values: values.quantile(0.05)),
            observed_at_least_once_proportion_p95=("observed_at_least_once_proportion", lambda values: values.quantile(0.95)),
            avg_observations_per_doc_mean=("avg_observations_per_doc", "mean"),
            avg_observations_per_doc_std=("avg_observations_per_doc", "std"),
            avg_observations_per_doc_p05=("avg_observations_per_doc", lambda values: values.quantile(0.05)),
            avg_observations_per_doc_p95=("avg_observations_per_doc", lambda values: values.quantile(0.95)),
            n_random_states=("random_state", "nunique"),
        )
        .sort_values(["dataset", "n_sessions"])
        .reset_index(drop=True)
    )

    metric_specs = [
        (
            "single_observation_proportion",
            "Single-observation proportion",
            "P(obs = 1 | obs >= 1)",
            "Training_Histograms_Single_Observation.pdf",
            "tab:blue",
        ),
        (
            "observed_at_least_once_proportion",
            "Observed-at-least-once proportion",
            "P(obs >= 1)",
            "Training_Histograms_Observed_At_Least_Once.pdf",
            "tab:orange",
        ),
        (
            "avg_observations_per_doc",
            "Average observations per doc",
            "Average observations per document",
            "Training_Histograms_Avg_Observations_Per_Doc.pdf",
            "tab:green",
        ),
    ]

    for metric_key, title, ylabel, filename, color in metric_specs:
        fig, axes = plt.subplots(1, len(dataset_order), figsize=(15, 4.3), sharex=True)

        for ax, dataset in zip(axes, dataset_order):
            dataset_df = summary[summary["dataset"] == dataset].sort_values("n_sessions")
            x = dataset_df["n_sessions"].to_numpy()
            y = dataset_df[f"{metric_key}_mean"].to_numpy()
            low = dataset_df[f"{metric_key}_p05"].fillna(dataset_df[f"{metric_key}_mean"]).to_numpy()
            high = dataset_df[f"{metric_key}_p95"].fillna(dataset_df[f"{metric_key}_mean"]).to_numpy()

            ax.plot(x, y, marker="o", color=color, linewidth=2)
            ax.fill_between(x, low, high, color=color, alpha=0.2)
            ax.set_xscale("log")
            ax.set_title(dataset_labels[dataset])
            ax.set_xlabel("Number of sessions")
            ax.grid(True, which="both", linestyle="--", alpha=0.4)

        axes[0].set_ylabel(ylabel)
        fig.suptitle(title, y=1.02)
        fig.tight_layout()
        fig.savefig(os.path.join(PLOTS_DIR, filename), bbox_inches="tight")
        plt.close(fig)

    combined_fig, combined_axes = plt.subplots(len(metric_specs), len(dataset_order), figsize=(15, 11), sharex=True)
    combined_axes = np.atleast_2d(combined_axes)

    for row_idx, (metric_key, title, ylabel, _, color) in enumerate(metric_specs):
        for col_idx, dataset in enumerate(dataset_order):
            ax = combined_axes[row_idx, col_idx]
            dataset_df = summary[summary["dataset"] == dataset].sort_values("n_sessions")
            x = dataset_df["n_sessions"].to_numpy()
            y = dataset_df[f"{metric_key}_mean"].to_numpy()
            low = dataset_df[f"{metric_key}_p05"].fillna(dataset_df[f"{metric_key}_mean"]).to_numpy()
            high = dataset_df[f"{metric_key}_p95"].fillna(dataset_df[f"{metric_key}_mean"]).to_numpy()

            ax.plot(x, y, marker="o", color=color, linewidth=2)
            ax.fill_between(x, low, high, color=color, alpha=0.2)
            ax.set_xscale("log")
            ax.grid(True, which="both", linestyle="--", alpha=0.4)
            if row_idx == 0:
                ax.set_title(dataset_labels[dataset])
            if col_idx == 0:
                ax.set_ylabel(ylabel)
            if row_idx == len(metric_specs) - 1:
                ax.set_xlabel("Number of sessions")

    combined_fig.tight_layout()
    combined_fig.savefig(os.path.join(PLOTS_DIR, "Training_Histograms.pdf"), bbox_inches="tight")
    plt.close(combined_fig)


def plot_temperature_analysis(
    agg: pd.DataFrame,
    baselines: Optional[pd.DataFrame] = None,
    metrics_to_plot=("RCTR", "NDCG"),
    n_sessions_list=(1000, 5000, 10000),
    include_distance_models=False,
    filename_suffix="",
    interval_mode: str = "std",
):
    temps = sorted(agg["tmp"].unique())
    propensity_models = [m for m in sorted(agg["propensity_model"].unique()) if pd.notna(m)]
    if not include_distance_models:
        propensity_models = [pm for pm in propensity_models if pm not in {"cosine", "knn", "kmeans"}]
        
    filters = sorted(agg["plot_filter"].unique())

    display_aliases = {
        "MLPregression": "MLP Propensity",
        "frequency-based": "Frequency-based Propensity",
        "true_propensity": "Oracle Propensity",
        "max-score": "Label-Trained (Skyline)",
        "logging-policy": "Logging Policy (Baseline)",
        "cosine": "Cosine",
        "knn": "KNN",
        "kmeans": "K-Means",
    }

    colors = plt.cm.tab10.colors

    for filt in filters:
        agg_f = agg[agg["plot_filter"] == filt]
        if agg_f.empty:
            continue

        print(
            f"Temperature Analysis | filter_single_display_pairs={filt} | "
            "Mean with standard-deviation band over random_state"
        )
        
        n_cols = len(metrics_to_plot) * len(n_sessions_list)
        fig, axes = plt.subplots(1, n_cols, figsize=(6.8 * n_cols, 4.8), sharey=False)
        if hasattr(axes, 'flatten'):
            axes_flat = axes.flatten()
        else:
            axes_flat = [axes]
        
        ax_idx = 0
        for metric in metrics_to_plot:
            mdf = agg_f[agg_f["metric"] == metric]
            
            for n_sessions in n_sessions_list:
                ax = axes_flat[ax_idx]
                ax_idx += 1
                
                print(f"Subplot: n_sessions={n_sessions} | {metric}")
                cell = mdf[mdf["n_sessions"] == n_sessions]

                for i, pm in enumerate(propensity_models):
                    sub = cell[cell["propensity_model"] == pm].sort_values("tmp")
                    if sub.empty:
                        continue

                    x = sub["tmp"].to_numpy()
                    y = sub["mean_value"].to_numpy()
                    low, high = get_interval_bounds(sub, interval_mode=interval_mode)
                    color = propensity_display_color(pm, i, colors)

                    linestyle = "-." if pm == "true_propensity" else "-"
                    ax.plot(x, y, marker="o", color=color, linestyle=linestyle)
                    ax.fill_between(x, low, high, color=color, alpha=0.2)

                if baselines is not None and not baselines.empty:
                    b = baselines[baselines["metric"] == metric]
                    for _, row in b.iterrows():
                        name = row["baseline_name"]
                        mu = float(row["mean_value"])
                        low, high = get_baseline_interval(row, interval_mode=interval_mode)
                        style = dict(color="black", linestyle="--", linewidth=2) if name == "max-score" else dict(color="black", linestyle=":", linewidth=2)
                        alpha = 0.25 if name == "max-score" else 0.15
                        ax.axhline(mu, zorder=4, **style)
                        ax.axhspan(low, high, color="grey", alpha=alpha, zorder=3)

                ax.grid(True, which="both", linestyle="--", alpha=0.4)
                ax.set_xlabel("policy_temperature")
                ax.set_ylabel(metric)
                ax.set_xticks(temps)
                ax.set_xlim(min(temps), max(temps))

        # Build legend
        legend_handles = [
            Line2D(
                [0],
                [0],
                color=propensity_display_color(pm, i, colors),
                marker="o",
                linestyle="-." if pm == "true_propensity" else "-",
                label=display_aliases.get(pm, pm),
            )
            for i, pm in enumerate(propensity_models)
        ]
        
        if baselines is not None and not baselines.empty:
            baseline_models_present = set(baselines["baseline_name"].unique())
            if "max-score" in baseline_models_present:
                legend_handles.append(
                    Line2D([0], [0], color="black", linestyle="--", linewidth=2, label="Label-Trained (Skyline)")
                )
            if "logging-policy" in baseline_models_present:
                legend_handles.append(
                    Line2D([0], [0], color="black", linestyle=":", linewidth=2, label="Logging Policy (Baseline)")
                )

        fig.legend(handles=legend_handles, loc="lower center",
                   ncol=min(len(legend_handles), 6), bbox_to_anchor=(0.5, -0.05))

        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15)
        plt.savefig(os.path.join(PLOTS_DIR, f"Temperature_Analysis_{filt}{filename_suffix}.pdf"), bbox_inches='tight')
        plt.close(fig)
