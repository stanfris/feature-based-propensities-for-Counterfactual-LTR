import argparse
import os
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(__file__).resolve().parents[1] / "thesis_plots" / ".matplotlib"),
)

import matplotlib

matplotlib.use("Agg")
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, MaxNLocator
import numpy as np
import pandas as pd

from notebooks.analysis.plot_style import apply_thesis_plot_style

apply_thesis_plot_style(plt)


DEFAULT_RESULTS_DIR = REPO_ROOT / "results" / "compare_propensity_estimation"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "notebooks" / "thesis_plots"
DEFAULT_THESIS_IMAGES_DIR = REPO_ROOT.parent / "Thesis" / "images"
LOG_SCALE_EPS = 1e-6
PROPENSITY_YTICKS = [0.03, 0.05, 0.1, 0.3, 0.5]
THESIS_OUTPUT_ALIASES = {
    "propensity_estimation_Cosine_Sim.pdf": "propensity_estimation_Cosine_Sim.pdf",
    "propensity_estimation_Euclidean.pdf": "propensity_estimation_Euclidean_model_comp.pdf",
    "propensity_estimation_Euclidean_no_3p5_with_MLP.pdf": "propensity_estimation_Euclidean_no_3p5_with_MLP.pdf",
    "propensity_estimation_Euclidean_no_3p5_with_MLP_lines.pdf": "propensity_estimation_Euclidean_no_3p5_with_MLP_lines.pdf",
    "propensity_estimation_Euclidean_no_3p5_with_MLP_mean_lines.pdf": "propensity_estimation_Euclidean_no_3p5_with_MLP_mean_lines.pdf",
    "propensity_estimation_FINAL_COMPARISON.pdf": "propensity_estimation_FINAL_COMPARISON.pdf",
    "propensity_estimation_KMeans.pdf": "propensity_estimation_KMeans.pdf",
    "propensity_estimation_KNN.pdf": "propensity_estimation_KNN.pdf",
    "propensity_estimation_MLP_Classifier.pdf": "propensity_estimation_MLP_Classifier.pdf",
    "propensity_estimation_frequency_based_Euclidean_euclidean_t3p5_k50.pdf": "propensity_estimation_Euclidean_vs_OBS.pdf",
    "propensity_estimation_frequency_based_MLPClassifier.pdf": "propensity_estimation_frequency_based_MLPClassifier.pdf",
    "propensity_estimation_frequency_based_MLPRegressor_mlp_regression_l4_h128_d0p0.pdf": "propensity_estimation_frequency_based_MLPRegressor.pdf",
    "propensity_estimation_frequency_based_only.pdf": "propensity_estimation_frequency_based_none.pdf",
}


@dataclass(frozen=True)
class PropensityRun:
    csv_path: Path
    family: str
    method: str
    label: str
    detail_filename: str
    sort_key: tuple


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return slug or "plot"


def _abbr_to_float(value: str) -> float:
    return float(value.replace("p", "."))


def _format_float(value: float) -> str:
    return f"{value:g}"


def logits_to_propensity(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    propensities = 1.0 / (1.0 + np.exp(-values))
    return np.clip(propensities, LOG_SCALE_EPS, 1.0)


def logit_band_to_propensity(
    mean_values: np.ndarray,
    std_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    lower = logits_to_propensity(mean_values - std_values)
    upper = logits_to_propensity(mean_values + std_values)
    return lower, upper


def format_propensity_tick(value: float, _position: int) -> str:
    if np.isclose(value, 0.0):
        return "0"
    return f"{value:g}"


def configure_propensity_axis(ax) -> None:
    ax.set_yscale("log")
    ax.set_ylim(PROPENSITY_YTICKS[0], PROPENSITY_YTICKS[-1])
    ax.yaxis.set_major_locator(FixedLocator(PROPENSITY_YTICKS))
    ax.yaxis.set_major_formatter(FuncFormatter(format_propensity_tick))


def describe_csv(csv_path: Path) -> PropensityRun:
    name = csv_path.stem

    match = re.search(r"propensity_mlp_classifier_clf_l(\d+)_h(\d+)_d([0-9p]+)", name)
    if match:
        layers, hidden, dropout = match.groups()
        method = f"mlp_classifier_l{layers}_h{hidden}_d{dropout}"
        label = f"MLP Classifier l={layers}, h={hidden}, d={_format_float(_abbr_to_float(dropout))}"
        return PropensityRun(
            csv_path=csv_path,
            family="mlp_classifier",
            method=method,
            label=label,
            detail_filename=f"propensity_estimation_frequency_based_MLPClassifier_{method}.pdf",
            sort_key=(30, int(layers), int(hidden), _abbr_to_float(dropout)),
        )

    match = re.search(r"propensity_mlp_regression_reg_l(\d+)_h(\d+)_d([0-9p]+)", name)
    if match:
        layers, hidden, dropout = match.groups()
        method = f"mlp_regression_l{layers}_h{hidden}_d{dropout}"
        label = f"Propensity MLP l={layers}, h={hidden}, d={_format_float(_abbr_to_float(dropout))}"
        return PropensityRun(
            csv_path=csv_path,
            family="mlp_regression",
            method=method,
            label=label,
            detail_filename=f"propensity_estimation_frequency_based_MLPRegressor_{method}.pdf",
            sort_key=(40, int(layers), int(hidden), _abbr_to_float(dropout)),
        )

    match = re.search(r"unsupervised_cosine_cos_t([0-9p]+)_k(\d+)", name)
    if match:
        threshold, k = match.groups()
        threshold_value = _abbr_to_float(threshold)
        method = f"cosine_t{threshold}_k{k}"
        label = f"Cosine Similarity Grouping sim={_format_float(threshold_value)}"
        return PropensityRun(
            csv_path=csv_path,
            family="cosine",
            method=method,
            label=label,
            detail_filename=f"propensity_estimation_frequency_based_cosine_{method}.pdf",
            sort_key=(10, threshold_value, int(k)),
        )

    match = re.search(r"unsupervised_euclidean_euc_t([0-9p]+)_k(\d+)", name)
    if match:
        threshold, k = match.groups()
        threshold_value = _abbr_to_float(threshold)
        method = f"euclidean_t{threshold}_k{k}"
        label = f"Euclidean Grouping $\\epsilon={_format_float(threshold_value)}$"
        return PropensityRun(
            csv_path=csv_path,
            family="euclidean",
            method=method,
            label=label,
            detail_filename=f"propensity_estimation_frequency_based_Euclidean_{method}.pdf",
            sort_key=(15, threshold_value, int(k)),
        )

    match = re.search(r"unsupervised_knn_knn_k(\d+)", name)
    if match:
        k = int(match.group(1))
        method = f"knn_k{k}"
        label = f"KNN Grouping K={k}"
        return PropensityRun(
            csv_path=csv_path,
            family="knn",
            method=method,
            label=label,
            detail_filename=f"propensity_estimation_frequency_based_KNN_{method}.pdf",
            sort_key=(20, k),
        )

    match = re.search(r"unsupervised_kmeans_kmeans_g(\d+)_min(\d+)_it(\d+)", name)
    if match:
        groups, min_size, iters = (int(v) for v in match.groups())
        method = f"kmeans_g{groups}_min{min_size}_it{iters}"
        label = f"K-Means Grouping K={groups}"
        return PropensityRun(
            csv_path=csv_path,
            family="kmeans",
            method=method,
            label=label,
            detail_filename=f"propensity_estimation_frequency_based_KMeans_{method}.pdf",
            sort_key=(25, groups, min_size, iters),
        )

    method = slugify(name.removesuffix("_obs1_12"))
    return PropensityRun(
        csv_path=csv_path,
        family="other",
        method=method,
        label=method.replace("_", " "),
        detail_filename=f"propensity_estimation_frequency_based_{method}.pdf",
        sort_key=(99, method),
    )


def discover_runs(results_dir: Path) -> list[PropensityRun]:
    runs_by_method = {}
    for csv_path in sorted(results_dir.glob("**/compare_propensity_*_obs1-12.csv")):
        run = describe_csv(csv_path)
        if run.family in {"mlp_classifier", "mlp_regression"} and not run.method.endswith("_d0p0"):
            continue
        runs_by_method.setdefault(run.method, run)

    runs = sorted(runs_by_method.values(), key=lambda run: run.sort_key)
    if not runs:
        raise RuntimeError(f"No compare_propensity CSV files found under {results_dir}.")
    return runs


def read_obs_counts(value: str) -> list[int]:
    counts = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not counts:
        raise argparse.ArgumentTypeError("At least one observation count is required.")
    return counts


def panel_model_label(run: PropensityRun) -> str:
    return {
        "cosine": "Cosine Grouped Propensity",
        "euclidean": "Euclidean Grouped Propensity",
        "knn": "KNN Grouped Propensity",
        "kmeans": "K-Means Grouped Propensity",
        "mlp_classifier": "MLP Classifier",
        "mlp_regression": "Propensity MLP",
    }.get(run.family, run.label)


def plot_multi_panel(run: PropensityRun, obs_counts: list[int], output_dir: Path) -> Path:
    df = pd.read_csv(run.csv_path)
    positions = df["position"].to_numpy()
    effective_alpha = logits_to_propensity(df["effective_alpha"].to_numpy())
    model_label = panel_model_label(run)

    n_cols = 2
    n_rows = int(np.ceil(len(obs_counts) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows), sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()

    for idx, obs_count in enumerate(obs_counts):
        ax = axes[idx]
        mean_pred = df[f"pred_mean_obs{obs_count}"].to_numpy()
        std_pred = df[f"pred_std_obs{obs_count}"].to_numpy()
        mean_expected = df[f"exp_mean_obs{obs_count}"].to_numpy()
        std_expected = df[f"exp_std_obs{obs_count}"].to_numpy()
        valid = (
            np.isfinite(mean_pred)
            & np.isfinite(std_pred)
            & np.isfinite(mean_expected)
            & np.isfinite(std_expected)
        )
        pos_valid = positions[valid]
        mean_pred_propensity = logits_to_propensity(mean_pred[valid])
        pred_lower, pred_upper = logit_band_to_propensity(mean_pred[valid], std_pred[valid])
        mean_expected_propensity = logits_to_propensity(mean_expected[valid])
        expected_lower, expected_upper = logit_band_to_propensity(
            mean_expected[valid],
            std_expected[valid],
        )

        ax.plot(
            positions,
            effective_alpha,
            label="Oracle Propensity",
            color="black",
            linestyle="-.",
            linewidth=2.0,
            zorder=3,
        )
        ax.plot(
            pos_valid,
            mean_expected_propensity,
            label="Frequency-Based Propensity",
            marker="s",
            markersize=6,
            color="tab:blue",
            alpha=0.8,
            zorder=2,
        )
        ax.fill_between(
            pos_valid,
            expected_lower,
            expected_upper,
            alpha=0.2,
            color="tab:blue",
            label="_nolegend_",
        )
        ax.plot(
            pos_valid,
            mean_pred_propensity,
            label=model_label,
            marker="o",
            markersize=5,
            color="tab:orange",
            alpha=0.8,
            zorder=2,
        )
        ax.fill_between(
            pos_valid,
            pred_lower,
            pred_upper,
            alpha=0.2,
            color="tab:orange",
            label="_nolegend_",
        )
        ax.set_title(f"Observations = {obs_count}")
        configure_propensity_axis(ax)
        ax.grid(True, linestyle="--", alpha=0.35)

    for idx in range(len(obs_counts), len(axes)):
        axes[idx].set_visible(False)

    for ax in axes[: len(obs_counts)]:
        ax.set_xlabel("Position")
    for ax in axes[::n_cols]:
        ax.set_ylabel("Propensity")

    handles, labels = axes[0].get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    fig.legend(
        unique.values(),
        unique.keys(),
        loc="lower center",
        ncol=len(unique),
        frameon=False,
        bbox_to_anchor=(0.54, 0.04),
    )
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.16)

    output_path = output_dir / run.detail_filename
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    return output_path


def plot_frequency_only_panel(run: PropensityRun, obs_counts: list[int], output_dir: Path) -> Path:
    df = pd.read_csv(run.csv_path)
    positions = df["position"].to_numpy()
    effective_alpha = logits_to_propensity(df["effective_alpha"].to_numpy())

    n_cols = 2
    n_rows = int(np.ceil(len(obs_counts) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows), sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()

    for idx, obs_count in enumerate(obs_counts):
        ax = axes[idx]
        mean_expected = df[f"exp_mean_obs{obs_count}"].to_numpy()
        std_expected = df[f"exp_std_obs{obs_count}"].to_numpy()
        valid = np.isfinite(mean_expected) & np.isfinite(std_expected)
        pos_valid = positions[valid]
        mean_expected_propensity = logits_to_propensity(mean_expected[valid])
        expected_lower, expected_upper = logit_band_to_propensity(
            mean_expected[valid],
            std_expected[valid],
        )

        ax.plot(
            positions,
            effective_alpha,
            label="Oracle Propensity",
            color="black",
            linestyle="-.",
            linewidth=2.0,
            zorder=3,
        )
        ax.plot(
            pos_valid,
            mean_expected_propensity,
            label="Frequency-Based Propensity",
            marker="s",
            markersize=6,
            color="tab:blue",
            alpha=0.8,
            zorder=2,
        )
        ax.fill_between(
            pos_valid,
            expected_lower,
            expected_upper,
            alpha=0.2,
            color="tab:blue",
            label="_nolegend_",
        )
        ax.set_title(f"Observations = {obs_count}")
        configure_propensity_axis(ax)
        ax.grid(True, linestyle="--", alpha=0.35)

    for idx in range(len(obs_counts), len(axes)):
        axes[idx].set_visible(False)

    for ax in axes[: len(obs_counts)]:
        ax.set_xlabel("Position")
    for ax in axes[::n_cols]:
        ax.set_ylabel("Propensity")

    handles, labels = axes[0].get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    fig.legend(
        unique.values(),
        unique.keys(),
        loc="lower center",
        ncol=len(unique),
        frameon=False,
        bbox_to_anchor=(0.54, 0.04),
    )
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.16)

    output_path = output_dir / "propensity_estimation_frequency_based_only.pdf"
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    return output_path


def create_summary_table(runs: list[PropensityRun], obs_counts: list[int]) -> pd.DataFrame:
    rows = []
    expected_added = set()

    for run in runs:
        df = pd.read_csv(run.csv_path)
        effective = logits_to_propensity(df["effective_alpha"].to_numpy())

        for obs_count in obs_counts:
            for prefix, method, label, family, sort_key in (
                ("pred", run.method, run.label, run.family, run.sort_key),
                ("exp", "frequency_based", "Frequency-Based", "frequency_based", (0,)),
            ):
                if method == "frequency_based":
                    key = (method, obs_count)
                    if key in expected_added:
                        continue
                    expected_added.add(key)

                mean_col = f"{prefix}_mean_obs{obs_count}"
                std_col = f"{prefix}_std_obs{obs_count}"
                if mean_col not in df.columns or std_col not in df.columns:
                    continue

                mean_vals = df[mean_col].to_numpy()
                std_vals = df[std_col].to_numpy()
                valid = np.isfinite(mean_vals) & np.isfinite(std_vals) & np.isfinite(effective)
                if not np.any(valid):
                    continue
                mean_propensity = logits_to_propensity(mean_vals[valid])
                lower_propensity, upper_propensity = logit_band_to_propensity(
                    mean_vals[valid],
                    std_vals[valid],
                )
                propensity_std = (upper_propensity - lower_propensity) / 2.0

                rows.append(
                    {
                        "obs_count": obs_count,
                        "method": method,
                        "label": label,
                        "family": family,
                        "sort_key": sort_key,
                        "avg_std": float(np.mean(propensity_std)),
                        "mean_abs_deviation_from_true": float(
                            np.mean(np.abs(mean_propensity - effective[valid]))
                        ),
                        "abs_mean_deviation_from_true": float(
                            np.abs(np.mean(mean_propensity) - np.mean(effective[valid]))
                        ),
                    }
                )

    if not rows:
        raise RuntimeError("No summary rows could be computed from discovered CSV files.")
    return pd.DataFrame(rows)


def plot_summary(
    summary_df: pd.DataFrame,
    output_path: Path,
    title: str | None = None,
    sort_by_mean_deviation: bool = False,
    legend_ncol: int = 4,
) -> Path:
    summary_df = summary_df.drop_duplicates(subset=["obs_count", "method"])
    labels = summary_df.drop_duplicates("method").set_index("method")["label"].to_dict()
    sort_keys = summary_df.drop_duplicates("method").set_index("method")["sort_key"].to_dict()

    pivot_dev = summary_df.pivot(
        index="obs_count",
        columns="method",
        values="mean_abs_deviation_from_true",
    )
    pivot_std = summary_df.pivot(index="obs_count", columns="method", values="avg_std")
    if sort_by_mean_deviation:
        methods = pivot_dev.mean(axis=0).sort_values().index.tolist()
    else:
        methods = sorted(pivot_dev.columns.tolist(), key=lambda method: sort_keys[method])
    pivot_dev = pivot_dev[methods]
    pivot_std = pivot_std[methods]
    obs_counts = pivot_dev.index.tolist()
    std_pivot = pivot_std[pivot_std.index <= 8]
    std_obs_counts = std_pivot.index.tolist()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    mean_over_obs = pivot_dev.mean(axis=0)
    std_over_obs = pivot_dev.std(axis=0)
    x = np.arange(len(methods))
    for idx, method in enumerate(methods):
        axes[0].bar(
            x[idx],
            mean_over_obs[method],
            yerr=std_over_obs[method],
            capsize=4,
            color=colors[idx % len(colors)],
        )

    axes[0].set_xticks([])
    axes[0].set_ylabel("Deviation from Oracle Propensity")
    axes[0].set_ylim(bottom=0)
    axes[0].yaxis.set_major_locator(MaxNLocator(nbins=5, min_n_ticks=5))
    axes[0].grid(True, axis="y", linestyle="--", alpha=0.35)

    for idx, method in enumerate(methods):
        axes[1].plot(
            std_obs_counts,
            std_pivot[method].values,
            marker="o",
            color=colors[idx % len(colors)],
            label=labels[method],
        )

    axes[1].set_xlabel("Observation Count")
    axes[1].set_ylabel("Average Standard Deviation")
    axes[1].set_xticks([x for x in [3, 6, 8] if x in std_obs_counts])
    axes[1].grid(True, linestyle="--", alpha=0.35)

    if title:
        fig.suptitle(title)

    legend_handles = [
        mpatches.Patch(color=colors[idx % len(colors)], label=labels[method])
        for idx, method in enumerate(methods)
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=min(len(methods), legend_ncol),
        frameon=False,
        bbox_to_anchor=(0.5, -0.04),
    )
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.25)
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    return output_path


def plot_summary_lines(
    summary_df: pd.DataFrame,
    output_path: Path,
    title: str | None = None,
    legend_ncol: int = 4,
    deviation_column: str = "mean_abs_deviation_from_true",
    deviation_ylabel: str = "Deviation from Oracle Propensity",
) -> Path:
    summary_df = summary_df.drop_duplicates(subset=["obs_count", "method"])
    labels = summary_df.drop_duplicates("method").set_index("method")["label"].to_dict()
    sort_keys = summary_df.drop_duplicates("method").set_index("method")["sort_key"].to_dict()

    pivot_dev = summary_df.pivot(
        index="obs_count",
        columns="method",
        values=deviation_column,
    )
    pivot_std = summary_df.pivot(index="obs_count", columns="method", values="avg_std")
    methods = sorted(pivot_dev.columns.tolist(), key=lambda method: sort_keys[method])
    pivot_dev = pivot_dev[methods]
    pivot_std = pivot_std[methods]
    pivot_dev = pivot_dev[pivot_dev.index <= 8]
    pivot_std = pivot_std[pivot_std.index <= 8]
    obs_counts = pivot_dev.index.tolist()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharex=True)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for idx, method in enumerate(methods):
        color = colors[idx % len(colors)]
        axes[0].plot(
            obs_counts,
            pivot_dev[method].values,
            marker="o",
            color=color,
            label=labels[method],
        )
        axes[1].plot(
            obs_counts,
            pivot_std[method].values,
            marker="o",
            color=color,
            label=labels[method],
        )

    for ax in axes:
        ax.set_xlabel("Observation Count")
        ax.set_xticks([x for x in [1, 3, 6, 8] if x in obs_counts])
        ax.grid(True, linestyle="--", alpha=0.35)

    axes[0].set_ylabel(deviation_ylabel)
    axes[0].set_ylim(bottom=0)
    axes[0].yaxis.set_major_locator(MaxNLocator(nbins=5, min_n_ticks=5))

    axes[1].set_ylabel("Average Standard Deviation")
    axes[1].set_ylim(bottom=0)
    axes[1].yaxis.set_major_locator(MaxNLocator(nbins=5, min_n_ticks=5))

    if title:
        fig.suptitle(title)

    legend_handles = [
        Line2D([0], [0], color=colors[idx % len(colors)], marker="o", label=labels[method])
        for idx, method in enumerate(methods)
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=min(len(methods), legend_ncol),
        frameon=False,
        bbox_to_anchor=(0.5, -0.04),
    )
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.25)
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    return output_path


def copy_outputs(paths: list[Path], thesis_images_dir: Path) -> int:
    thesis_images_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    for path in paths:
        target_name = THESIS_OUTPUT_ALIASES.get(path.name)
        if target_name is None:
            continue
        shutil.copy2(path, thesis_images_dir / target_name)
        copied += 1
    return copied


def final_comparison_summary(summary_df: pd.DataFrame) -> pd.DataFrame:
    selected_grouped_methods = {
        "cosine_t0p995_k50": "Cosine Similarity",
        "euclidean_t3p5_k50": "Euclidean Distance",
        "kmeans_g200_min1_it3": "K-Means",
        "knn_k2": "KNN",
        "mlp_regression_l4_h128_d0p0": "Propensity MLP",
    }
    grouped_families = {"cosine", "euclidean", "kmeans", "knn"}

    keep = (
        summary_df["family"].isin({"frequency_based", "mlp_classifier"})
        | (
            (summary_df["family"] == "mlp_regression")
            & (summary_df["method"] == "mlp_regression_l4_h128_d0p0")
        )
        | (
            summary_df["family"].isin(grouped_families)
            & summary_df["method"].isin(selected_grouped_methods)
        )
    )
    final_df = summary_df[keep].copy()
    final_df["label"] = final_df["method"].map(selected_grouped_methods).fillna(final_df["label"])
    return final_df


def euclidean_without_3p5_with_mlp_summary(summary_df: pd.DataFrame) -> pd.DataFrame:
    selected_methods = {
        "euclidean_t1p0_k50",
        "euclidean_t3_k50",
        "euclidean_t4_k50",
        "euclidean_t10_k50",
        "mlp_regression_l4_h128_d0p0",
    }
    keep = (
        summary_df["family"].eq("frequency_based")
        | summary_df["method"].isin(selected_methods)
    )
    sub = summary_df[keep].copy()
    sub.loc[sub["method"] == "mlp_regression_l4_h128_d0p0", "label"] = "Propensity MLP"
    sub.loc[sub["method"] == "mlp_regression_l4_h128_d0p0", "sort_key"] = pd.Series(
        [(15, 3.5, 50)] * (sub["method"] == "mlp_regression_l4_h128_d0p0").sum(),
        index=sub.index[sub["method"] == "mlp_regression_l4_h128_d0p0"],
    )
    return sub


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate thesis plots from compare_propensity_estimation CSV outputs."
    )
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--obs-counts", type=read_obs_counts, default=read_obs_counts("1,2,3,4"))
    parser.add_argument("--summary-obs-counts", type=read_obs_counts, default=read_obs_counts("1,2,3,4,5,6,7,8,9,10,11,12"))
    parser.add_argument("--copy-to-thesis", action="store_true")
    parser.add_argument("--thesis-images-dir", type=Path, default=DEFAULT_THESIS_IMAGES_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    runs = discover_runs(args.results_dir)
    print(f"Discovered {len(runs)} compare_propensity_estimation CSV files.")

    output_paths = []
    output_paths.append(plot_frequency_only_panel(runs[0], args.obs_counts, args.output_dir))
    for run in runs:
        output_paths.append(plot_multi_panel(run, args.obs_counts, args.output_dir))

    summary_df = create_summary_table(runs, args.summary_obs_counts)
    final_summary_df = final_comparison_summary(summary_df)
    output_paths.append(
        plot_summary(
            final_summary_df,
            args.output_dir / "propensity_estimation_FINAL_COMPARISON.pdf",
            title=None,
            sort_by_mean_deviation=True,
        )
    )
    output_paths.append(
        plot_summary(
            euclidean_without_3p5_with_mlp_summary(summary_df),
            args.output_dir / "propensity_estimation_Euclidean_no_3p5_with_MLP.pdf",
            title=None,
            legend_ncol=3,
        )
    )
    output_paths.append(
        plot_summary_lines(
            euclidean_without_3p5_with_mlp_summary(summary_df),
            args.output_dir / "propensity_estimation_Euclidean_no_3p5_with_MLP_lines.pdf",
            title=None,
            legend_ncol=3,
        )
    )
    output_paths.append(
        plot_summary_lines(
            euclidean_without_3p5_with_mlp_summary(summary_df),
            args.output_dir / "propensity_estimation_Euclidean_no_3p5_with_MLP_mean_lines.pdf",
            title=None,
            legend_ncol=3,
            deviation_column="abs_mean_deviation_from_true",
            deviation_ylabel="Deviation of Mean Propensity",
        )
    )

    family_filenames = {
        "cosine": "propensity_estimation_Cosine_Sim.pdf",
        "euclidean": "propensity_estimation_Euclidean.pdf",
        "kmeans": "propensity_estimation_KMeans.pdf",
        "knn": "propensity_estimation_KNN.pdf",
        "mlp_classifier": "propensity_estimation_MLP_Classifier.pdf",
        "mlp_regression": "propensity_estimation_MLP_Regression.pdf",
    }
    family_excluded_methods = {
        "euclidean": {"euclidean_t2_k50", "euclidean_t5_k50"},
    }
    family_legend_columns = {
        "euclidean": 3,
    }
    for family, filename in family_filenames.items():
        family_methods = set(summary_df.loc[summary_df["family"] == family, "method"])
        family_methods -= family_excluded_methods.get(family, set())
        if not family_methods:
            continue
        sub = summary_df[
            (summary_df["family"].isin({family, "frequency_based"}))
            & (summary_df["method"].isin(family_methods | {"frequency_based"}))
        ]
        output_paths.append(
            plot_summary(
                sub,
                args.output_dir / filename,
                title=None,
                legend_ncol=family_legend_columns.get(family, 4),
            )
        )

    if args.copy_to_thesis:
        copied_count = copy_outputs(output_paths, args.thesis_images_dir)
        print(f"Copied {copied_count} thesis-used PDFs to {args.thesis_images_dir}.")

    print(f"Generated {len(output_paths)} PDFs in {args.output_dir}.")


if __name__ == "__main__":
    main()
