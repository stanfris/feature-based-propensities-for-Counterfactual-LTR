import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from notebooks.scripts.plot_compare_propensity_estimation import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_RESULTS_DIR,
    apply_propensity_clip,
    configure_propensity_axis,
    create_summary_table,
    discover_runs,
    logit_band_to_propensity,
    logits_to_propensity,
    plot_summary_lines,
)


DEFAULT_CLIPPED_RESULTS_DIR = REPO_ROOT / "results" / "clip_prop"
TARGET_CLIPS = (0.08, 0.1001)
SUMMARY_OBS_COUNTS = list(range(1, 13))
PANEL_OBS_COUNTS = [1, 2, 3, 4]


def select_clipped_runs(runs, targets=TARGET_CLIPS, atol=7e-4):
    selected = []
    for target in targets:
        matches = [
            run
            for run in runs
            if run.clip_value is not None and np.isclose(run.clip_value, target, atol=atol)
        ]
        if not matches:
            raise RuntimeError(f"No clipped frequency-based run found for clip {target}.")
        selected.append(sorted(matches, key=lambda run: abs(run.clip_value - target))[0])
    return selected


def selected_line_summary(runs, clipped_runs):
    summary_df = create_summary_table(runs + clipped_runs, SUMMARY_OBS_COUNTS)
    selected_methods = {
        "euclidean_t1p0_k50",
        "euclidean_t3_k50",
        "euclidean_t4_k50",
        "euclidean_t10_k50",
        "mlp_regression_l4_h128_d0p0",
    }
    keep = (
        summary_df["family"].isin({"frequency_based", "frequency_based_clipped"})
        | summary_df["method"].isin(selected_methods)
    )
    sub = summary_df[keep].copy()
    sub.loc[sub["method"] == "mlp_regression_l4_h128_d0p0", "label"] = "Propensity MLP"
    sub.loc[sub["method"] == "mlp_regression_l4_h128_d0p0", "sort_key"] = pd.Series(
        [(15, 3.5, 50)] * (sub["method"] == "mlp_regression_l4_h128_d0p0").sum(),
        index=sub.index[sub["method"] == "mlp_regression_l4_h128_d0p0"],
    )
    return sub


def plot_mlp_regressor_with_clipping(mlp_run, clipped_runs, output_dir: Path) -> Path:
    df = pd.read_csv(mlp_run.csv_path)
    positions = df["position"].to_numpy()
    effective_alpha = logits_to_propensity(df["effective_alpha"].to_numpy())
    clipped_runs = sorted(clipped_runs, key=lambda run: run.clip_value or 0.0)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    n_cols = 2
    n_rows = int(np.ceil(len(PANEL_OBS_COUNTS) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows), sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()

    for idx, obs_count in enumerate(PANEL_OBS_COUNTS):
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

        expected = logits_to_propensity(mean_expected[valid])
        expected_lower, expected_upper = logit_band_to_propensity(
            mean_expected[valid],
            std_expected[valid],
        )
        pred = logits_to_propensity(mean_pred[valid])
        pred_lower, pred_upper = logit_band_to_propensity(
            mean_pred[valid],
            std_pred[valid],
        )

        ax.plot(
            positions,
            effective_alpha,
            label="Oracle Propensity",
            color="black",
            linestyle="-.",
            linewidth=2.0,
            zorder=5,
        )
        ax.plot(
            pos_valid,
            expected,
            label="Frequency-Based Propensity",
            marker="s",
            markersize=6,
            color="tab:blue",
            alpha=0.8,
            zorder=4,
        )
        ax.fill_between(
            pos_valid,
            expected_lower,
            expected_upper,
            alpha=0.2,
            color="tab:blue",
            label="_nolegend_",
        )

        for run_idx, run in enumerate(clipped_runs):
            color = colors[(run_idx + 2) % len(colors)]
            clipped = apply_propensity_clip(expected, run.clip_value)
            clipped_lower = apply_propensity_clip(expected_lower, run.clip_value)
            clipped_upper = apply_propensity_clip(expected_upper, run.clip_value)
            ax.plot(
                pos_valid,
                clipped,
                label=run.label,
                marker="o",
                markersize=4,
                color=color,
                alpha=0.82,
                zorder=3,
            )
            ax.fill_between(
                pos_valid,
                clipped_lower,
                clipped_upper,
                alpha=0.14,
                color=color,
                label="_nolegend_",
            )

        ax.plot(
            pos_valid,
            pred,
            label="Propensity MLP",
            marker="o",
            markersize=5,
            color="tab:orange",
            alpha=0.8,
            zorder=4,
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

    for ax in axes[2 : len(PANEL_OBS_COUNTS)]:
        ax.set_xlabel("Position")
    for ax in axes[::n_cols]:
        ax.set_ylabel("Propensity")

    handles, labels = axes[0].get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    fig.legend(
        unique.values(),
        unique.keys(),
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.54, 0.0),
    )
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.16, bottom=0.22)

    output_path = output_dir / "propensity_estimation_frequency_based_MLPRegressor.pdf"
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the selected clipped propensity line comparison plot."
    )
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--clipped-results-dir", type=Path, default=DEFAULT_CLIPPED_RESULTS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR / "clipping")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    runs = discover_runs(args.results_dir)
    clipped_runs = select_clipped_runs(discover_runs(args.clipped_results_dir))
    mlp_runs = [run for run in runs if run.method == "mlp_regression_l4_h128_d0p0"]
    if not mlp_runs:
        raise RuntimeError("No MLP regression l4 h128 d0p0 run found.")
    summary_df = selected_line_summary(runs, clipped_runs)
    outputs = [
        plot_summary_lines(
        summary_df,
        args.output_dir / "propensity_estimation_Euclidean_no_3p5_with_MLP_lines.pdf",
        title=None,
        legend_ncol=3,
        legend_y=-0.12,
        legend_columnspacing=1.1,
        legend_handletextpad=0.45,
        ),
        plot_mlp_regressor_with_clipping(mlp_runs[0], clipped_runs, args.output_dir),
    ]
    for output in outputs:
        print(f"Generated {output}")


if __name__ == "__main__":
    main()
