import dataclasses
import os
from pathlib import Path
import shutil
import sys

import pandas as pd

# Make sure we can import from notebooks.analysis
# The script is in notebooks/scripts, so project root is ../..
if os.path.basename(os.getcwd()) == "scripts":
    os.chdir("../..")
elif os.path.basename(os.getcwd()) == "notebooks":
    os.chdir("..")
sys.path.append(os.getcwd())

from notebooks.analysis import (
    ExperimentConfig,
    aggregate_mean_std,
    load_baselines_from_folder,
    load_long_metrics,
    plot_dm_dr_ips_naiveho_frequency_based,
    plot_grid,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
PLOTS_DIR = REPO_ROOT / "notebooks" / "thesis_plots"
THESIS_IMAGES_DIR = REPO_ROOT.parent / "Thesis" / "images"


def copy_to_thesis_images(*filenames: str) -> None:
    THESIS_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    for filename in filenames:
        src = PLOTS_DIR / filename
        if not src.exists():
            raise FileNotFoundError(f"Expected plot was not generated: {src}")
        shutil.copy2(src, THESIS_IMAGES_DIR / filename)


def add_percentage_aligned_session_counts(
    df: pd.DataFrame,
    *,
    dataset_col: str,
    x_col: str,
) -> pd.DataFrame:
    if df.empty:
        return df

    max_sessions = (
        df.groupby(dataset_col, as_index=False)[x_col]
        .max()
        .rename(columns={x_col: "dataset_max_sessions"})
    )
    aligned = df.merge(max_sessions, on=dataset_col, how="left")
    aligned["percentage_aligned_n_sessions"] = (
        aligned["dataset_max_sessions"] * aligned["n_session_percentage"] / 100.0
    )
    aligned["percentage_aligned_n_sessions"] = (
        aligned["percentage_aligned_n_sessions"].apply(lambda value: int(-(-value // 1)))
    )
    return aligned.drop(columns=["dataset_max_sessions"])


def main():
    datasets = ("istella", "mslr30k", "yahoo")
    metrics_to_plot = ("NDCG",)
    filename_namespace = "chunks"
    agg_parts = []
    baseline_parts = []

    for ds in datasets:
        print(f"\n--- Loading dataset: {ds} ---")
        cfg = dataclasses.replace(
            ExperimentConfig(dataset_name=ds),
            base_path_policy_models="results/disj_chunks/",
            experiment_name_policy="disj_chunks",
            ips_models=("dm", "dr", "ips"),
            propensity_models=("frequency-based", "true_propensity", "MLPregression"),
            n_session_percentage_list=(10, 25, 50, 75, 100),
            temperatures=(0.5,),
            random_states=(40, 41, 42),
            filter_single_display_pairs_values=(False, True),
        )

        try:
            df_long = load_long_metrics(cfg)
        except ValueError as e:
            print(f"Skipping dataset {ds}: {e}")
            continue

        agg = aggregate_mean_std(df_long).assign(dataset_name=ds)
        baselines = load_baselines_from_folder(
            cfg,
            metrics_to_plot=metrics_to_plot,
            baselines_path="results/true_baselines",
        ).assign(dataset_name=ds)

        agg_parts.append(agg)
        baseline_parts.append(baselines)

    if not agg_parts:
        raise RuntimeError("No datasets were loaded successfully, cannot generate triple plots.")

    agg_triple = pd.concat(agg_parts, ignore_index=True)
    baselines_triple = pd.concat(baseline_parts, ignore_index=True) if baseline_parts else pd.DataFrame()
    agg_triple = add_percentage_aligned_session_counts(
        agg_triple,
        dataset_col="dataset_name",
        x_col="n_sessions_used",
    )

    print("\nGenerating triple-dataset propensity estimator comparison plots...")
    plot_grid(
        agg_triple,
        baselines_triple,
        metrics_to_plot=metrics_to_plot,
        include_distance_models=False,
        filename_namespace=filename_namespace,
        dataset_col="dataset_name",
        dataset_order=datasets,
        x_col="percentage_aligned_n_sessions",
        x_label="Number of Sessions",
        x_scale="linear",
        sharex=False,
    )

    print("Generating triple-dataset all-model propensity estimator comparison plots...")
    # plot_grid(
    #     agg_triple,
    #     baselines_triple,
    #     metrics_to_plot=metrics_to_plot,
    #     include_distance_models=True,
    #     include_std_bars=False,
    #     filename_namespace=filename_namespace,
    #     dataset_col="dataset_name",
    #     dataset_order=datasets,
    #     x_col="n_session_percentage",
    #     x_label="Session Percentage (%)",
    #     x_scale="linear",
    # )

    try:
        print("Generating triple-dataset method comparison plots...")
        plot_dm_dr_ips_naiveho_frequency_based(
            agg_triple,
            baselines=baselines_triple,
            filter_val=False,
            metrics_to_plot=metrics_to_plot,
            filename_namespace=filename_namespace,
            dataset_col="dataset_name",
            dataset_order=datasets,
            x_col="percentage_aligned_n_sessions",
            x_label="Number of Sessions",
            x_scale="linear",
            sharex=False,
        )
    except ValueError as e:
        print(f"Skipping method comparison: {e}")

    copy_to_thesis_images(
        "chunks-prop-ips.pdf",
        "chunks-prop-dm.pdf",
        "chunks-prop-dr.pdf",
        "chunks-methods.pdf",
    )

    print("\nDone. Disjoint-chunks plots generated.")


if __name__ == "__main__":
    main()
