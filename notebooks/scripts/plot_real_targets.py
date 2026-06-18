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
RANDOM_STATES = tuple(range(40, 60))


def copy_to_thesis_images(*filenames: str) -> None:
    THESIS_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    for filename in filenames:
        src = PLOTS_DIR / filename
        if not src.exists():
            raise FileNotFoundError(f"Expected plot was not generated: {src}")
        shutil.copy2(src, THESIS_IMAGES_DIR / filename)


def main():
    datasets = ("istella", "mslr30k", "yahoo")
    metrics_to_plot = ("NDCG",)
    filename_namespace = "real"
    agg_parts = []
    method_agg_parts = []
    baseline_parts = []

    for ds in datasets:
        print(f"\n--- Loading dataset: {ds} ---")
        cfg = dataclasses.replace(
            ExperimentConfig(dataset_name=ds),
            base_path_policy_models="results/real_targets/",
            experiment_name_policy="real_targets",
            ips_models=("dm", "dr", "ips", "naive"),
            propensity_models=("frequency-based", "true_propensity", "MLPregression"),
            n_sessions_list=(500, 1000, 2500, 5000, 10000, 50000, 100000, 500000, 1000000),
            temperatures=(0.5,),
            random_states=RANDOM_STATES,
            filter_single_display_pairs_values=(False, True),
        )

        try:
            df_long = load_long_metrics(cfg)
        except ValueError as e:
            print(f"Skipping dataset {ds}: {e}")
            continue

        agg = aggregate_mean_std(df_long).assign(dataset_name=ds)
        method_cfg = dataclasses.replace(
            cfg,
            base_path_two_tower="results/two_towers",
            ips_models=("dm", "dr", "ips", "naive", "mul-two-tower", "add-two-tower"),
        )
        try:
            method_df_long = load_long_metrics(method_cfg)
        except ValueError as e:
            print(f"Skipping two-tower method overlay for {ds}: {e}")
            method_agg = agg
        else:
            method_agg = aggregate_mean_std(method_df_long).assign(dataset_name=ds)

        baselines = load_baselines_from_folder(
            cfg,
            metrics_to_plot=metrics_to_plot,
            baselines_path="results/true_baselines",
        ).assign(dataset_name=ds)

        agg_parts.append(agg)
        method_agg_parts.append(method_agg)
        baseline_parts.append(baselines)

    if not agg_parts:
        raise RuntimeError("No datasets were loaded successfully, cannot generate triple plots.")

    agg_triple = pd.concat(agg_parts, ignore_index=True)
    method_agg_triple = pd.concat(method_agg_parts, ignore_index=True)
    baselines_triple = pd.concat(baseline_parts, ignore_index=True) if baseline_parts else pd.DataFrame()

    print("\nGenerating triple-dataset propensity estimator comparison plots...")
    plot_grid(
        agg_triple,
        baselines_triple,
        metrics_to_plot=metrics_to_plot,
        include_distance_models=False,
        filename_namespace=filename_namespace,
        dataset_col="dataset_name",
        dataset_order=datasets,
    )

    try:
        print("Generating triple-dataset method comparison plots...")
        plot_dm_dr_ips_naiveho_frequency_based(
            method_agg_triple,
            baselines=baselines_triple,
            filter_val=False,
            metrics_to_plot=metrics_to_plot,
            filename_namespace=filename_namespace,
            dataset_col="dataset_name",
            dataset_order=datasets,
        )
    except ValueError as e:
        print(f"Skipping method comparison: {e}")

    copy_to_thesis_images(
        "real-prop-ips.pdf",
        "real-prop-dm.pdf",
        "real-prop-dr.pdf",
        "real-methods.pdf",
    )

    print("\nDone. Real-target plots generated.")


if __name__ == "__main__":
    main()
