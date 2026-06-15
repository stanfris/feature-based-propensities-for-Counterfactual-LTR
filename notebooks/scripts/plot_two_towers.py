import dataclasses
import os
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


def main():
    datasets = ("istella", "mslr30k", "yahoo")
    metrics_to_plot = ("NDCG",)
    filename_namespace = "two-tower"
    agg_parts = []
    baseline_parts = []

    for ds in datasets:
        print(f"\n--- Loading dataset: {ds} ---")
        cfg = dataclasses.replace(
            ExperimentConfig(dataset_name=ds),
            base_path_policy_models="results/empty",
            base_path_two_tower="results/two_towers",
            experiment_name_policy="two_towers",
            ips_models=("mul-two-tower", "add-two-tower"),
            propensity_models=("frequency-based",),
            n_sessions_list=(500, 1000, 5000, 10000, 50000, 100000, 500000, 1000000),
            temperatures=(0.5,),
            filter_single_display_pairs_values=(False, True),
        )

        try:
            df_long = load_long_metrics(cfg)
        except ValueError as e:
            print(f"No matching two-tower runs for {ds}: {e}")
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
        raise RuntimeError("No two-tower datasets were loaded successfully, cannot generate plots.")

    agg_triple = pd.concat(agg_parts, ignore_index=True)
    baselines_triple = pd.concat(baseline_parts, ignore_index=True) if baseline_parts else pd.DataFrame()

    print("\nGenerating triple-dataset two-tower propensity estimator comparison plots...")
    plot_grid(
        agg_triple,
        baselines_triple,
        metrics_to_plot=metrics_to_plot,
        include_distance_models=False,
        filename_namespace=filename_namespace,
        dataset_col="dataset_name",
        dataset_order=datasets,
    )

    print("Generating combined additive-vs-multiplicative two-tower plot...")
    plot_dm_dr_ips_naiveho_frequency_based(
        agg_triple,
        baselines=baselines_triple,
        filter_val=False,
        metrics_to_plot=metrics_to_plot,
        filename_namespace=filename_namespace,
        dataset_col="dataset_name",
        dataset_order=datasets,
    )

    print("\nDone. Two-tower plots generated.")


if __name__ == "__main__":
    main()
