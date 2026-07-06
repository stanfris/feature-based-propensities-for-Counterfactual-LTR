import dataclasses
import os
from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
os.chdir(REPO_ROOT)
sys.path.insert(0, str(REPO_ROOT))

from result_parsing.analysis import (
    ExperimentConfig,
    aggregate_mean_std,
    load_baselines_from_folder,
    load_long_metrics,
    plot_dm_dr_ips_naiveho_frequency_based,
    plot_grid,
    plot_ips_dm_dr_stacked_propensity_grid,
)


RANDOM_STATES = tuple(range(40, 60))


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
        interval_mode="ci",
    )

    print("Generating stacked IPS/DM/DR real-target propensity plot...")
    plot_ips_dm_dr_stacked_propensity_grid(
        agg_triple,
        baselines_triple,
        metric="NDCG",
        filename_namespace=filename_namespace,
        dataset_col="dataset_name",
        dataset_order=datasets,
        interval_mode="ci",
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
            interval_mode="ci",
        )
    except ValueError as e:
        print(f"Skipping method comparison: {e}")

    print("\nDone. Real-target plots generated.")


if __name__ == "__main__":
    main()
