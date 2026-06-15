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


def temp_namespace(temp: float) -> str:
    return f"two-tower-pl-temp-{temp:.1f}"


def main():
    datasets = ("istella", "mslr30k", "yahoo")
    metrics_to_plot = ("NDCG",)
    session_counts = (500, 1000, 5000, 10000, 50000, 100000)
    temperatures = (0.5, 1.0)

    for temp in temperatures:
        agg_parts = []
        baseline_parts = []
        namespace = temp_namespace(temp)

        print(f"\n=== Loading two_towers_pl results for temperature={temp:.1f} ===")

        for ds in datasets:
            print(f"\n--- Loading dataset: {ds} ---")
            cfg = dataclasses.replace(
                ExperimentConfig(dataset_name=ds),
                base_path_policy_models="results/empty",
                base_path_two_tower="results/two_towers_pl",
                experiment_name_policy="two_towers_pl",
                ips_models=("mul-two-tower", "add-two-tower"),
                propensity_models=("frequency-based",),
                n_sessions_list=session_counts,
                temperatures=(temp,),
                filter_single_display_pairs_values=(False, True),
                random_states=(40,),
            )

            try:
                df_long = load_long_metrics(cfg)
            except ValueError as e:
                print(f"No matching two_towers_pl runs for {ds} at temp={temp:.1f}: {e}")
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
            print(f"No datasets loaded for temperature={temp:.1f}; skipping plot generation.")
            continue

        agg_triple = pd.concat(agg_parts, ignore_index=True)
        baselines_triple = (
            pd.concat(baseline_parts, ignore_index=True) if baseline_parts else pd.DataFrame()
        )

        print(
            "\nGenerating triple-dataset two_towers_pl propensity estimator comparison plots "
            f"for temperature={temp:.1f}..."
        )
        plot_grid(
            agg_triple,
            baselines_triple,
            metrics_to_plot=metrics_to_plot,
            include_distance_models=False,
            filename_namespace=namespace,
            dataset_col="dataset_name",
            dataset_order=datasets,
        )

        print(
            "Generating combined additive-vs-multiplicative two_towers_pl plot "
            f"for temperature={temp:.1f}..."
        )
        plot_dm_dr_ips_naiveho_frequency_based(
            agg_triple,
            baselines=baselines_triple,
            filter_val=False,
            metrics_to_plot=metrics_to_plot,
            filename_namespace=namespace,
            dataset_col="dataset_name",
            dataset_order=datasets,
        )

    print("\nDone. two_towers_pl plots generated.")


if __name__ == "__main__":
    main()
