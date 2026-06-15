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
)

def main():
    datasets = ("istella", "mslr30k")
    metrics_to_plot = ("NDCG",)
    dual_dataset_plot = True
    metric_suffix = "_".join(metric.lower() for metric in metrics_to_plot)

    if dual_dataset_plot:
        agg_parts = []
        baseline_parts = []

        for ds in datasets:
            print(f"\n--- Loading dataset: {ds} ---")
            try:
                cfg = ExperimentConfig(dataset_name=ds)
                df_long = load_long_metrics(cfg)
                agg = aggregate_mean_std(df_long).assign(dataset_name=ds)
                baselines = load_baselines_from_folder(cfg, metrics_to_plot=metrics_to_plot).assign(dataset_name=ds)

                agg_parts.append(agg)
                baseline_parts.append(baselines)
            except Exception as e:
                print(f"An error occurred while processing {ds}: {e}")

        if not agg_parts:
            raise RuntimeError("No datasets were loaded successfully, cannot generate dual plots.")

        agg_dual = pd.concat(agg_parts, ignore_index=True)
        baselines_dual = pd.concat(baseline_parts, ignore_index=True) if baseline_parts else pd.DataFrame()

        print("\nGenerating dual-dataset method comparison plots...")
        plot_dm_dr_ips_naiveho_frequency_based(
            agg_dual,
            baselines=baselines_dual,
            filter_val=False,
            metrics_to_plot=metrics_to_plot,
            filename_suffix=f"_dual_{metric_suffix}",
            dataset_col="dataset_name",
            dataset_order=datasets,
        )
    else:
        for ds in datasets:
            print(f"\n--- Processing dataset: {ds} ---")
            try:
                cfg = ExperimentConfig(dataset_name=ds)
                df_long = load_long_metrics(cfg)
                agg = aggregate_mean_std(df_long)
                baselines = load_baselines_from_folder(cfg, metrics_to_plot=metrics_to_plot)

                print(f"Generating method comparison plots for {ds}...")
                plot_dm_dr_ips_naiveho_frequency_based(
                    agg,
                    baselines=baselines,
                    filter_val=False,
                    metrics_to_plot=metrics_to_plot,
                    filename_suffix=f"_{ds}_{metric_suffix}",
                )
            except Exception as e:
                print(f"An error occurred while processing {ds}: {e}")

    print("\nDone. Method comparison plots generated.")

if __name__ == "__main__":
    main()
