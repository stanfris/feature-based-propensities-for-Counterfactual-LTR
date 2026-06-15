import os
import sys
import dataclasses
from pathlib import Path
import shutil
import numpy as np
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
    plot_temperature_analysis,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
PLOTS_DIR = REPO_ROOT / "notebooks" / "thesis_plots"
THESIS_IMAGES_DIR = REPO_ROOT.parent / "Thesis" / "images"


def main():
    datasets = ["mslr30k", "istella"]
    
    for ds in datasets:
        print(f"\n--- Processing dataset: {ds} ---")
        try:
            cfg = ExperimentConfig(dataset_name=ds)
            print("Loading baselines...")
            baselines = load_baselines_from_folder(cfg, metrics_to_plot=("RCTR", "NDCG"))

            print("Loading temperature configuration results...")
            cfg_temp = dataclasses.replace(
                cfg,
                base_path_policy_models="results/temp/",
                launcher_policy="slurmcpu",
                experiment_name_policy="temp",
                ips_models=("ips",),
                n_sessions_list=(5000,),
                temperatures=tuple(round(x, 1) for x in np.arange(0.0, 1.0 + 1e-9, 0.1)),
                random_states=np.arange(0, 54)
            )
            df_temp = load_long_metrics(cfg_temp)

            new_policy_dir = REPO_ROOT / "results" / "new_policy"
            if new_policy_dir.exists():
                print("Loading optional new_policy configuration result for tmp=0.5...")
                cfg_temp_05 = dataclasses.replace(
                    cfg,
                    base_path_policy_models="results/new_policy/",
                    launcher_policy="slurm",
                    experiment_name_policy="new_policy",
                    ips_models=("ips",),
                    n_sessions_list=(5000,),
                    temperatures=(0.5,),
                    random_states=np.arange(0, 54)
                )
                try:
                    df_temp_05 = load_long_metrics(cfg_temp_05)
                except ValueError as e:
                    print(f"Skipping optional new_policy results for {ds}: {e}")
                    df_temp_05 = pd.DataFrame()
            else:
                print("Skipping optional new_policy results; directory is not present.")
                df_temp_05 = pd.DataFrame()

            if df_temp.empty and df_temp_05.empty:
                print(f"Skipping temperature analysis for {ds}: No results found.")
                continue

            print("Aggregating temperature results...")
            df_temp_combined = pd.concat([df_temp, df_temp_05], ignore_index=True)
            agg_temp = aggregate_mean_std(df_temp_combined)

            print(f"Generating temperature analysis plots for {ds}...")
            plot_temperature_analysis(
                agg_temp,
                baselines=baselines,
                metrics_to_plot=("RCTR", "NDCG"),
                n_sessions_list=(5000,),
                include_distance_models=False,
                filename_suffix=f"_{ds}"
            )
            if ds == "mslr30k":
                filename = "Temperature_Analysis_merged_mslr30k.pdf"
                shutil.copy2(PLOTS_DIR / filename, THESIS_IMAGES_DIR / filename)
        except Exception as e:
            print(f"An error occurred while processing {ds}: {e}")

    print("\nDone. Temperature analysis plots generated.")

if __name__ == "__main__":
    main()
