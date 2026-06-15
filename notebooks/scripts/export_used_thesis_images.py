import shutil
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
PLOTS_DIR = REPO_ROOT / "notebooks" / "thesis_plots"
THESIS_IMAGES_DIR = REPO_ROOT.parent / "Thesis" / "images"

USED_PLOT_EXPORTS = {
    "Click_dataset_counts_histogram.pdf": "Click_dataset_counts_histogram.pdf",
    "propensity_estimation_Cosine_Sim.pdf": "propensity_estimation_Cosine_Sim.pdf",
    "propensity_estimation_FINAL_COMPARISON.pdf": "propensity_estimation_FINAL_COMPARISON.pdf",
    "propensity_estimation_KMeans.pdf": "propensity_estimation_KMeans.pdf",
    "propensity_estimation_KNN.pdf": "propensity_estimation_KNN.pdf",
    "propensity_estimation_MLP_Classifier.pdf": "propensity_estimation_MLP_Classifier.pdf",
    "propensity_estimation_frequency_based_MLPClassifier.pdf": "propensity_estimation_frequency_based_MLPClassifier.pdf",
    "propensity_estimation_frequency_based_MLPRegressor.pdf": "propensity_estimation_frequency_based_MLPRegressor.pdf",
    "propensity_estimation_frequency_based_cosine.pdf": "propensity_estimation_frequency_based_cosine.pdf",
    "propensity_estimation_frequency_based_none.pdf": "propensity_estimation_frequency_based_none.pdf",
    "two_tower_bias_position_bias_ndcg_grid.pdf": "two_tower_bias_position_bias_ndcg_grid.pdf",
    "two_tower_bias_frequency_vs_mlp_all_results.pdf": "two_tower_bias_frequency_vs_mlp_all_results.pdf",
    "two_tower_position_bias_vs_csv_methods.pdf": "two_tower_position_bias_vs_csv_methods.pdf",
}


def main() -> None:
    THESIS_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    for source_name, target_name in USED_PLOT_EXPORTS.items():
        source_path = PLOTS_DIR / source_name
        if not source_path.exists():
            raise FileNotFoundError(f"Missing thesis plot source: {source_path}")
        shutil.copy2(source_path, THESIS_IMAGES_DIR / target_name)
        print(f"Copied {source_name} -> {target_name}")


if __name__ == "__main__":
    main()
