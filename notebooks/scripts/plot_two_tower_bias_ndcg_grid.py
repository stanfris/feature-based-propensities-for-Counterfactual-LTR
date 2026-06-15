import os
import shutil
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
PLOTS_DIR = REPO_ROOT / "notebooks" / "thesis_plots"
THESIS_IMAGES_DIR = REPO_ROOT.parent / "Thesis" / "images"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("MPLCONFIGDIR", str(PLOTS_DIR / ".matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(PLOTS_DIR / ".cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from notebooks.analysis.plot_style import (
    DENSE_TICK_LABEL_SIZE,
    apply_thesis_plot_style,
)
from generate_two_tower_bias_comparison_table import (
    OBJECTIVE_NAMES,
    PROP_NAMES,
    SOURCE_NAMES,
    build_summary,
    load_records,
    parse_random_states,
    sample_std,
)

apply_thesis_plot_style(plt)

OUTPUT_FILENAME = "two_tower_bias_position_bias_ndcg_grid.pdf"
PROP_ORDER = ("frequency-based", "MLPregression", "true_propensity")
SESSION_ORDER = (10000, 100000)
OBJECTIVE = "ips"
SOURCE_ORDER = (
    "real-targets",
    "add-two-tower",
    "mul-two-tower",
    "pb-adjacent_chain",
    "pb-ctr",
    "pb-global_all_pairs",
    "pb-pivot_one",
)
SOURCE_COLORS = {
    "real-targets": "#1f77b4",
    "add-two-tower": "#ff7f0e",
    "mul-two-tower": "#d62728",
    "pb-adjacent_chain": "#2ca02c",
    "pb-ctr": "#9467bd",
    "pb-global_all_pairs": "#8c564b",
    "pb-pivot_one": "#e377c2",
}


def format_panel(summary, propensity, n_sessions):
    means = []
    stds = []
    labels = []
    colors = []

    for source in SOURCE_ORDER:
        values = summary[(OBJECTIVE, propensity, source, n_sessions)]["NDCG"]
        means.append(sum(values) / len(values))
        stds.append(sample_std(values))
        labels.append(SOURCE_NAMES[source])
        colors.append(SOURCE_COLORS[source])

    return labels, means, stds, colors


def create_plot():
    all_records = []
    random_states = parse_random_states("40,41,42")
    for n_sessions in SESSION_ORDER:
        all_records.extend(
            load_records(
                dataset="mslr30k",
                n_sessions=n_sessions,
                temperature=0.5,
                random_states=random_states,
            )
        )

    grouped = {}
    for record in all_records:
        key = (
            record["objective"],
            record["propensity_model"],
            record["source"],
            record["n_sessions"],
        )
        if key not in grouped:
            grouped[key] = {"NDCG": [], "RCTR": []}
        for metric, value in record["metric_values"].items():
            grouped[key][metric].append(value)

    fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharey=True)
    axes = np.asarray(axes)

    for row_idx, n_sessions in enumerate(SESSION_ORDER):
        for col_idx, propensity in enumerate(PROP_ORDER):
            ax = axes[row_idx, col_idx]
            labels, means, stds, colors = format_panel(grouped, propensity, n_sessions)
            x = np.arange(len(labels))

            ax.bar(
                x,
                means,
                yerr=stds,
                color=colors,
                edgecolor="black",
                linewidth=0.6,
                error_kw={"elinewidth": 1.0, "capsize": 2},
            )
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=DENSE_TICK_LABEL_SIZE)
            ax.set_ylim(0.0, 0.55)
            ax.grid(axis="y", linestyle="--", alpha=0.4)
            ax.set_title(f"{n_sessions:,} Sessions | {PROP_NAMES[propensity]}")

            if col_idx == 0:
                ax.set_ylabel("NDCG")

    fig.suptitle(
        "IPS NDCG by Bias Estimator for Matched MSLR-WEB30K Runs",
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    return fig


def main():
    fig = create_plot()
    output_path = PLOTS_DIR / OUTPUT_FILENAME
    thesis_output_path = THESIS_IMAGES_DIR / OUTPUT_FILENAME
    THESIS_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    shutil.copy2(output_path, thesis_output_path)
    print(f"Saved {output_path}")
    print(f"Copied {thesis_output_path}")


if __name__ == "__main__":
    main()
