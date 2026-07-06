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
from matplotlib.patches import Patch

from notebooks.analysis.plot_style import (
    DENSE_ANNOTATION_SIZE,
    DENSE_PANEL_TITLE_SIZE,
    DENSE_SUPTITLE_SIZE,
    DENSE_TICK_LABEL_SIZE,
    apply_thesis_plot_style,
)
from generate_two_tower_bias_comparison_table import (
    OBJECTIVE_NAMES,
    load_records,
    parse_random_states,
)
from table_significance import sample_std

apply_thesis_plot_style(plt)

OUTPUT_FILENAME = "two_tower_bias_frequency_vs_mlp_all_results.pdf"
DATASET_ORDER = ("mslr30k", "yahoo", "istella")
DATASET_NAMES = {
    "mslr30k": "MSLR-WEB30K",
    "yahoo": "Yahoo!",
    "istella": "Istella-S",
}
SESSION_ORDER = (10000, 100000)
OBJECTIVE_ORDER = ("ips", "dm", "dr")
PROPENSITY_ORDER = ("frequency-based", "MLPregression")
PROPENSITY_LABELS = {
    "frequency-based": "Frequency-based Propensity",
    "MLPregression": "MLP Propensity",
}
PROPENSITY_COLORS = {
    "frequency-based": "#F58518",
    "MLPregression": "#4C78A8",
}
SOURCE_ORDER = (
    "real-targets",
    "add-two-tower",
    "mul-two-tower",
    "pb-adjacent_chain",
    "pb-ctr",
    "pb-global_all_pairs",
    "pb-pivot_one",
)
SOURCE_SHORT_NAMES = {
    "real-targets": "Real",
    "add-two-tower": "Add",
    "mul-two-tower": "Mul",
    "pb-adjacent_chain": "Adj",
    "pb-ctr": "CTR",
    "pb-global_all_pairs": "Global",
    "pb-pivot_one": "Pivot",
}
DATASET_GAP = 1.4
BAR_WIDTH = 0.36


def build_positions():
    positions = []
    xticks = []
    xticklabels = []
    dataset_blocks = {}
    cursor = 0.0

    for dataset in DATASET_ORDER:
        block_positions = []
        for source in SOURCE_ORDER:
            positions.append((dataset, source, cursor))
            xticks.append(cursor)
            xticklabels.append(SOURCE_SHORT_NAMES[source])
            block_positions.append(cursor)
            cursor += 1.0
        dataset_blocks[dataset] = {
            "start": block_positions[0],
            "end": block_positions[-1],
            "center": sum(block_positions) / len(block_positions),
        }
        cursor += DATASET_GAP

    separators = []
    for left, right in zip(DATASET_ORDER, DATASET_ORDER[1:]):
        separators.append((dataset_blocks[left]["end"] + dataset_blocks[right]["start"]) / 2.0)

    return positions, xticks, xticklabels, dataset_blocks, separators


def aggregate_records():
    random_states = parse_random_states("40,41,42")
    grouped = {}

    for dataset in DATASET_ORDER:
        for n_sessions in SESSION_ORDER:
            for record in load_records(
                dataset=dataset,
                n_sessions=n_sessions,
                temperature=0.5,
                random_states=random_states,
            ):
                if record["propensity_model"] not in PROPENSITY_ORDER:
                    continue
                key = (
                    dataset,
                    record["objective"],
                    record["n_sessions"],
                    record["source"],
                    record["propensity_model"],
                )
                grouped.setdefault(key, []).append(record["metric_values"]["NDCG"])

    if not grouped:
        raise RuntimeError("No matched records found for the frequency-vs-MLP comparison plot.")
    return grouped


def compute_ylim(grouped):
    upper = 0.0
    for values in grouped.values():
        mean_value = sum(values) / len(values)
        upper = max(upper, mean_value + sample_std(values))
    return 0.0, min(0.9, upper + 0.04)


def draw_panel(ax, grouped, objective, n_sessions, xticks, xticklabels, dataset_blocks, separators, show_xlabels):
    for dataset, source, center in POSITIONS:
        for propensity, offset in (("frequency-based", -BAR_WIDTH / 2), ("MLPregression", BAR_WIDTH / 2)):
            values = grouped.get((dataset, objective, n_sessions, source, propensity))
            if not values:
                continue
            mean_value = sum(values) / len(values)
            std_value = sample_std(values)
            ax.bar(
                center + offset,
                mean_value,
                width=BAR_WIDTH,
                yerr=std_value,
                color=PROPENSITY_COLORS[propensity],
                edgecolor="black",
                linewidth=0.5,
                error_kw={"elinewidth": 0.9, "capsize": 2},
                zorder=3,
            )

    for separator in separators:
        ax.axvline(separator, color="0.75", linewidth=0.8, linestyle="--", zorder=1)

    ax.grid(axis="y", linestyle="--", alpha=0.35, zorder=0)
    ax.set_xticks(xticks)

    if show_xlabels:
        ax.set_xticklabels(xticklabels, rotation=45, ha="right", fontsize=DENSE_TICK_LABEL_SIZE)
        for dataset, block in dataset_blocks.items():
            ax.text(
                block["center"],
                -0.26,
                DATASET_NAMES[dataset],
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="top",
                fontsize=DENSE_ANNOTATION_SIZE,
                fontweight="bold",
            )
    else:
        ax.set_xticklabels([])
        ax.tick_params(axis="x", length=0)


def create_plot():
    grouped = aggregate_records()
    ylim = compute_ylim(grouped)

    fig, axes = plt.subplots(2, 3, figsize=(22, 11), sharex=True, sharey=True)
    axes = np.asarray(axes)

    for row_idx, n_sessions in enumerate(SESSION_ORDER):
        for col_idx, objective in enumerate(OBJECTIVE_ORDER):
            ax = axes[row_idx, col_idx]
            draw_panel(
                ax=ax,
                grouped=grouped,
                objective=objective,
                n_sessions=n_sessions,
                xticks=XTICKS,
                xticklabels=XTLABELS,
                dataset_blocks=DATASET_BLOCKS,
                separators=SEPARATORS,
                show_xlabels=row_idx == len(SESSION_ORDER) - 1,
            )
            ax.set_ylim(*ylim)
            if row_idx == 0:
                ax.set_title(OBJECTIVE_NAMES[objective], fontsize=DENSE_PANEL_TITLE_SIZE)
            if col_idx == 0:
                ax.set_ylabel(f"NDCG\n{n_sessions:,} Sessions")

    fig.suptitle(
        "Two-Tower Bias Results: Frequency-Based vs Feature-Based Propensity Estimation",
        fontsize=DENSE_SUPTITLE_SIZE,
        y=0.995,
    )
    fig.legend(
        handles=[
            Patch(facecolor=PROPENSITY_COLORS[propensity], edgecolor="black", label=PROPENSITY_LABELS[propensity])
            for propensity in PROPENSITY_ORDER
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=2,
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0.08, 1, 0.97))
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


POSITIONS, XTICKS, XTLABELS, DATASET_BLOCKS, SEPARATORS = build_positions()


if __name__ == "__main__":
    main()
