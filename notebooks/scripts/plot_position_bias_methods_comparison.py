import argparse
import csv
import math
import os
import re
import shutil
import sys
from collections import defaultdict
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
from matplotlib.lines import Line2D

from notebooks.analysis.plot_style import apply_thesis_plot_style

apply_thesis_plot_style(plt)

DEFAULT_RESULTS_ROOT = REPO_ROOT / "results" / "two_towers"
DEFAULT_SMOKE_CSV = REPO_ROOT / "results" / "position_bias_csv_smoke.csv"
DEFAULT_OUTPUT_PATH = PLOTS_DIR / "two_tower_position_bias_vs_csv_methods.pdf"
SESSION_COUNTS = (10000, 100000)
CSV_METHODS = ("adjacent_chain", "ctr", "global_all_pairs", "pivot_one")
TWO_TOWER_CONFIGS = (
    ("add-two-tower", "additive_position_bias.csv", "Additive Two-Tower"),
    ("mul-two-tower", "multiplicative_position_bias.csv", "Multiplicative Two-Tower"),
)
CSV_METHOD_LABELS = {
    "adjacent_chain": "Adjacent Chain",
    "ctr": "CTR",
    "global_all_pairs": "Global All Pairs",
    "pivot_one": "Pivot One",
}
CSV_METHOD_COLORS = {
    "adjacent_chain": "#2ca02c",
    "ctr": "#9467bd",
    "global_all_pairs": "#8c564b",
    "pivot_one": "#e377c2",
}
TWO_TOWER_STYLES = {
    "add-two-tower": {"color": "#1f77b4", "marker": "o", "label": "Additive Two-Tower"},
    "mul-two-tower": {"color": "#ff7f0e", "marker": "s", "label": "Multiplicative Two-Tower"},
}


def compute_true_bias(positions: np.ndarray, strength: float = 1.0) -> np.ndarray:
    values = -strength * np.log(positions + 1)
    return values - values[0]


def sample_std(values):
    if len(values) < 2:
        return 0.0
    mean_value = sum(values) / len(values)
    variance = sum((value - mean_value) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(variance)


def load_two_tower_summary(results_root: Path, ips_model: str, csv_name: str):
    pattern = re.compile(
        r"data=(?P<dataset>[^,]+).*?ips\.model=(?P<ips_model>[^,]+).*?ips\.n_sessions=(?P<n_sessions>\d+).*?random_state=(?P<random_state>\d+)"
    )
    grouped = defaultdict(list)

    for csv_path in sorted(results_root.glob(f"*/{csv_name}")):
        match = pattern.search(str(csv_path.parent))
        if match is None:
            continue
        if match.group("dataset") != "mslr30k" or match.group("ips_model") != ips_model:
            continue

        n_sessions = int(match.group("n_sessions"))
        if n_sessions not in SESSION_COUNTS:
            continue

        with csv_path.open("r", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                position = int(row["position"])
                value = float(row["relative_position_bias"])
                grouped[(n_sessions, position)].append(value)

    summary = {}
    for key, values in grouped.items():
        summary[key] = {
            "mean": sum(values) / len(values),
            "std": sample_std(values),
        }
    return summary


def load_csv_method_curves(smoke_csv_path: Path):
    curves = {}
    with smoke_csv_path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row["dataset"] != "MSLR30K":
                continue
            n_sessions = int(row["n_sessions_used"])
            if n_sessions not in SESSION_COUNTS:
                continue
            estimator = row["estimator"]
            if estimator not in CSV_METHODS:
                continue
            position = int(row["position"])
            value = float(row["relative_position_bias_logit"])
            curves[(estimator, n_sessions, position)] = value
    return curves


def positions_from_summary(summary):
    return sorted({position for (_, position) in summary.keys()})


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Plot additive and multiplicative two-tower position bias against CSV position-bias estimators."
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=DEFAULT_RESULTS_ROOT,
        help=f"Directory containing two_towers run folders. Default: {DEFAULT_RESULTS_ROOT}",
    )
    parser.add_argument(
        "--smoke-csv",
        type=Path,
        default=DEFAULT_SMOKE_CSV,
        help=f"CSV file with position-bias estimator outputs. Default: {DEFAULT_SMOKE_CSV}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Output PDF path. Default: {DEFAULT_OUTPUT_PATH}",
    )
    return parser


def plot_panel(ax, n_sessions, two_tower_summaries, csv_curves):
    positions = np.array(
        sorted(
            {
                position
                for summary in two_tower_summaries.values()
                for (_, position) in summary.keys()
            }
        ),
        dtype=float,
    )
    true_bias = compute_true_bias(positions)

    ax.plot(positions, true_bias, color="black", linestyle=":", linewidth=2.0, zorder=4)

    for ips_model, _, _ in TWO_TOWER_CONFIGS:
        curve_positions = []
        means = []
        stds = []
        for position in positions.astype(int):
            point = two_tower_summaries[ips_model].get((n_sessions, position))
            if point is None:
                continue
            curve_positions.append(position)
            means.append(point["mean"])
            stds.append(point["std"])

        x = np.array(curve_positions, dtype=float)
        y = np.array(means, dtype=float)
        e = np.array(stds, dtype=float)
        style = TWO_TOWER_STYLES[ips_model]

        ax.plot(
            x,
            y,
            color=style["color"],
            linewidth=2.2,
            marker=style["marker"],
            markersize=4,
            label=style["label"],
            zorder=3,
        )

    for estimator in CSV_METHODS:
        x = []
        y = []
        for position in positions.astype(int):
            value = csv_curves.get((estimator, n_sessions, position))
            if value is None:
                continue
            x.append(position)
            y.append(value)

        ax.plot(
            np.array(x, dtype=float),
            np.array(y, dtype=float),
            color=CSV_METHOD_COLORS[estimator],
            linewidth=1.8,
            alpha=0.95,
            zorder=1,
        )

    ax.set_title(f"{n_sessions:,} Sessions")
    ax.set_xlabel("Position")
    ax.set_ylabel("Relative position bias")
    ax.grid(True, linestyle="--", alpha=0.45)


def main():
    args = build_arg_parser().parse_args()
    csv_curves = load_csv_method_curves(args.smoke_csv.resolve())
    two_tower_summaries = {
        ips_model: load_two_tower_summary(args.results_root.resolve(), ips_model, csv_name)
        for ips_model, csv_name, _ in TWO_TOWER_CONFIGS
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharex=True, sharey=True)

    for ax, n_sessions in zip(axes, SESSION_COUNTS):
        plot_panel(ax, n_sessions, two_tower_summaries, csv_curves)

    method_handles = [
        Line2D([0], [0], color="black", linestyle=":", linewidth=2, label="True bias"),
        Line2D(
            [0],
            [0],
            color=TWO_TOWER_STYLES["add-two-tower"]["color"],
            marker=TWO_TOWER_STYLES["add-two-tower"]["marker"],
            linewidth=2.2,
            label=TWO_TOWER_STYLES["add-two-tower"]["label"],
        ),
        Line2D(
            [0],
            [0],
            color=TWO_TOWER_STYLES["mul-two-tower"]["color"],
            marker=TWO_TOWER_STYLES["mul-two-tower"]["marker"],
            linewidth=2.2,
            label=TWO_TOWER_STYLES["mul-two-tower"]["label"],
        ),
    ]
    csv_handles = [
        Line2D([0], [0], color=CSV_METHOD_COLORS[method], linewidth=1.8, label=CSV_METHOD_LABELS[method])
        for method in CSV_METHODS
    ]

    fig.suptitle("Position-Bias Estimates on MSLR-WEB30K by Sample Size", y=0.99)
    fig.legend(
        handles=method_handles + csv_handles,
        labels=[handle.get_label() for handle in method_handles + csv_handles],
        loc="lower center",
        ncol=4,
        bbox_to_anchor=(0.5, -0.06),
    )
    fig.tight_layout(rect=(0, 0.08, 1, 0.96))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output.resolve(), bbox_inches="tight")
    THESIS_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.output.resolve(), THESIS_IMAGES_DIR / args.output.name)
    plt.close(fig)
    print(f"Saved {args.output.resolve()}")
    print(f"Copied {(THESIS_IMAGES_DIR / args.output.name).resolve()}")


if __name__ == "__main__":
    main()
