import json
import os
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
RESULTS_DIR = REPO_ROOT / "results" / "real_targets"
PLOTS_DIR = REPO_ROOT / "notebooks" / "thesis_plots"
THESIS_IMAGES_DIR = REPO_ROOT.parent / "Thesis" / "images"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
THESIS_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("MPLCONFIGDIR", str(PLOTS_DIR / ".matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from notebooks.analysis.plot_style import apply_thesis_plot_style

apply_thesis_plot_style(plt)

DATASET_ORDER = ["istella", "mslr30k", "yahoo"]
DATASET_LABELS = {
    "istella": "Istella-S",
    "mslr30k": "MSLR-WEB30K",
    "yahoo": "Yahoo!",
}
TAB10_COLORS = plt.cm.tab10.colors
DATASET_COLORS = {
    "istella": TAB10_COLORS[0],
    "mslr30k": TAB10_COLORS[1],
    "yahoo": TAB10_COLORS[2],
}
MAX_N_SESSIONS = 1_000_000
METRICS = [
    {
        "key": "single_observation_proportion",
        "ylabel": "P(obs = 1 | obs >= 1)",
        "filename": "Training_Histograms_Single_Observation.pdf",
    },
    {
        "key": "observed_at_least_once_proportion",
        "ylabel": "P(obs >= 1)",
        "filename": "Training_Histograms_Observed_At_Least_Once.pdf",
    },
    {
        "key": "avg_observations_per_doc",
        "ylabel": "Average observations per doc",
        "filename": "Training_Histograms_Avg_Observations_Per_Doc.pdf",
    },
]


def mean(values):
    return sum(values) / len(values) if values else float("nan")


def percentile(values, q):
    if not values:
        return float("nan")
    return float(np.quantile(values, q))


def std(values):
    if len(values) < 2:
        return 0.0
    mu = mean(values)
    return float(np.std(values, ddof=1))


def extract_rows():
    pattern = re.compile(
        r"data=(?P<dataset>[^,]+).*?ips\.n_sessions=(?P<n_sessions>\d+).*?random_state=(?P<random_state>\d+)"
    )
    rows = []
    seen_keys = set()

    for histogram_path in sorted(RESULTS_DIR.glob("*/display_histograms/display_histogram_train.json")):
        match = pattern.search(str(histogram_path))
        if match is None:
            continue

        key = (
            match.group("dataset"),
            int(match.group("n_sessions")),
            int(match.group("random_state")),
        )
        if key[1] > MAX_N_SESSIONS:
            continue
        if key in seen_keys:
            continue
        seen_keys.add(key)

        with histogram_path.open() as f:
            payload = json.load(f)

        histogram = {int(obs): int(count) for obs, count in payload["histogram"].items()}
        total_docs = int(payload["total_docs"])
        observed_docs = sum(count for obs, count in histogram.items() if obs >= 1)
        single_observation_docs = histogram.get(1, 0)
        weighted_observation_sum = sum(obs * count for obs, count in histogram.items() if obs >= 1)

        rows.append(
            {
                "dataset": key[0],
                "n_sessions": key[1],
                "random_state": key[2],
                "total_docs": total_docs,
                "observed_docs": observed_docs,
                "single_observation_docs": single_observation_docs,
                "weighted_observation_sum": weighted_observation_sum,
                "single_observation_proportion": single_observation_docs / observed_docs if observed_docs else float("nan"),
                "observed_at_least_once_proportion": observed_docs / total_docs if total_docs else float("nan"),
                "avg_observations_per_doc": weighted_observation_sum / total_docs if total_docs else float("nan"),
                "source_path": os.path.relpath(histogram_path, REPO_ROOT),
            }
        )

    if not rows:
        raise RuntimeError(f"No display histogram files found under {RESULTS_DIR}.")

    return sorted(rows, key=lambda row: (row["dataset"], row["n_sessions"], row["random_state"]))


def aggregate_rows(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row["n_sessions"])].append(row)

    summary_rows = []
    plot_data = {metric["key"]: {dataset: [] for dataset in DATASET_ORDER} for metric in METRICS}

    for dataset in DATASET_ORDER:
        dataset_sessions = sorted(n for d, n in grouped if d == dataset)
        for n_sessions in dataset_sessions:
            bucket = grouped[(dataset, n_sessions)]
            summary_row = {
                "dataset": dataset,
                "n_sessions": n_sessions,
                "n_random_states": len(bucket),
            }
            for metric in METRICS:
                values = [row[metric["key"]] for row in bucket]
                mu = mean(values)
                sigma = std(values)
                p05 = percentile(values, 0.05)
                p95 = percentile(values, 0.95)
                summary_row[f"{metric['key']}_mean"] = mu
                summary_row[f"{metric['key']}_std"] = sigma
                summary_row[f"{metric['key']}_p05"] = p05
                summary_row[f"{metric['key']}_p95"] = p95
                plot_data[metric["key"]][dataset].append((n_sessions, mu, p05, p95))
            summary_rows.append(summary_row)

    return summary_rows, plot_data


def make_legend_handles():
    handles = [
        Line2D([0], [0], color=DATASET_COLORS[dataset], marker="o", linewidth=2, label=DATASET_LABELS[dataset])
        for dataset in DATASET_ORDER
    ]
    handles.append(Line2D([0], [0], color="grey", linewidth=6, alpha=0.2, label="90% interval (p05-p95)"))
    return handles


def plot_metric(metric, plot_data):
    fig, axes = plt.subplots(1, len(DATASET_ORDER), figsize=(15, 4.8), sharex=True)
    axes = np.atleast_1d(axes)

    for ax, dataset in zip(axes, DATASET_ORDER):
        series = plot_data[metric["key"]][dataset]
        x = np.array([n_sessions for n_sessions, _, _, _ in series], dtype=float)
        y = np.array([mu for _, mu, _, _ in series], dtype=float)
        low = np.array([p05 for _, _, p05, _ in series], dtype=float)
        high = np.array([p95 for _, _, _, p95 in series], dtype=float)
        color = DATASET_COLORS[dataset]

        ax.plot(x, y, marker="o", color=color, linewidth=2)
        ax.fill_between(x, low, high, color=color, alpha=0.2)
        ax.set_xscale("log")
        ax.set_xlim(500, MAX_N_SESSIONS)
        ax.set_title(DATASET_LABELS[dataset])
        ax.set_xlabel("Number of Sessions")
        ax.grid(True, which="both", linestyle="--", alpha=0.4)

    axes[0].set_ylabel(metric["ylabel"])

    fig.legend(
        handles=make_legend_handles(),
        loc="lower center",
        ncol=4,
        bbox_to_anchor=(0.5, -0.02),
        frameon=False,
    )
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.22)

    output_path = PLOTS_DIR / metric["filename"]
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    shutil.copy2(output_path, THESIS_IMAGES_DIR / metric["filename"])


def main():
    print("Generating training histograms from results/real_targets...")

    raw_rows = extract_rows()
    _, plot_data = aggregate_rows(raw_rows)

    for metric in METRICS:
        plot_metric(metric, plot_data)

    print(f"Copied PDFs to {THESIS_IMAGES_DIR}")
    print("Done. Training histograms generated.")


if __name__ == "__main__":
    main()
