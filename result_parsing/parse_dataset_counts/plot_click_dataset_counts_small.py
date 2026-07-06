import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT))

from result_parsing.analysis.plot_style import apply_plot_style


DATASET_COUNTS_DIR = SCRIPT_DIR / "dataset_counts"
OUTPUT_PATH = REPO_ROOT / "result_parsing" / "result_plots" / "Click_dataset_counts_histogram_small.pdf"

FILES = (
    "baidu_ultr_hist.json",
    "yandex_hist.json",
)
NICE_NAMES = (
    "Baidu ULTR",
    "Yandex",
)


def load_hist(path: Path) -> Counter:
    with path.open("r") as f:
        data = json.load(f)
    return Counter({int(k): int(v) for k, v in data.items()})

def increase_font_sizes(delta: int = 4) -> None:
    keys = [
        "font.size",
        "axes.titlesize",
        "axes.labelsize",
        "xtick.labelsize",
        "ytick.labelsize",
        "legend.fontsize",
        "figure.titlesize",
    ]

    for key in keys:
        current = plt.rcParams[key]
        if isinstance(current, str):
            current = plt.rcParams["font.size"]
        plt.rcParams[key] = current + delta

def expand_hist(counter: Counter, cutoff: int = 50) -> np.ndarray:
    arr = []
    for k, v in counter.items():
        if k <= cutoff:
            arr.extend([k] * v)
    return np.array(arr)


def main() -> None:
    apply_plot_style(plt)
    increase_font_sizes(3)
    plt.rcParams["axes.labelsize"] -= 1


    fig, axes = plt.subplots(1, 2, figsize=(8, 2.2), sharey=True)

    for i, (ax, filename, name) in enumerate(zip(axes, FILES, NICE_NAMES)):
        hist = load_hist(DATASET_COUNTS_DIR / filename)
        counts = expand_hist(hist, cutoff=20)
        weights = np.ones_like(counts) / len(counts) * 100

        ax.hist(
            counts,
            bins=20,
            range=(1, 20),
            weights=weights,
            color=f"C{i}",
            label=name,
            zorder=3,
        )

        ax.set_xlabel("Total observations in dataset")
        ax.set_xlim(1, 20)
        ax.set_ylim(0, 105)
        ax.set_yticks(range(0, 101, 25))
        ax.grid(axis="y", linestyle="--", alpha=0.9, zorder=0)
        ax.legend(loc="upper right", frameon=False)

        if ax == axes[0]:
            ax.set_ylabel(r"Query-doc. pairs \%", y=0.42)

    fig.tight_layout(w_pad=0.0, pad=0.1)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(f"Saved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
