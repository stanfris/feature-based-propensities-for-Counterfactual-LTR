"""Plot true- and frequency-propensity IPS reward-estimate variance."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot reward-estimate variance from the simple IPS experiment."
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=REPO_ROOT / "results" / "simple_reward_variance_experiment.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT
        / "notebooks"
        / "thesis_plots"
        / "simple_reward_variance_experiment_variance.pdf",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = pd.read_csv(args.input_csv).sort_values("n_observations")

    required_columns = {
        "n_observations",
        "var_true_ips",
        "var_frequency_ips",
    }
    missing_columns = required_columns.difference(data.columns)
    if missing_columns:
        raise ValueError(
            f"Missing required CSV columns: {sorted(missing_columns)}"
        )

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 12,
            "axes.labelsize": 13,
            "axes.titlesize": 14,
            "legend.fontsize": 11,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
        }
    )
    fig, ax = plt.subplots(figsize=(7.2, 3))
    ax.plot(
        data["n_observations"],
        data["var_true_ips"],
        color="#3366A3",
        marker="o",
        linewidth=2,
        markersize=5,
        label="Oracle propensity",
    )
    ax.plot(
        data["n_observations"],
        data["var_frequency_ips"],
        color="tab:orange",
        marker="s",
        linewidth=2,
        markersize=5,
        label="Frequency-based Propensity",
    )

    ax.set_xlabel("Number of document observations")
    ax.set_ylabel("Average variance")
    ax.set_xticks(data["n_observations"])
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight")
    plt.close(fig)

    print(f"saved_plot={args.output}")


if __name__ == "__main__":
    main()
