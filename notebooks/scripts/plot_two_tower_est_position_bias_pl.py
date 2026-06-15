import argparse
import os
from pathlib import Path
import re
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
PLOTS_DIR = REPO_ROOT / "notebooks" / "thesis_plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("MPLCONFIGDIR", str(PLOTS_DIR / ".matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(PLOTS_DIR / ".cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from notebooks.analysis.plot_style import apply_thesis_plot_style

apply_thesis_plot_style(plt)

DEFAULT_RESULTS_ROOT = REPO_ROOT / "results" / "two_towers_pl"
DEFAULT_TEMPERATURES = (0.5, 1.0)
DATASET_ORDER = ("istella", "mslr30k", "yahoo")
DATASET_LABELS = {
    "istella": "Istella-S",
    "mslr30k": "MSLR-WEB30K",
    "yahoo": "Yahoo!",
}


def compute_true_bias(positions: np.ndarray, strength: float = 1.0) -> np.ndarray:
    values = -strength * np.log(positions + 1)
    return values - values[0]


def temperature_slug(value: float) -> str:
    return str(value).replace(".", "_")


def default_output_path(model_kind: str, temperature: float) -> Path:
    return PLOTS_DIR / f"{model_kind}_two_tower_pl_position_bias_temp_{temperature_slug(temperature)}.pdf"


def load_bias_rows(
    results_root: Path,
    *,
    csv_name: str,
    ips_model: str,
    temperature: float,
) -> pd.DataFrame:
    pattern = re.compile(
        r"data=(?P<dataset>[^,]+).*?"
        r"ips\.model=(?P<ips_model>[^,]+).*?"
        r"ips\.n_sessions=(?P<n_sessions>\d+).*?"
        r"policy_temperature=(?P<policy_temperature>[^,]+).*?"
        r"random_state=(?P<random_state>\d+)"
    )
    rows: list[dict[str, float | int | str]] = []

    for csv_path in sorted(results_root.glob(f"*/{csv_name}")):
        match = pattern.search(str(csv_path.parent))
        if match is None or match.group("ips_model") != ips_model:
            continue

        file_temperature = float(match.group("policy_temperature"))
        if not np.isclose(file_temperature, temperature, atol=1e-8):
            continue

        df = pd.read_csv(csv_path)
        df["dataset"] = match.group("dataset")
        df["n_sessions"] = int(match.group("n_sessions"))
        df["random_state"] = int(match.group("random_state"))
        df["policy_temperature"] = file_temperature
        rows.extend(df.to_dict(orient="records"))

    if not rows:
        raise RuntimeError(
            f"No '{csv_name}' files for ips.model={ips_model} and policy_temperature={temperature} "
            f"found under {results_root}."
        )

    return pd.DataFrame(rows)


def aggregate_bias_rows(df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        df.groupby(["dataset", "n_sessions", "position"], as_index=False)
        .agg(
            mean_bias=("relative_position_bias", "mean"),
            std_bias=("relative_position_bias", "std"),
            n_random_states=("random_state", "nunique"),
        )
        .sort_values(["dataset", "n_sessions", "position"])
        .reset_index(drop=True)
    )
    grouped["std_bias"] = grouped["std_bias"].fillna(0.0)
    return grouped


def plot_position_bias(summary: pd.DataFrame, output_path: Path, *, plot_title: str) -> Path:
    datasets = [dataset for dataset in DATASET_ORDER if dataset in set(summary["dataset"])]
    if not datasets:
        raise RuntimeError(f"No supported datasets found in {plot_title.lower()} summary.")

    session_counts = sorted(summary["n_sessions"].unique())
    colors = plt.cm.tab10.colors

    fig, axes = plt.subplots(
        1,
        len(datasets),
        figsize=(6.5 * len(datasets), 4.8),
        sharex=True,
        sharey=False,
    )
    axes = np.atleast_1d(axes)

    legend_handles: list[object] = [
        Line2D([0], [0], color="black", linestyle="--", linewidth=2, label="True bias")
    ]

    for color_idx, n_sessions in enumerate(session_counts):
        color = colors[color_idx % len(colors)]
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color=color,
                marker="o",
                linestyle="-",
                label=f"{n_sessions:,} sessions",
            )
        )

    legend_handles.append(
        Patch(facecolor="grey", edgecolor="none", alpha=0.2, label="±1 std. dev.")
    )

    for ax, dataset in zip(axes, datasets):
        dataset_df = summary[summary["dataset"] == dataset]
        positions = np.sort(dataset_df["position"].unique())
        true_bias = compute_true_bias(positions.astype(float))

        ax.plot(positions, true_bias, color="black", linestyle="--", linewidth=2, zorder=4)

        for color_idx, n_sessions in enumerate(session_counts):
            curve = dataset_df[dataset_df["n_sessions"] == n_sessions].sort_values("position")
            if curve.empty:
                continue

            x = curve["position"].to_numpy()
            y = curve["mean_bias"].to_numpy()
            e = curve["std_bias"].to_numpy()
            color = colors[color_idx % len(colors)]

            ax.plot(x, y, marker="o", color=color, zorder=2)
            ax.fill_between(x, y - e, y + e, color=color, alpha=0.2, zorder=1)

        ax.set_title(DATASET_LABELS.get(dataset, dataset))
        ax.set_xlabel("Position")
        ax.set_ylabel("Relative position bias")
        ax.grid(True, which="both", linestyle="--", alpha=0.5)

    fig.suptitle(plot_title)
    fig.legend(
        handles=legend_handles,
        labels=[handle.get_label() for handle in legend_handles],
        loc="lower center",
        ncol=min(len(legend_handles), 6),
        bbox_to_anchor=(0.5, -0.05),
    )
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot true vs estimated additive and multiplicative two_towers_pl position bias."
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=DEFAULT_RESULTS_ROOT,
        help=f"Directory containing two_towers_pl run folders. Default: {DEFAULT_RESULTS_ROOT}",
    )
    parser.add_argument(
        "--temperatures",
        type=float,
        nargs="+",
        default=DEFAULT_TEMPERATURES,
        help="Policy temperatures to visualize separately. Default: 0.5 1.0",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    results_root = args.results_root.resolve()

    for temperature in args.temperatures:
        additive_df = load_bias_rows(
            results_root,
            csv_name="additive_position_bias.csv",
            ips_model="add-two-tower",
            temperature=temperature,
        )
        additive_summary = aggregate_bias_rows(additive_df)
        additive_output_path = plot_position_bias(
            additive_summary,
            default_output_path("additive", temperature).resolve(),
            plot_title=f"Additive Two-Tower PL Position Bias (T={temperature})",
        )

        multiplicative_df = load_bias_rows(
            results_root,
            csv_name="multiplicative_position_bias.csv",
            ips_model="mul-two-tower",
            temperature=temperature,
        )
        multiplicative_summary = aggregate_bias_rows(multiplicative_df)
        multiplicative_output_path = plot_position_bias(
            multiplicative_summary,
            default_output_path("multiplicative", temperature).resolve(),
            plot_title=f"Multiplicative Two-Tower PL Position Bias (T={temperature})",
        )

        print(
            f"Saved additive two_towers_pl position-bias plot for T={temperature} to "
            f"{additive_output_path}"
        )
        print(
            f"Saved multiplicative two_towers_pl position-bias plot for T={temperature} to "
            f"{multiplicative_output_path}"
        )


if __name__ == "__main__":
    main()
