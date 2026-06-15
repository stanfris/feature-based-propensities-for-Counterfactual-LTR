import argparse
import re
import statistics
from collections import Counter
from pathlib import Path

QID_PATTERN = re.compile(r"qid:(\d+)")
SPLITS = ("train", "vali", "test")
DATASETS = {
    "mslr30k": {
        "name": "MSLR-WEB30K",
        "default_root": Path("/Users/stanf/Documents/Uni/MSc Thesis/ltr_datasets/dataset/MSLR-WEB30K"),
        "fold_dir": lambda root, fold: root / f"Fold{fold}",
        "split_file": lambda split, fold: f"{split}.txt",
        "subtitle": "Train, validation, and test distributions from the raw fold split files",
    },
    "yahoo-c14": {
        "name": "Yahoo C14",
        "default_root": Path("/Users/stanf/Documents/Uni/MSc Thesis/ltr_datasets/dataset/ltrc_yahoo"),
        "fold_dir": lambda root, fold: root,
        "split_file": lambda split, fold: {
            "train": f"set{fold}.train.txt",
            "vali": f"set{fold}.valid.txt",
            "test": f"set{fold}.test.txt",
        }[split],
        "subtitle": "Train, validation, and test distributions from the raw set split files",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot the documents-per-query distribution for supported SVMLight ranking datasets."
    )
    parser.add_argument(
        "--dataset",
        choices=sorted(DATASETS.keys()),
        default="mslr30k",
        help="Dataset to plot.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="Optional override for the dataset root directory.",
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=1,
        help="Fold/set identifier to use.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("notebooks/thesis_plots"),
        help="Directory to save the plot files in.",
    )
    return parser.parse_args()


def count_documents_per_query(path: Path) -> list[int]:
    counts = Counter()
    with path.open("r") as handle:
        for line_number, line in enumerate(handle, start=1):
            match = QID_PATTERN.search(line)
            if match is None:
                raise ValueError(f"Missing qid on line {line_number} of {path}")
            counts[match.group(1)] += 1
    return list(counts.values())


def percentile(sorted_counts: list[int], q: float) -> float:
    if not sorted_counts:
        raise ValueError("Cannot compute percentile of an empty sequence.")
    if len(sorted_counts) == 1:
        return float(sorted_counts[0])
    position = (len(sorted_counts) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(sorted_counts) - 1)
    weight = position - lower
    return sorted_counts[lower] * (1 - weight) + sorted_counts[upper] * weight


def summarize(counts: list[int]) -> dict[str, float]:
    sorted_counts = sorted(counts)
    return {
        "queries": len(counts),
        "documents": sum(counts),
        "min_docs": min(counts),
        "median_docs": float(statistics.median(sorted_counts)),
        "mean_docs": float(statistics.fmean(counts)),
        "p95_docs": float(percentile(sorted_counts, 0.95)),
        "max_docs": max(counts),
    }


def build_bin_edges(max_docs: int, step: int = 10) -> list[float]:
    edges = [0.5]
    current = 0.5 + step
    while current < max_docs + 0.5:
        edges.append(current)
        current += step
    edges.append(max_docs + 0.5)
    return edges


def histogram(counts: list[int], bin_edges: list[float]) -> list[int]:
    hist = [0 for _ in range(len(bin_edges) - 1)]
    step = bin_edges[1] - bin_edges[0]
    for value in counts:
        idx = int((value - bin_edges[0]) // step)
        idx = max(0, min(idx, len(hist) - 1))
        hist[idx] += 1
    return hist


def svg_histogram_bars(
    hist: list[int],
    x0: float,
    y0: float,
    width: float,
    height: float,
    max_y: int,
    color: str,
    alpha: float = 1.0,
) -> list[str]:
    bars = []
    n_bins = len(hist)
    bar_width = width / max(n_bins, 1)
    for idx, value in enumerate(hist):
        if value <= 0:
            continue
        bar_height = height * (value / max_y)
        x = x0 + idx * bar_width
        y = y0 + height - bar_height
        bars.append(
            f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_width:.2f}" height="{bar_height:.2f}" '
            f'fill="{color}" fill-opacity="{alpha:.2f}" />'
        )
    return bars


def svg_vertical_line(
    value: float,
    min_x: float,
    max_x: float,
    x0: float,
    y0: float,
    width: float,
    height: float,
    color: str,
    dasharray: str,
) -> str:
    x = x0 + width * ((value - min_x) / (max_x - min_x))
    return (
        f'<line x1="{x:.2f}" y1="{y0:.2f}" x2="{x:.2f}" y2="{y0 + height:.2f}" '
        f'stroke="{color}" stroke-width="2" stroke-dasharray="{dasharray}" />'
    )


def plot_distribution(
    split_counts: dict[str, list[int]],
    output_base: Path,
    fold: int,
    dataset_name: str,
    dataset_subtitle: str,
) -> None:
    all_counts = [count for counts in split_counts.values() for count in counts]
    max_docs = max(all_counts)
    bins = build_bin_edges(max_docs=max_docs, step=10)

    colors = {
        "train": "#1b9e77",
        "vali": "#d95f02",
        "test": "#7570b3",
    }
    split_hists = {split: histogram(counts, bins) for split, counts in split_counts.items()}
    combined_hist = histogram(all_counts, bins)

    width = 1200
    height = 950
    margin_left = 90
    margin_right = 40
    panel_width = width - margin_left - margin_right
    panel_height = 300
    top_y = 110
    bottom_y = 540
    x_min = float(bins[0])
    x_max = float(bins[-1])
    max_y_top = max(max(hist) for hist in split_hists.values())
    max_y_bottom = max(combined_hist)
    all_stats = summarize(all_counts)

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white" />',
        '<style>',
        'text { font-family: Georgia, "Times New Roman", serif; fill: #222; }',
        '.title { font-size: 26px; font-weight: 600; }',
        '.subtitle { font-size: 16px; }',
        '.axis { font-size: 14px; }',
        '.legend { font-size: 14px; }',
        '</style>',
        f'<text x="{width / 2:.0f}" y="42" text-anchor="middle" class="title">{dataset_name} fold {fold} documents per query</text>',
        f'<text x="{width / 2:.0f}" y="70" text-anchor="middle" class="subtitle">{dataset_subtitle}</text>',
    ]

    for panel_y, panel_max_y, panel_title in (
        (top_y, max_y_top, "By split"),
        (bottom_y, max_y_bottom, "Combined"),
    ):
        svg.append(f'<text x="{margin_left}" y="{panel_y - 22}" class="subtitle">{panel_title}</text>')
        svg.append(
            f'<rect x="{margin_left}" y="{panel_y}" width="{panel_width}" height="{panel_height}" fill="none" stroke="#333" stroke-width="1" />'
        )
        for frac in (0.25, 0.5, 0.75):
            y = panel_y + panel_height * (1 - frac)
            svg.append(
                f'<line x1="{margin_left}" y1="{y:.2f}" x2="{margin_left + panel_width}" y2="{y:.2f}" stroke="#d9d9d9" stroke-width="1" />'
            )
            svg.append(
                f'<text x="{margin_left - 10}" y="{y + 5:.2f}" text-anchor="end" class="axis">{int(panel_max_y * frac):,}</text>'
            )
        svg.append(
            f'<text x="{margin_left - 10}" y="{panel_y + panel_height + 5}" text-anchor="end" class="axis">0</text>'
        )
        svg.append(
            f'<text x="{margin_left + panel_width / 2:.2f}" y="{panel_y + panel_height + 45}" text-anchor="middle" class="axis">Documents per query</text>'
        )
        svg.append(
            f'<text x="28" y="{panel_y + panel_height / 2:.2f}" text-anchor="middle" transform="rotate(-90 28 {panel_y + panel_height / 2:.2f})" class="axis">Number of queries</text>'
        )

    for split in SPLITS:
        svg.extend(
            svg_histogram_bars(
                hist=split_hists[split],
                x0=margin_left,
                y0=top_y,
                width=panel_width,
                height=panel_height,
                max_y=max_y_top,
                color=colors[split],
                alpha=0.45,
            )
        )

    svg.extend(
        svg_histogram_bars(
            hist=combined_hist,
            x0=margin_left,
            y0=bottom_y,
            width=panel_width,
            height=panel_height,
            max_y=max_y_bottom,
            color="#4c78a8",
            alpha=0.9,
        )
    )
    svg.append(
        svg_vertical_line(
            value=all_stats["mean_docs"],
            min_x=x_min,
            max_x=x_max,
            x0=margin_left,
            y0=bottom_y,
            width=panel_width,
            height=panel_height,
            color="#e45756",
            dasharray="8,6",
        )
    )
    svg.append(
        svg_vertical_line(
            value=all_stats["median_docs"],
            min_x=x_min,
            max_x=x_max,
            x0=margin_left,
            y0=bottom_y,
            width=panel_width,
            height=panel_height,
            color="#72b7b2",
            dasharray="2,6",
        )
    )

    tick_values = [1, 100, 200, 400, 600, 800, 1000, max_docs]
    tick_values = sorted({tick for tick in tick_values if 1 <= tick <= max_docs})
    for panel_y in (top_y, bottom_y):
        for tick in tick_values:
            x = margin_left + panel_width * ((tick - x_min) / (x_max - x_min))
            svg.append(
                f'<line x1="{x:.2f}" y1="{panel_y + panel_height}" x2="{x:.2f}" y2="{panel_y + panel_height + 6}" stroke="#333" stroke-width="1" />'
            )
            svg.append(
                f'<text x="{x:.2f}" y="{panel_y + panel_height + 24}" text-anchor="middle" class="axis">{tick}</text>'
            )

    legend_y = 470
    legend_x = margin_left
    legend_items = [
        ("train", colors["train"]),
        ("vali", colors["vali"]),
        ("test", colors["test"]),
        (f"mean = {all_stats['mean_docs']:.1f}", "#e45756"),
        (f"median = {all_stats['median_docs']:.0f}", "#72b7b2"),
    ]
    x_cursor = legend_x
    for label, color in legend_items:
        svg.append(f'<rect x="{x_cursor}" y="{legend_y}" width="18" height="18" fill="{color}" />')
        svg.append(f'<text x="{x_cursor + 26}" y="{legend_y + 14}" class="legend">{label}</text>')
        x_cursor += 130 if label in {"train", "vali", "test"} else 180

    svg.append("</svg>")
    output_base.with_suffix(".svg").write_text("\n".join(svg))


def main() -> None:
    args = parse_args()
    dataset_cfg = DATASETS[args.dataset]
    dataset_root = args.dataset_root or dataset_cfg["default_root"]
    fold_dir = dataset_cfg["fold_dir"](dataset_root, args.fold)
    split_counts = {
        split: count_documents_per_query(
            fold_dir / dataset_cfg["split_file"](split, args.fold)
        )
        for split in SPLITS
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset_slug = args.dataset.replace("-", "_")
    output_base = args.output_dir / f"{dataset_slug}_query_doc_distribution_fold{args.fold}"
    plot_distribution(
        split_counts,
        output_base=output_base,
        fold=args.fold,
        dataset_name=dataset_cfg["name"],
        dataset_subtitle=dataset_cfg["subtitle"],
    )

    print(f"Saved plot to {output_base.with_suffix('.svg')}")
    for split in SPLITS:
        stats = summarize(split_counts[split])
        print(
            f"{split}: queries={stats['queries']:,}, docs={stats['documents']:,}, "
            f"min={stats['min_docs']}, median={stats['median_docs']:.1f}, "
            f"mean={stats['mean_docs']:.2f}, p95={stats['p95_docs']:.1f}, max={stats['max_docs']}"
        )
    overall = summarize([count for counts in split_counts.values() for count in counts])
    print(
        f"all: queries={overall['queries']:,}, docs={overall['documents']:,}, "
        f"min={overall['min_docs']}, median={overall['median_docs']:.1f}, "
        f"mean={overall['mean_docs']:.2f}, p95={overall['p95_docs']:.1f}, max={overall['max_docs']}"
    )


if __name__ == "__main__":
    main()
