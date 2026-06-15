import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

from table_significance import format_summary_cell, get_sig_symbol, get_valid_pairs, paired_test_pvalue, sample_std


current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = Path(os.path.abspath(os.path.join(current_dir, "..", "..")))


METRICS = ("NDCG",)
DATASET_LABELS = {
    "mslr30k": "MSLR-WEB30K",
    "yahoo": "Yahoo!",
    "istella": "Istella-S",
}
VARIANT_NAMES = {
    "baseline": "True Bias$^{\\mathrm{oracle}}$",
    "bias-add": "Bias Only",
    "reg-add": "Regression Only",
    "full-add": "Full Two-Tower",
    "full-add-shallow": "Full Two-Tower (Shallow)",
    "bias-mul": "Bias Only",
    "reg-mul": "Regression Only",
    "full-mul": "Full Two-Tower",
    "full-mul-shallow": "Full Two-Tower (Shallow)",
    "pb-adjacent_chain": "Adjacent Chain",
    "pb-ctr": "CTR",
    "pb-global_all_pairs": "Global All Pairs",
    "pb-pivot_one": "Pivot One",
}
NOTEBOOK_TABLES_DIR = project_root / "notebooks" / "thesis_tables"
THESIS_TABLES_DIR = project_root.parent / "Thesis" / "tables"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a frequency-based DR LaTeX table grouped by additive and multiplicative two-tower variants."
    )
    parser.add_argument(
        "--layout",
        choices=("per-dataset", "cross-dataset"),
        default="per-dataset",
        help="Choose between the original per-dataset layout and a single cross-dataset table.",
    )
    parser.add_argument("--dataset", default="mslr30k")
    parser.add_argument("--dataset-label", default="MSLR-WEB30K")
    parser.add_argument(
        "--datasets",
        default="mslr30k,yahoo,istella",
        help="Comma-separated datasets to include in the cross-dataset layout.",
    )
    parser.add_argument(
        "--n-sessions",
        default="10000,100000",
        help="Comma-separated session counts to include.",
    )
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--random-states", default="40,41,42")
    parser.add_argument(
        "--output-name",
        default="tabel_dr_two_tower_holistic_comparison.txt",
        help="Filename used for both output table locations.",
    )
    parser.add_argument(
        "--label",
        default="tab:dr_two_tower_holistic_comparison",
        help="LaTeX label used in the generated table.",
    )
    return parser.parse_args()


def parse_folder_name(folder_name):
    parsed = {}
    for part in folder_name.split(","):
        if "=" in part:
            key, value = part.lstrip("+").split("=", 1)
            parsed[key] = value
    return parsed


def parse_random_states(spec):
    return {int(part.strip()) for part in spec.split(",") if part.strip()}


def parse_int_list(spec):
    return tuple(int(part.strip()) for part in spec.split(",") if part.strip())


def parse_dataset_list(spec):
    return tuple(part.strip() for part in spec.split(",") if part.strip())


def normalize_two_tower_name(value):
    aliases = {
        "add_two_tower": "add-two-tower",
        "mul_two_tower": "mul-two-tower",
    }
    return aliases.get(value, value)


def normalize_position_bias_model(method):
    if method in {"adjacent_chain", "ctr", "global_all_pairs", "pivot_one"}:
        return f"pb-{method}"
    return method


def safe_load_json(path):
    try:
        with path.open("r") as handle:
            return json.load(handle)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def classify_variant(experiment, parsed):
    position_bias = normalize_two_tower_name(parsed.get("ips.position_bias.source"))
    regression_method = normalize_two_tower_name(parsed.get("ips.regression.method"))

    if experiment == "real_targets":
        if position_bias is None and regression_method in {None, "regression"}:
            return "baseline"
        return None

    if experiment == "two_tower_bias":
        if position_bias == "add-two-tower" and regression_method is None:
            return "bias-add"
        if position_bias == "mul-two-tower" and regression_method is None:
            return "bias-mul"
        return None

    if experiment == "two_tower_for_DMDR":
        if position_bias is None and regression_method == "add-two-tower":
            return "reg-add"
        if position_bias is None and regression_method == "mul-two-tower":
            return "reg-mul"
        return None

    if experiment == "DR_with_two_towers":
        if position_bias == "add-two-tower" and regression_method == "add-two-tower":
            return "full-add"
        if position_bias == "mul-two-tower" and regression_method == "mul-two-tower":
            return "full-mul"
        return None

    if experiment in {"shallow_DR_with_two_towers", "DR_with_two_towers_shallow"}:
        if position_bias == "add-two-tower" and regression_method == "add-two-tower":
            return "full-add-shallow"
        if position_bias == "mul-two-tower" and regression_method == "mul-two-tower":
            return "full-mul-shallow"
        return None

    if experiment == "position_bias_models":
        if position_bias != "csv" or regression_method not in {None, "regression"}:
            return None
        method = normalize_position_bias_model(parsed.get("ips.position_bias.csv.method"))
        if method in {
            "pb-adjacent_chain",
            "pb-ctr",
            "pb-global_all_pairs",
            "pb-pivot_one",
        }:
            return method
        return None

    return None


def load_records(datasets, n_sessions_list, temperature, random_states):
    records = []
    wanted_n_sessions = set(n_sessions_list)
    wanted_datasets = set(datasets)
    experiments = (
        "real_targets",
        "two_tower_bias",
        "two_tower_for_DMDR",
        "DR_with_two_towers",
        "shallow_DR_with_two_towers",
        "DR_with_two_towers_shallow",
        "position_bias_models",
    )

    for experiment in experiments:
        base_dir = project_root / "results" / experiment
        if not base_dir.exists():
            continue
        for path in base_dir.rglob("ips_results.json"):
            parsed = parse_folder_name(path.parent.name)
            if parsed.get("data") not in wanted_datasets:
                continue
            if parsed.get("ips.model") != "dr":
                continue
            if parsed.get("propensity_model") != "frequency-based":
                continue

            raw_n_sessions = parsed.get("ips.n_sessions")
            raw_temperature = parsed.get("policy_temperature")
            raw_random_state = parsed.get("random_state")

            if raw_n_sessions is None or int(raw_n_sessions) not in wanted_n_sessions:
                continue
            if raw_temperature is None or float(raw_temperature) != temperature:
                continue
            if raw_random_state is None or int(raw_random_state) not in random_states:
                continue

            variant = classify_variant(experiment, parsed)
            if variant is None:
                continue

            payload = safe_load_json(path)
            if not payload:
                continue
            metrics = payload.get("results", {}).get("metrics", {})
            if not all(metric in metrics for metric in METRICS):
                continue

            records.append(
                {
                    "dataset": parsed["data"],
                    "variant": variant,
                    "n_sessions": int(raw_n_sessions),
                    "temperature": float(raw_temperature),
                    "random_state": int(raw_random_state),
                    "metric_values": {metric: float(metrics[metric]) for metric in METRICS},
                }
            )

    if not records:
        raise RuntimeError("No matching records found for the requested DR comparison.")
    return records


def build_summary(records):
    grouped = defaultdict(lambda: defaultdict(list))
    for record in records:
        key = (record["dataset"], record["variant"], record["n_sessions"])
        for metric, value in record["metric_values"].items():
            grouped[key][metric].append(value)
    return grouped


def build_table_cells(summary, records):
    paired = defaultdict(lambda: defaultdict(dict))
    for record in records:
        pair_id = (record["n_sessions"], record["temperature"], record["random_state"])
        for metric, value in record["metric_values"].items():
            paired[(record["dataset"], record["n_sessions"], metric)][pair_id][record["variant"]] = value

    cell_map = {}
    diagnostics = []
    best_variant_map = {}

    for (dataset, variant, n_sessions), metrics in summary.items():
        for metric in metrics:
            context_key = (dataset, n_sessions, metric)
            if context_key in best_variant_map:
                continue

            candidate_variants = [
                candidate_variant
                for candidate_variant in VARIANT_NAMES
                if candidate_variant != "baseline"
                and (dataset, candidate_variant, n_sessions) in summary
                and metric in summary[(dataset, candidate_variant, n_sessions)]
            ]
            if not candidate_variants:
                best_variant_map[context_key] = None
                continue

            best_variant_map[context_key] = max(
                candidate_variants,
                key=lambda candidate_variant: (
                    sum(summary[(dataset, candidate_variant, n_sessions)][metric])
                    / len(summary[(dataset, candidate_variant, n_sessions)][metric]),
                    -tuple(VARIANT_NAMES).index(candidate_variant),
                ),
            )

    for (dataset, variant, n_sessions), metrics in summary.items():
        for metric, values in metrics.items():
            mean_value = sum(values) / len(values)
            std_value = sample_std(values)
            context_key = (dataset, n_sessions, metric)
            best_variant = best_variant_map.get(context_key)

            if best_variant is None:
                cell_map[(dataset, variant, n_sessions, metric)] = format_summary_cell(
                    mean_value,
                    std_value,
                    "",
                    bold=False,
                )
                continue

            if variant == best_variant:
                cell_map[(dataset, variant, n_sessions, metric)] = format_summary_cell(
                    mean_value,
                    std_value,
                    "",
                    bold=True,
                )
                continue

            reference_values = summary.get((dataset, best_variant, n_sessions), {}).get(metric, [])
            matched_values = []
            matched_reference_values = []
            for pair in paired.get((dataset, n_sessions, metric), {}).values():
                if variant in pair and best_variant in pair:
                    matched_values.append(pair[variant])
                    matched_reference_values.append(pair[best_variant])

            valid_pairs = get_valid_pairs(matched_values, matched_reference_values)
            p_value = paired_test_pvalue(
                [x for x, _ in valid_pairs],
                [y for _, y in valid_pairs],
            )
            reference_mean = sum(reference_values) / len(reference_values) if reference_values else mean_value
            symbol = get_sig_symbol(mean_value, reference_mean, p_value)
            cell = format_summary_cell(mean_value, std_value, symbol, bold=False)
            matched_count = len(valid_pairs)
            if reference_values and matched_count < 2:
                diagnostics.append(
                    {
                        "dataset": dataset,
                        "n_sessions": n_sessions,
                        "variant": variant,
                        "best_variant": best_variant,
                        "metric": metric,
                        "matched_pairs": matched_count,
                    }
                )
            cell_map[(dataset, variant, n_sessions, metric)] = cell

    return cell_map, diagnostics


def format_n_sessions(n_sessions):
    exponent_map = {
        10_000: 4,
        100_000: 5,
        1_000_000: 6,
    }
    if n_sessions in exponent_map:
        return rf"\sampleNheader{{{exponent_map[n_sessions]}}}"
    return f"{n_sessions:,}"


def render_table(cell_map, summary, dataset, dataset_label, n_sessions_list, temperature, random_states, label):
    available_variants = {variant for summary_dataset, variant, _ in summary.keys() if summary_dataset == dataset}
    groups = (
        (
            "Position-Bias Estimation",
            tuple(
                variant
                for variant in ("baseline", "pb-adjacent_chain", "pb-ctr", "pb-global_all_pairs", "pb-pivot_one")
                if variant in available_variants
            ),
        ),
        (
            "Additive Two-Tower",
            tuple(
                variant
                for variant in ("bias-add", "reg-add", "full-add", "full-add-shallow")
                if variant in available_variants
            ),
        ),
        (
            "Multiplicative Two-Tower",
            tuple(
                variant
                for variant in ("bias-mul", "reg-mul", "full-mul", "full-mul-shallow")
                if variant in available_variants
            ),
        ),
    )
    groups = tuple((group_name, row_order) for group_name, row_order in groups if row_order)

    col_spec = "ll" + "c" * len(n_sessions_list)
    header_groups = " ".join(f"& {format_n_sessions(n_sessions)}" for n_sessions in n_sessions_list)

    lines = []
    lines.append("\\begin{table}[h!]")
    lines.append(
        "  \\caption{Frequency-based DR comparison on "
        f"{dataset_label}. The table includes the oracle-bias baseline in the position-bias estimation "
        "group and groups additive, multiplicative, and CSV position-bias variants separately. "
        "Cells show mean (SD) NDCG. Bold marks the best non-oracle model within each comparison block. "
        "Significance markers compare each other model "
        "with this best model using a paired two-sided Student t-test: $p < 0.01$ "
        "($\\,^\\blacktriangle \\text{ and } ^\\blacktriangledown$), $p < 0.05$ "
        "($\\,^\\triangle \\text{ and } ^\\triangledown$).}"
    )
    lines.append("  \\centering")
    lines.append("  \\setlength{\\tabcolsep}{4pt}")
    lines.append("  \\resizebox{\\textwidth}{!}{")
    lines.append(f"    \\begin{{tabular}}{{{col_spec}}}")
    lines.append("      \\toprule")
    lines.append(f"      & {header_groups} \\\\")
    lines.append(
        "      \\textbf{Comparison Group} & \\textbf{Model} & "
        + " & ".join(["NDCG"] * len(n_sessions_list))
        + " \\\\"
    )
    lines.append("      \\midrule")

    last_col = 2 + len(datasets) * len(n_sessions_list)
    for group_index, (group_name, row_order) in enumerate(groups):
        lines.append(f"      \\multirow{{{len(row_order)}}}{{*}}{{{group_name}}}")
        for row_index, row_key in enumerate(row_order):
            prefix = "      " if row_index > 0 else ""
            row = f"{prefix}& {VARIANT_NAMES[row_key]}"
            for n_sessions in n_sessions_list:
                for metric in METRICS:
                    row += f" & {cell_map.get((dataset, row_key, n_sessions, metric), '-')}"
            row += " \\\\"
            lines.append(row)
        if group_index != len(groups) - 1:
            lines.append("      \\midrule")

    lines.append("      \\bottomrule")
    lines.append("  \\end{tabular}}")
    lines.append(f"  \\label{{{label}}}")
    lines.append("\\end{table}")
    return "\n".join(lines)


def render_cross_dataset_table(cell_map, summary, datasets, n_sessions_list, temperature, random_states, label):
    available_variants = {variant for _, variant, _ in summary.keys()}
    groups = (
        (
            "Position-Bias Estimation",
            tuple(
                variant
                for variant in ("baseline", "pb-adjacent_chain", "pb-ctr", "pb-global_all_pairs", "pb-pivot_one")
                if variant in available_variants
            ),
        ),
        (
            "Additive Two-Tower",
            tuple(
                variant
                for variant in ("bias-add", "reg-add", "full-add", "full-add-shallow")
                if variant in available_variants
            ),
        ),
        (
            "Multiplicative Two-Tower",
            tuple(
                variant
                for variant in ("bias-mul", "reg-mul", "full-mul", "full-mul-shallow")
                if variant in available_variants
            ),
        ),
    )
    groups = tuple((group_name, row_order) for group_name, row_order in groups if row_order)

    col_spec = "ll" + "c" * (len(datasets) * len(n_sessions_list))
    last_col = 2 + len(datasets) * len(n_sessions_list)
    dataset_header = " & ".join(
        f"\\multicolumn{{{len(n_sessions_list)}}}{{c}}{{\\textbf{{{DATASET_LABELS[dataset]}}}}}"
        for dataset in datasets
    )
    cmidrules = "".join(
        f"\\cmidrule(lr){{{3 + idx * len(n_sessions_list)}-{2 + (idx + 1) * len(n_sessions_list)}}}"
        for idx in range(len(datasets))
    )
    session_header = " & ".join(format_n_sessions(n_sessions) for _ in datasets for n_sessions in n_sessions_list)

    lines = []
    lines.append("\\begin{table}[h!]")
    lines.append(
        "  \\caption{Frequency-based DR comparison on \\DatasetMSLR, \\DatasetYahoo, and \\DatasetIstella. "
        "The table includes the oracle-bias baseline in the position-bias estimation group and groups "
        "additive, multiplicative, and CSV position-bias variants separately. Cells show mean (SD) NDCG. "
        "Bold marks the best non-oracle model within each comparison block. Significance markers compare each other model "
        "with this best model using a paired two-sided Student t-test: $p < 0.01$ "
        "($\\,^\\blacktriangle \\text{ and } ^\\blacktriangledown$), $p < 0.05$ "
        "($\\,^\\triangle \\text{ and } ^\\triangledown$).}"
    )
    lines.append("  \\centering")
    lines.append("  \\setlength{\\tabcolsep}{4pt}")
    lines.append("  \\resizebox{\\textwidth}{!}{")
    lines.append(f"    \\begin{{tabular}}{{{col_spec}}}")
    lines.append("      \\toprule")
    lines.append(f"      & & {dataset_header} \\\\")
    lines.append(f"      {cmidrules}")
    lines.append("      \\textbf{Comparison Group} & \\textbf{Model} & " + session_header + " \\\\")
    lines.append("      \\midrule")

    for group_index, (group_name, row_order) in enumerate(groups):
        lines.append(f"      \\multirow{{{len(row_order)}}}{{*}}{{{group_name}}}")
        for row_index, row_key in enumerate(row_order):
            prefix = "      " if row_index > 0 else ""
            row = f"{prefix}& {VARIANT_NAMES[row_key]}"
            for dataset in datasets:
                for n_sessions in n_sessions_list:
                    for metric in METRICS:
                        row += f" & {cell_map.get((dataset, row_key, n_sessions, metric), '-')}"
            row += " \\\\"
            lines.append(row)
            if row_key == "baseline":
                lines.append(f"      \\cmidrule(lr){{2-{last_col}}}")
        if group_index != len(groups) - 1:
            lines.append("      \\midrule")

    lines.append("      \\bottomrule")
    lines.append("  \\end{tabular}}")
    lines.append(f"  \\label{{{label}}}")
    lines.append("\\end{table}")
    return "\n".join(lines)


def write_outputs(table_str, output_name):
    for output_dir in (NOTEBOOK_TABLES_DIR, THESIS_TABLES_DIR):
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / output_name
        with output_path.open("w") as handle:
            handle.write(table_str)
        print(f"Wrote {output_path}")


def print_diagnostics(diagnostics):
    if not diagnostics:
        print("All significance comparisons had at least two matched pairs.")
        return

    print("Significance not computed for the following cells due to fewer than two matched pairs:")
    for item in diagnostics:
        print(
            "  - "
            f"Dataset={DATASET_LABELS[item['dataset']]}, "
            f"Sessions={item['n_sessions']:,}, "
            f"Variant={VARIANT_NAMES[item['variant']]}, "
            f"Reference={VARIANT_NAMES[item['best_variant']]}, "
            f"Metric={item['metric']}, "
            f"matched_pairs={item['matched_pairs']}"
        )


def main():
    args = parse_args()
    n_sessions_list = parse_int_list(args.n_sessions)
    datasets = parse_dataset_list(args.datasets)
    random_states = parse_random_states(args.random_states)
    records = load_records(
        datasets=(args.dataset,) if args.layout == "per-dataset" else datasets,
        n_sessions_list=n_sessions_list,
        temperature=args.temperature,
        random_states=random_states,
    )
    summary = build_summary(records)
    cell_map, diagnostics = build_table_cells(summary, records)
    if args.layout == "per-dataset":
        table_str = render_table(
            cell_map=cell_map,
            summary=summary,
            dataset=args.dataset,
            dataset_label=args.dataset_label,
            n_sessions_list=n_sessions_list,
            temperature=args.temperature,
            random_states=random_states,
            label=args.label,
        )
    else:
        table_str = render_cross_dataset_table(
            cell_map=cell_map,
            summary=summary,
            datasets=datasets,
            n_sessions_list=n_sessions_list,
            temperature=args.temperature,
            random_states=random_states,
            label=args.label,
        )
    write_outputs(table_str, args.output_name)
    print_diagnostics(diagnostics)


if __name__ == "__main__":
    main()
