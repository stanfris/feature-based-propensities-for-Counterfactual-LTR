import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

from table_significance import format_summary_cell, get_sig_symbol, get_valid_pairs, paired_test_pvalue, sample_std


current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
os.chdir(project_root)


METRICS = ("NDCG",)
OBJECTIVE_NAMES = {
    "ips": "IPS",
    "dm": "DM",
    "dr": "DR",
}
DATASET_LABELS = {
    "mslr30k": "MSLR-WEB30K",
    "yahoo": "Yahoo!",
    "istella": "Istella-S",
}
DATASET_ROW_LABELS = {
    "mslr30k": "\\DatasetMSLR",
    "yahoo": "\\DatasetYahoo",
    "istella": "\\DatasetIstella",
}
PROP_NAMES = {
    "frequency-based": "Frequency-Based",
    "MLPregression": "MLP",
    "true_propensity": "Oracle Propensity",
}
PROP_CAPTION_NAMES = {
    "frequency-based": "frequency-based",
    "MLPregression": "MLP",
    "true_propensity": "oracle propensity baseline",
}
SOURCE_NAMES = {
    "real-targets": "True Bias$^{\\mathrm{oracle}}$",
    "add-two-tower": "Add.\\ Two-Tower Bias",
    "mul-two-tower": "Mul.\\ Two-Tower Bias",
    "pb-adjacent_chain": "Adjacent Chain",
    "pb-ctr": "CTR",
    "pb-global_all_pairs": "Global All Pairs",
    "pb-pivot_one": "Pivot One",
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
NOTEBOOK_TABLES_DIR = Path("notebooks/thesis_tables")
THESIS_TABLES_DIR = Path("..") / "Thesis" / "tables"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate LaTeX tables comparing real_targets, two_tower_bias, "
            "and CSV position-bias models across datasets."
        )
    )
    parser.add_argument(
        "--layout",
        choices=("per-session", "per-propensity"),
        default="per-session",
        help="Choose between the original single-session layout and the propensity-specific cross-dataset layout.",
    )
    parser.add_argument(
        "--datasets",
        default="mslr30k,yahoo,istella",
        help="Comma-separated datasets to include.",
    )
    parser.add_argument(
        "--n-sessions",
        default="10000",
        help="Comma-separated session counts to include.",
    )
    parser.add_argument(
        "--objectives",
        default="ips,dm,dr",
        help="Comma-separated objectives to include.",
    )
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--random-states", default="40,41,42")
    parser.add_argument(
        "--propensity-model",
        choices=tuple(PROP_NAMES),
        help="Required for the per-propensity layout.",
    )
    parser.add_argument(
        "--output-name",
        default="tabel_two_tower_bias_all_datasets_10k.txt",
        help="Filename used for both output table locations.",
    )
    parser.add_argument(
        "--label",
        default="tab:two_tower_bias_all_datasets_10k",
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


def parse_objectives(spec):
    return tuple(part.strip() for part in spec.split(",") if part.strip())


def parse_datasets(spec):
    return tuple(part.strip() for part in spec.split(",") if part.strip())


def normalize_two_tower_name(value):
    aliases = {
        "add_two_tower": "add-two-tower",
        "mul_two_tower": "mul-two-tower",
    }
    return aliases.get(value, value)


def safe_load_json(path):
    try:
        with path.open("r") as handle:
            return json.load(handle)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def load_records(datasets, n_sessions_values, objectives, temperature, random_states):
    records = []
    wanted_datasets = set(datasets)
    wanted_n_sessions = set(n_sessions_values)
    wanted_objectives = set(objectives)
    for experiment in ("real_targets", "two_tower_bias", "position_bias_models"):
        base_dir = Path("results") / experiment
        for path in base_dir.rglob("ips_results.json"):
            parsed = parse_folder_name(path.parent.name)
            if parsed.get("data") not in wanted_datasets:
                continue
            if parsed.get("ips.model") not in wanted_objectives:
                continue
            if parsed.get("propensity_model") not in PROP_NAMES:
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

            source = normalize_two_tower_name(parsed.get("ips.position_bias.source", "real-targets"))
            if experiment == "real_targets" and source != "real-targets":
                continue
            if experiment == "two_tower_bias" and source not in {"add-two-tower", "mul-two-tower"}:
                continue
            if experiment == "position_bias_models":
                method = parsed.get("ips.position_bias.csv.method")
                if method not in {"adjacent_chain", "ctr", "global_all_pairs", "pivot_one"}:
                    continue
                source = f"pb-{method}"

            payload = safe_load_json(path)
            if not payload:
                continue
            metrics = payload.get("results", {}).get("metrics", {})
            if not all(metric in metrics for metric in METRICS):
                continue

            records.append(
                {
                    "dataset": parsed["data"],
                    "objective": parsed["ips.model"],
                    "propensity_model": parsed["propensity_model"],
                    "source": source,
                    "n_sessions": int(raw_n_sessions),
                    "temperature": float(raw_temperature),
                    "random_state": int(raw_random_state),
                    "metric_values": {metric: float(metrics[metric]) for metric in METRICS},
                }
            )

    if not records:
        raise RuntimeError("No matching records found for the requested cross-dataset position-bias comparison.")
    return records


def build_summary(records):
    grouped = defaultdict(lambda: defaultdict(list))
    for record in records:
        key = (record["dataset"], record["objective"], record["propensity_model"], record["source"], record["n_sessions"])
        for metric, value in record["metric_values"].items():
            grouped[key][metric].append(value)
    return grouped


def build_table_cells(summary, records):
    paired = defaultdict(lambda: defaultdict(dict))
    for record in records:
        pair_id = (record["n_sessions"], record["temperature"], record["random_state"])
        for metric, value in record["metric_values"].items():
            paired[(record["dataset"], record["objective"], record["propensity_model"], record["n_sessions"], metric)][pair_id][
                record["source"]
            ] = value

    cell_map = {}
    diagnostics = []
    best_source_map = {}

    for (dataset, objective, propensity_model, source, n_sessions), metrics in summary.items():
        for metric in metrics:
            context_key = (dataset, objective, propensity_model, n_sessions, metric)
            if context_key in best_source_map:
                continue

            candidate_sources = [
                candidate_source
                for candidate_source in SOURCE_ORDER
                if candidate_source != "real-targets"
                and (dataset, objective, propensity_model, candidate_source, n_sessions) in summary
                and metric in summary[(dataset, objective, propensity_model, candidate_source, n_sessions)]
            ]
            if not candidate_sources:
                best_source_map[context_key] = None
                continue

            best_source_map[context_key] = max(
                candidate_sources,
                key=lambda candidate_source: (
                    sum(summary[(dataset, objective, propensity_model, candidate_source, n_sessions)][metric])
                    / len(summary[(dataset, objective, propensity_model, candidate_source, n_sessions)][metric]),
                    -SOURCE_ORDER.index(candidate_source),
                ),
            )

    for (dataset, objective, propensity_model, source, n_sessions), metrics in summary.items():
        for metric, values in metrics.items():
            mean_value = sum(values) / len(values)
            std_value = sample_std(values)
            context_key = (dataset, objective, propensity_model, n_sessions, metric)
            best_source = best_source_map.get(context_key)

            if best_source is None:
                cell_map[(dataset, objective, propensity_model, source, n_sessions, metric)] = format_summary_cell(
                    mean_value,
                    std_value,
                    "",
                    bold=False,
                )
                continue

            if source == best_source:
                cell_map[(dataset, objective, propensity_model, source, n_sessions, metric)] = format_summary_cell(
                    mean_value,
                    std_value,
                    "",
                    bold=True,
                )
                continue

            baseline_values = summary.get((dataset, objective, propensity_model, best_source, n_sessions), {}).get(metric, [])
            matched_values = []
            matched_baseline_values = []
            for pair in paired.get((dataset, objective, propensity_model, n_sessions, metric), {}).values():
                if source in pair and best_source in pair:
                    matched_values.append(pair[source])
                    matched_baseline_values.append(pair[best_source])

            valid_pairs = get_valid_pairs(matched_values, matched_baseline_values)
            p_value = paired_test_pvalue(
                [x for x, _ in valid_pairs],
                [y for _, y in valid_pairs],
            )
            base_mean = sum(baseline_values) / len(baseline_values) if baseline_values else mean_value
            symbol = get_sig_symbol(mean_value, base_mean, p_value)
            cell = format_summary_cell(mean_value, std_value, symbol, bold=False)
            matched_count = len(valid_pairs)
            if baseline_values and matched_count < 2:
                diagnostics.append(
                    {
                        "dataset": dataset,
                        "objective": objective,
                        "propensity_model": propensity_model,
                        "source": source,
                        "best_source": best_source,
                        "n_sessions": n_sessions,
                        "metric": metric,
                        "matched_pairs": matched_count,
                    }
                )
            cell_map[(dataset, objective, propensity_model, source, n_sessions, metric)] = cell

    return cell_map, diagnostics


def available_sources_for_prop(summary, prop, datasets, n_sessions_values, objectives):
    sources = []
    for source in SOURCE_ORDER:
        found = any(
            (dataset, objective, prop, source, n_sessions) in summary
            for dataset in datasets
            for n_sessions in n_sessions_values
            for objective in objectives
        )
        if found:
            sources.append(source)
    return sources


def format_n_sessions(n_sessions):
    sample_n_macros = {
        10_000: "\\sampleN{4}",
        100_000: "\\sampleN{5}",
    }
    return sample_n_macros.get(n_sessions, f"{n_sessions:,}")


def format_n_sessions_header(n_sessions):
    sample_n_header_macros = {
        10_000: "\\sampleNheader{4}",
        100_000: "\\sampleNheader{5}",
    }
    return sample_n_header_macros.get(n_sessions, f"\\textbf{{{format_n_sessions(n_sessions)}}}")


def render_per_session_table(summary, cell_map, datasets, n_sessions, objectives, temperature, random_states, label):
    lines = []
    lines.append("\\begin{table}[h!]")
    dataset_phrase = ", ".join(DATASET_LABELS[dataset] for dataset in datasets)
    lines.append(
        "  \\caption{Comparison of matched oracle-bias, two-tower-bias, and CSV position-bias estimators on "
        f"{dataset_phrase}. Columns are grouped by dataset and counterfactual learning objective. "
        "Cells show mean (SD) NDCG. Bold marks the best non-oracle model within each comparison block. "
        "Significance markers compare each other model "
        "with this best model using a paired two-sided Student t-test: $p < 0.01$ "
        "($\\,^\\blacktriangle \\text{ and } ^\\blacktriangledown$), $p < 0.05$ "
        "($\\,^\\triangle \\text{ and } ^\\triangledown$).}"
    )
    lines.append("  \\centering")
    lines.append("  \\setlength{\\tabcolsep}{4pt}")
    lines.append("  \\resizebox{\\textwidth}{!}{")
    lines.append(f"    \\begin{{tabular}}{{ll{'c' * (len(datasets) * len(objectives))}}}")
    lines.append("      \\toprule")
    dataset_header = " ".join(
        f"& \\multicolumn{{{len(objectives)}}}{{c}}{{\\textbf{{{DATASET_LABELS[dataset]}}}}}"
        for dataset in datasets
    )
    lines.append(f"      & {dataset_header} \\\\")
    cmidrules = "".join(
        f"\\cmidrule(lr){{{3 + idx * len(objectives)}-{2 + (idx + 1) * len(objectives)}}}"
        for idx in range(len(datasets))
    )
    lines.append(f"      {cmidrules}")
    objective_header = " & ".join(OBJECTIVE_NAMES[objective] for _ in datasets for objective in objectives)
    lines.append("      \\textbf{Propensity} & \\textbf{Bias Source} & " + objective_header + " \\\\")
    lines.append("      \\midrule")

    prop_order = ("frequency-based", "MLPregression", "true_propensity")
    last_col = 2 + len(datasets) * len(objectives)
    for prop_index, prop in enumerate(prop_order):
        source_order = available_sources_for_prop(summary, prop, datasets, (n_sessions,), objectives)
        lines.append(f"      \\multirow{{{len(source_order)}}}{{*}}{{{PROP_NAMES[prop]}}}")
        for source in source_order:
            row = f"      & {SOURCE_NAMES[source]}"
            for dataset in datasets:
                for objective in objectives:
                    for metric in METRICS:
                        row += f" & {cell_map.get((dataset, objective, prop, source, n_sessions, metric), '-')}"
            row += " \\\\"
            lines.append(row)
            if source == "real-targets":
                lines.append(f"      \\cmidrule(lr){{2-{last_col}}}")
        if prop_index != len(prop_order) - 1:
            lines.append("      \\midrule")

    lines.append("      \\bottomrule")
    lines.append("  \\end{tabular}}")
    lines.append(f"  \\label{{{label}}}")
    lines.append("\\end{table}")
    return "\n".join(lines)


def render_per_propensity_table(
    summary,
    cell_map,
    datasets,
    n_sessions_values,
    objectives,
    temperature,
    random_states,
    propensity_model,
    label,
):
    lines = []
    lines.append("\\begin{table}[h!]")
    dataset_phrase = ", ".join(DATASET_LABELS[dataset] for dataset in datasets)
    lines.append(
        "  \\caption{Comparison of the oracle-bias baseline, learned two-tower bias estimators, and CSV "
        "position-bias estimators on "
        f"{dataset_phrase} under the {PROP_CAPTION_NAMES[propensity_model]} setting. Rows are grouped by "
        "dataset, and columns are grouped first by counterfactual learning objective and then by sample "
        "count. Cells show mean (SD) NDCG. Bold marks the best non-oracle model within each comparison block. "
        "Significance markers compare each other model "
        "with this best model using a paired two-sided Student t-test: $p < 0.01$ "
        "($\\,^\\blacktriangle \\text{ and } ^\\blacktriangledown$), $p < 0.05$ "
        "($\\,^\\triangle \\text{ and } ^\\triangledown$).}"
    )
    lines.append("  \\centering")
    lines.append("  \\setlength{\\tabcolsep}{4pt}")
    lines.append("  \\resizebox{\\textwidth}{!}{")
    lines.append(f"    \\begin{{tabular}}{{ll{'c' * (len(n_sessions_values) * len(objectives))}}}")
    lines.append("      \\toprule")
    objective_header = " ".join(
        f"& \\multicolumn{{{len(n_sessions_values)}}}{{c}}{{\\textbf{{{OBJECTIVE_NAMES[objective]}}}}}"
        for objective in objectives
    )
    lines.append(f"      & {objective_header} \\\\")
    cmidrules = "".join(
        f"\\cmidrule(lr){{{3 + idx * len(n_sessions_values)}-{2 + (idx + 1) * len(n_sessions_values)}}}"
        for idx in range(len(objectives))
    )
    lines.append(f"      {cmidrules}")
    session_header = " & ".join(
        format_n_sessions_header(n_sessions) for _ in objectives for n_sessions in n_sessions_values
    )
    lines.append("      \\textbf{Dataset} & \\textbf{Bias Source} & " + session_header + " \\\\")
    lines.append("      \\midrule")

    last_col = 2 + len(n_sessions_values) * len(objectives)
    for dataset_index, dataset in enumerate(datasets):
        lines.append(f"      \\multirow{{{len(SOURCE_ORDER)}}}{{*}}{{{DATASET_ROW_LABELS[dataset]}}}")
        for source in SOURCE_ORDER:
            row = f"      & {SOURCE_NAMES[source]}"
            for objective in objectives:
                for n_sessions in n_sessions_values:
                    for metric in METRICS:
                        row += f" & {cell_map.get((dataset, objective, propensity_model, source, n_sessions, metric), '-')}"
            row += " \\\\"
            lines.append(row)
            if source == "real-targets":
                lines.append(f"      \\cmidrule(lr){{2-{last_col}}}")
        if dataset_index != len(datasets) - 1:
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
            f"Objective={OBJECTIVE_NAMES[item['objective']]}, "
            f"Propensity={PROP_NAMES[item['propensity_model']]}, "
            f"Bias Source={SOURCE_NAMES[item['source']]}, "
            f"Sessions={item['n_sessions']:,}, "
            f"Metric={item['metric']}, "
            f"matched_pairs={item['matched_pairs']}"
        )


def main():
    args = parse_args()
    datasets = parse_datasets(args.datasets)
    n_sessions_values = parse_int_list(args.n_sessions)
    objectives = parse_objectives(args.objectives)
    random_states = parse_random_states(args.random_states)
    records = load_records(
        datasets=datasets,
        n_sessions_values=n_sessions_values,
        objectives=objectives,
        temperature=args.temperature,
        random_states=random_states,
    )
    summary = build_summary(records)
    cell_map, diagnostics = build_table_cells(summary, records)
    if args.layout == "per-session":
        if len(n_sessions_values) != 1:
            raise ValueError("The per-session layout requires exactly one session count.")
        table_str = render_per_session_table(
            summary=summary,
            cell_map=cell_map,
            datasets=datasets,
            n_sessions=n_sessions_values[0],
            objectives=objectives,
            temperature=args.temperature,
            random_states=random_states,
            label=args.label,
        )
    else:
        if not args.propensity_model:
            raise ValueError("The per-propensity layout requires --propensity-model.")
        table_str = render_per_propensity_table(
            summary=summary,
            cell_map=cell_map,
            datasets=datasets,
            n_sessions_values=n_sessions_values,
            objectives=objectives,
            temperature=args.temperature,
            random_states=random_states,
            propensity_model=args.propensity_model,
            label=args.label,
        )
    write_outputs(table_str, args.output_name)
    print_diagnostics(diagnostics)


if __name__ == "__main__":
    main()
