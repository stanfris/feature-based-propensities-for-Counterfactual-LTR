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
DATASET_ROW_LABELS = {
    "mslr30k": "\\DatasetMSLR",
    "yahoo": "\\DatasetYahoo",
    "istella": "\\DatasetIstella",
}
OBJECTIVE_NAMES = {
    "dm": "DM",
    "dr": "DR",
}
MODEL_NAMES = {
    "regular": "True Bias$^{\\mathrm{oracle}}$",
    "add-two-tower": "Add.\\ Two-Tower",
    "mul-two-tower": "Mul.\\ Two-Tower",
    "pb-adjacent_chain": "Adjacent Chain",
    "pb-ctr": "CTR",
    "pb-global_all_pairs": "Global All Pairs",
    "pb-pivot_one": "Pivot One",
}
NOTEBOOK_TABLES_DIR = project_root / "notebooks" / "thesis_tables"
THESIS_TABLES_DIR = project_root.parent / "Thesis" / "tables"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate LaTeX tables comparing regular and two-tower regression estimators."
    )
    parser.add_argument(
        "--layout",
        choices=("per-dataset", "cross-dataset"),
        default="per-dataset",
        help="Choose between the original per-dataset layout and the new combined cross-dataset layout.",
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
    parser.add_argument("--propensity-model", default="frequency-based")
    parser.add_argument("--random-states", default="40,41,42")
    parser.add_argument(
        "--output-name",
        default="tabel_two_tower_dmdr_regression_comparison.txt",
        help="Filename used for both output table locations.",
    )
    parser.add_argument(
        "--label",
        default="tab:two_tower_dmdr_regression_comparison",
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


def load_records(datasets, n_sessions_list, temperature, propensity_model, random_states, include_pb_models=False):
    records = []
    wanted_n_sessions = set(n_sessions_list)
    wanted_datasets = set(datasets)
    experiment_specs = (
        ("real_targets", "regular"),
        ("two_tower_for_DMDR", "add-two-tower"),
        ("two_tower_for_DMDR", "mul-two-tower"),
    )
    if include_pb_models:
        experiment_specs = experiment_specs + (("position_bias_models", "csv"),)

    for experiment, model in experiment_specs:
        base_dir = project_root / "results" / experiment
        for path in base_dir.rglob("ips_results.json"):
            parsed = parse_folder_name(path.parent.name)
            if parsed.get("data") not in wanted_datasets:
                continue
            if parsed.get("ips.model") not in OBJECTIVE_NAMES:
                continue
            if parsed.get("propensity_model") != propensity_model:
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

            if experiment == "real_targets":
                if normalize_two_tower_name(parsed.get("ips.position_bias.source")) is not None:
                    continue
                if normalize_two_tower_name(parsed.get("ips.regression.method")) not in {None, "regression"}:
                    continue
            elif experiment == "two_tower_for_DMDR":
                if normalize_two_tower_name(parsed.get("ips.regression.method")) != model:
                    continue
            else:
                if parsed.get("ips.position_bias.source") != "csv":
                    continue
                model = normalize_position_bias_model(parsed.get("ips.position_bias.csv.method"))
                if model not in MODEL_NAMES:
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
                    "objective": parsed["ips.model"],
                    "model": model,
                    "n_sessions": int(raw_n_sessions),
                    "temperature": float(raw_temperature),
                    "random_state": int(raw_random_state),
                    "metric_values": {metric: float(metrics[metric]) for metric in METRICS},
                }
            )

    if not records:
        raise RuntimeError("No matching records found for the requested DM/DR regression comparison.")
    return records


def build_summary(records):
    grouped = defaultdict(lambda: defaultdict(list))
    for record in records:
        key = (record["dataset"], record["n_sessions"], record["objective"], record["model"])
        for metric, value in record["metric_values"].items():
            grouped[key][metric].append(value)
    return grouped


def infer_model_order(summary, preferred_order=None):
    if preferred_order is None:
        preferred_order = ("regular", "add-two-tower", "mul-two-tower")
    available_models = {model for _, _, _, model in summary.keys()}
    model_order = tuple(model for model in preferred_order if model in available_models)
    if not model_order:
        raise RuntimeError("No models found for the requested comparison.")
    return model_order


def infer_method_order(summary):
    preferred_order = ("regular", "add-two-tower", "mul-two-tower")
    return infer_model_order(summary, preferred_order=preferred_order)


def build_table_cells(summary, records):
    paired = defaultdict(lambda: defaultdict(dict))
    for record in records:
        pair_id = (record["n_sessions"], record["temperature"], record["random_state"])
        for metric, value in record["metric_values"].items():
            paired[(record["dataset"], record["n_sessions"], record["objective"], metric)][pair_id][record["model"]] = value

    cell_map = {}
    diagnostics = []
    best_model_map = {}

    for (dataset, n_sessions, objective, model), metrics in summary.items():
        for metric in metrics:
            context_key = (dataset, n_sessions, objective, metric)
            if context_key in best_model_map:
                continue

            candidate_models = [
                candidate_model
                for candidate_model in MODEL_NAMES
                if candidate_model != "regular"
                and (dataset, n_sessions, objective, candidate_model) in summary
                and metric in summary[(dataset, n_sessions, objective, candidate_model)]
            ]
            if not candidate_models:
                best_model_map[context_key] = None
                continue

            best_model_map[context_key] = max(
                candidate_models,
                key=lambda candidate_model: (
                    sum(summary[(dataset, n_sessions, objective, candidate_model)][metric])
                    / len(summary[(dataset, n_sessions, objective, candidate_model)][metric]),
                    -tuple(MODEL_NAMES).index(candidate_model),
                ),
            )

    for (dataset, n_sessions, objective, model), metrics in summary.items():
        for metric, values in metrics.items():
            mean_value = sum(values) / len(values)
            std_value = sample_std(values)
            context_key = (dataset, n_sessions, objective, metric)
            best_model = best_model_map.get(context_key)

            if best_model is None:
                cell_map[(dataset, n_sessions, objective, model, metric)] = format_summary_cell(
                    mean_value,
                    std_value,
                    "",
                    bold=False,
                )
                continue

            if model == best_model:
                cell_map[(dataset, n_sessions, objective, model, metric)] = format_summary_cell(
                    mean_value,
                    std_value,
                    "",
                    bold=True,
                )
                continue

            baseline_values = summary.get((dataset, n_sessions, objective, best_model), {}).get(metric, [])
            matched_values = []
            matched_baseline_values = []
            for pair in paired.get((dataset, n_sessions, objective, metric), {}).values():
                if model in pair and best_model in pair:
                    matched_values.append(pair[model])
                    matched_baseline_values.append(pair[best_model])

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
                        "n_sessions": n_sessions,
                        "objective": objective,
                        "model": model,
                        "best_model": best_model,
                        "metric": metric,
                        "matched_pairs": matched_count,
                    }
                )
            cell_map[(dataset, n_sessions, objective, model, metric)] = cell

    return cell_map, diagnostics


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


def render_per_dataset_table(summary, cell_map, dataset, dataset_label, n_sessions_list, temperature, propensity_model, random_states, label):
    method_order = infer_method_order(summary)
    objective_order = ("dm", "dr")
    header_groups = " ".join(f"& \\textbf{{{MODEL_NAMES[method]}}}" for method in method_order)
    if method_order == ("regular", "add-two-tower", "mul-two-tower"):
        method_phrase = "the additive and multiplicative two-tower"
    elif method_order == ("regular", "add-two-tower"):
        method_phrase = "the additive two-tower"
    elif method_order == ("regular", "mul-two-tower"):
        method_phrase = "the multiplicative two-tower"
    else:
        method_phrase = "the available two-tower"

    lines = []
    lines.append("\\begin{table}[h!]")
    lines.append(
        f"  \\caption{{Comparison of the default regression estimator and {method_phrase} regression estimator "
        f"for DM and DR on {dataset_label}. Cells show mean (SD) NDCG. Bold marks the best non-oracle "
        "model within each comparison block. Significance markers compare each other model "
        "with this best model using a paired two-sided Student t-test: $p < 0.01$ "
        "($\\,^\\blacktriangle \\text{ and } ^\\blacktriangledown$), $p < 0.05$ "
        "($\\,^\\triangle \\text{ and } ^\\triangledown$).}"
    )
    lines.append("  \\centering")
    lines.append("  \\setlength{\\tabcolsep}{5pt}")
    lines.append("  \\resizebox{\\textwidth}{!}{")
    lines.append(f"  \\begin{{tabular}}{{ll{'c' * len(method_order)}}}")
    lines.append("    \\toprule")
    lines.append(f"    & {header_groups} \\\\")
    lines.append(
        "    \\textbf{Sessions} & \\textbf{Objective} & "
        + " & ".join(["NDCG"] * len(method_order))
        + " \\\\"
    )
    lines.append("    \\midrule")

    for index, n_sessions in enumerate(n_sessions_list):
        lines.append(f"    \\multirow{{{len(objective_order)}}}{{*}}{{{format_n_sessions(n_sessions)}}}")
        for row_index, objective in enumerate(objective_order):
            row_prefix = "    " if row_index > 0 else ""
            row = f"{row_prefix}& {OBJECTIVE_NAMES[objective]}"
            for method in method_order:
                for metric in METRICS:
                    row += f" & {cell_map.get((dataset, n_sessions, objective, method, metric), '-')}"
            row += " \\\\"
            lines.append(row)
        if index != len(n_sessions_list) - 1:
            lines.append("    \\midrule")

    lines.append("    \\bottomrule")
    lines.append("  \\end{tabular}}")
    lines.append(f"  \\label{{{label}}}")
    lines.append("\\end{table}")
    return "\n".join(lines)


def render_cross_dataset_table(
    summary,
    cell_map,
    datasets,
    n_sessions_list,
    temperature,
    propensity_model,
    random_states,
    label,
):
    model_order = infer_model_order(
        summary,
        preferred_order=(
            "regular",
            "add-two-tower",
            "mul-two-tower",
            "pb-adjacent_chain",
            "pb-ctr",
            "pb-global_all_pairs",
            "pb-pivot_one",
        ),
    )

    lines = []
    lines.append("\\begin{table}[h!]")
    lines.append(
        "  \\caption{Comparison of the oracle-bias baseline, additive and multiplicative two-tower regression "
        "estimators, and CSV position-bias baselines for DM and DR on \\DatasetMSLR, \\DatasetYahoo, and "
        "\\DatasetIstella. Rows are grouped by dataset. Cells show mean (SD) NDCG. Bold marks the best "
        "non-oracle model within each comparison block. Significance markers compare each other model "
        "with this best model using a paired two-sided Student t-test: $p < 0.01$ "
        "($\\,^\\blacktriangle \\text{ and } ^\\blacktriangledown$), $p < 0.05$ "
        "($\\,^\\triangle \\text{ and } ^\\triangledown$).}"
    )
    lines.append("  \\centering")
    lines.append("  \\setlength{\\tabcolsep}{5pt}")
    lines.append("  \\resizebox{\\textwidth}{!}{")
    lines.append(f"    \\begin{{tabular}}{{ll{'c' * (len(OBJECTIVE_NAMES) * len(n_sessions_list))}}}")
    lines.append("      \\toprule")
    objective_header = " ".join(
        f"& \\multicolumn{{{len(n_sessions_list)}}}{{c}}{{\\textbf{{{OBJECTIVE_NAMES[objective]}}}}}"
        for objective in ("dm", "dr")
    )
    lines.append(f"      & {objective_header} \\\\")
    cmidrules = "".join(
        f"\\cmidrule(lr){{{3 + idx * len(n_sessions_list)}-{2 + (idx + 1) * len(n_sessions_list)}}}"
        for idx in range(len(OBJECTIVE_NAMES))
    )
    lines.append(f"      {cmidrules}")
    session_header = " & ".join(
        format_n_sessions_header(n_sessions) for _ in ("dm", "dr") for n_sessions in n_sessions_list
    )
    lines.append(
        "      \\textbf{Dataset} & \\textbf{Model} & "
        + session_header
        + " \\\\"
    )
    lines.append("      \\midrule")

    last_col = 2 + len(("dm", "dr")) * len(n_sessions_list)
    for dataset_index, dataset in enumerate(datasets):
        lines.append(
            f"      \\multirow{{{len(model_order)}}}{{*}}{{{DATASET_ROW_LABELS[dataset]}}}"
            f" & {MODEL_NAMES[model_order[0]]}"
            + "".join(
                f" & {cell_map.get((dataset, n_sessions, objective, model_order[0], 'NDCG'), '-')}"
                for objective in ("dm", "dr")
                for n_sessions in n_sessions_list
            )
            + " \\\\"
        )
        lines.append(f"      \\cmidrule(lr){{2-{last_col}}}")
        for model in model_order[1:]:
            row = f"      & {MODEL_NAMES[model]}"
            for objective in ("dm", "dr"):
                for n_sessions in n_sessions_list:
                    row += f" & {cell_map.get((dataset, n_sessions, objective, model, 'NDCG'), '-')}"
            row += " \\\\"
            lines.append(row)
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
            f"Sessions={item['n_sessions']:,}, "
            f"Objective={OBJECTIVE_NAMES[item['objective']]}, "
            f"Model={MODEL_NAMES[item['model']]}, "
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
        propensity_model=args.propensity_model,
        random_states=random_states,
        include_pb_models=args.layout == "cross-dataset",
    )
    summary = build_summary(records)
    cell_map, diagnostics = build_table_cells(summary, records)
    if args.layout == "per-dataset":
        table_str = render_per_dataset_table(
            summary=summary,
            cell_map=cell_map,
            dataset=args.dataset,
            dataset_label=args.dataset_label,
            n_sessions_list=n_sessions_list,
            temperature=args.temperature,
            propensity_model=args.propensity_model,
            random_states=random_states,
            label=args.label,
        )
    else:
        table_str = render_cross_dataset_table(
            summary=summary,
            cell_map=cell_map,
            datasets=datasets,
            n_sessions_list=n_sessions_list,
            temperature=args.temperature,
            propensity_model=args.propensity_model,
            random_states=random_states,
            label=args.label,
        )
    write_outputs(table_str, args.output_name)
    print_diagnostics(diagnostics)


if __name__ == "__main__":
    main()
