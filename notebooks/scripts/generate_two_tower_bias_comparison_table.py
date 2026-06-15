import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

from table_significance import format_summary_with_significance, sample_std


current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
os.chdir(project_root)


METRICS = ("NDCG",)
PROP_NAMES = {
    "frequency-based": "Frequency-Based",
    "MLPregression": "MLP",
    "true_propensity": "Oracle Propensity",
}
OBJECTIVE_NAMES = {
    "ips": "IPS",
    "dm": "DM",
    "dr": "DR",
}
SOURCE_NAMES = {
    "real-targets": "Real Targets",
    "add-two-tower": "Add. Two-Tower Bias",
    "mul-two-tower": "Mul. Two-Tower Bias",
    "pb-adjacent_chain": "PB: Adjacent Chain",
    "pb-ctr": "PB: CTR",
    "pb-global_all_pairs": "PB: Global All Pairs",
    "pb-pivot_one": "PB: Pivot One",
}
NOTEBOOK_TABLES_DIR = Path("notebooks/thesis_tables")
THESIS_TABLES_DIR = Path("..") / "Thesis" / "tables"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a LaTeX table comparing two_tower_bias with matching real_targets runs."
    )
    parser.add_argument("--dataset", default="mslr30k")
    parser.add_argument("--dataset-label", default="MSLR-WEB30K")
    parser.add_argument("--n-sessions", type=int, default=100000)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--random-states", default="40,41,42")
    parser.add_argument(
        "--output-name",
        default="tabel_two_tower_bias_vs_real_targets.txt",
        help="Filename used for both output table locations.",
    )
    return parser.parse_args()


def parse_folder_name(folder_name):
    parsed = {}
    for part in folder_name.split(","):
        if "=" in part:
            key, value = part.lstrip("+").split("=", 1)
            parsed[key] = value
    return parsed


def normalize_two_tower_name(value):
    aliases = {
        "add_two_tower": "add-two-tower",
        "mul_two_tower": "mul-two-tower",
    }
    return aliases.get(value, value)


def parse_random_states(spec):
    return {int(part.strip()) for part in spec.split(",") if part.strip()}


def safe_load_json(path):
    try:
        with path.open("r") as handle:
            return json.load(handle)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def load_records(dataset, n_sessions, temperature, random_states):
    records = []
    for experiment in ("real_targets", "two_tower_bias", "position_bias_models"):
        base_dir = Path("results") / experiment
        for path in base_dir.rglob("ips_results.json"):
            parsed = parse_folder_name(path.parent.name)
            if parsed.get("data") != dataset:
                continue
            if parsed.get("ips.model") not in OBJECTIVE_NAMES:
                continue
            if parsed.get("propensity_model") not in PROP_NAMES:
                continue

            raw_n_sessions = parsed.get("ips.n_sessions")
            raw_temperature = parsed.get("policy_temperature")
            raw_random_state = parsed.get("random_state")
            if raw_n_sessions is None or int(raw_n_sessions) != n_sessions:
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
        raise RuntimeError("No matching records found for the requested comparison.")
    return records


def build_summary(records):
    grouped = defaultdict(lambda: defaultdict(list))
    for record in records:
        key = (record["objective"], record["propensity_model"], record["source"])
        for metric, value in record["metric_values"].items():
            grouped[key][metric].append(value)
    return grouped


def build_table_cells(summary, records):
    paired = defaultdict(lambda: defaultdict(dict))
    for record in records:
        pair_id = (record["n_sessions"], record["temperature"], record["random_state"])
        for metric, value in record["metric_values"].items():
            paired[(record["objective"], record["propensity_model"], metric)][pair_id][record["source"]] = value

    cell_map = {}
    diagnostics = []

    for (objective, propensity_model, source), metrics in summary.items():
        for metric, values in metrics.items():
            if source == "real-targets":
                cell, _, _ = format_summary_with_significance(values)
                cell_map[(objective, propensity_model, source, metric)] = cell
                continue

            baseline_values = summary.get((objective, propensity_model, "real-targets"), {}).get(metric, [])
            matched_values = []
            matched_baseline_values = []
            for pair in paired.get((objective, propensity_model, metric), {}).values():
                if source in pair and "real-targets" in pair:
                    matched_values.append(pair[source])
                    matched_baseline_values.append(pair["real-targets"])

            cell, matched_count, _ = format_summary_with_significance(
                values,
                base_values=baseline_values,
                matched_values=matched_values,
                matched_base_values=matched_baseline_values,
            )
            if baseline_values and matched_count < 2:
                diagnostics.append(
                    {
                        "propensity_model": propensity_model,
                        "source": source,
                        "objective": objective,
                        "metric": metric,
                        "matched_pairs": matched_count,
                    }
                )
            cell_map[(objective, propensity_model, source, metric)] = cell

    return cell_map, diagnostics


def render_table(cell_map, dataset_label, n_sessions, temperature, random_states):
    base_sources = ("real-targets", "add-two-tower", "mul-two-tower")
    position_bias_sources = ("pb-adjacent_chain", "pb-ctr", "pb-global_all_pairs", "pb-pivot_one")
    objectives = ("ips", "dm", "dr")
    prop_order = ("frequency-based", "MLPregression", "true_propensity")

    header_groups = " ".join(f"& \\textbf{{{OBJECTIVE_NAMES[objective]}}}" for objective in objectives)

    lines = []
    lines.append("\\begin{table}[h!]")
    lines.append(
        "  \\caption{Comparison of matched oracle-bias, two-tower-bias, and CSV position-bias estimators on "
        f"{dataset_label}. Columns are grouped by counterfactual learning objective. "
        "Cells show mean (SD) NDCG. Bold marks the best non-oracle model within each comparison block. "
        "Significance markers compare each other model "
        "with this best model using a paired two-sided Student t-test: $p < 0.01$ "
        "($\\,^\\blacktriangle \\text{ and } ^\\blacktriangledown$), $p < 0.05$ "
        "($\\,^\\triangle \\text{ and } ^\\triangledown$).}"
    )
    lines.append("  \\centering")
    lines.append("  \\setlength{\\tabcolsep}{4pt}")
    lines.append("  \\resizebox{0.95\\textwidth}{!}{")
    lines.append("    \\begin{tabular}{llccc}")
    lines.append("      \\toprule")
    lines.append(f"      & {header_groups} \\\\")
    lines.append("      \\textbf{Propensity} & \\textbf{Bias Source} & NDCG & NDCG & NDCG \\\\")
    lines.append("      \\midrule")

    for prop_index, prop in enumerate(prop_order):
        source_order = base_sources + position_bias_sources
        lines.append(f"      \\multirow{{{len(source_order)}}}{{*}}{{{PROP_NAMES[prop]}}}")
        for source in source_order:
            row = f"      & {SOURCE_NAMES[source]}"
            for objective in objectives:
                for metric in METRICS:
                    row += f" & {cell_map.get((objective, prop, source, metric), '-')}"
            row += " \\\\"
            lines.append(row)
        if prop_index != len(prop_order) - 1:
            lines.append("      \\midrule")

    lines.append("      \\bottomrule")
    lines.append("  \\end{tabular}}")
    lines.append("  \\label{tab:two_tower_bias_vs_real_targets}")
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
            f"Propensity={PROP_NAMES[item['propensity_model']]}, "
            f"Bias Source={SOURCE_NAMES[item['source']]}, "
            f"Objective={OBJECTIVE_NAMES[item['objective']]}, "
            f"Metric={item['metric']}, "
            f"matched_pairs={item['matched_pairs']}"
        )


def main():
    args = parse_args()
    random_states = parse_random_states(args.random_states)
    records = load_records(
        dataset=args.dataset,
        n_sessions=args.n_sessions,
        temperature=args.temperature,
        random_states=random_states,
    )
    summary = build_summary(records)
    cell_map, diagnostics = build_table_cells(summary, records)
    table_str = render_table(
        cell_map=cell_map,
        dataset_label=args.dataset_label,
        n_sessions=args.n_sessions,
        temperature=args.temperature,
        random_states=random_states,
    )
    write_outputs(table_str, args.output_name)
    print_diagnostics(diagnostics)


if __name__ == "__main__":
    main()
