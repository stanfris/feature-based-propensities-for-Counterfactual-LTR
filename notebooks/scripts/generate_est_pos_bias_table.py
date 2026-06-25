import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

from table_significance import format_summary_cell, get_sig_symbol, get_valid_pairs, paired_test_pvalue, sample_std


current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
os.chdir(project_root)


TABLES_DIR = Path("notebooks/thesis_tables")
THESIS_TABLES_DIR = Path(__file__).resolve().parents[3] / "Thesis" / "tables"
PROP_ORDER = ("true_propensity", "frequency-based", "MLPregression")
PROP_NAMES = {
    "frequency-based": "Frequency-Based",
    "MLPregression": "MLP",
    "true_propensity": "Oracle Propensity",
}
IPS_MODELS = ("ips", "dm", "dr")
IPS_MODEL_NAMES = {
    "ips": "IPS",
    "dm": "DM",
    "dr": "DR",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Generate estimated-position-bias comparison table.")
    parser.add_argument("--experiment", default="est_pos_bias", help="Experiment folder under results/.")
    parser.add_argument(
        "--datasets",
        default="mslr30k:MSLR-WEB30K,istella:Istella-S,yahoo:Yahoo!",
        help="Comma-separated dataset_key:Dataset Label pairs.",
    )
    parser.add_argument("--n-sessions", type=int, default=10000, help="Session count to include.")
    parser.add_argument("--temperature", type=float, default=0.5, help="Policy temperature to include.")
    parser.add_argument(
        "--random-states",
        default="40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59",
        help="Comma-separated random states to include.",
    )
    parser.add_argument("--output-prefix", default="est_pos_bias", help="Prefix for generated table filenames.")
    return parser.parse_args()


def parse_folder_name(folder_name):
    parsed = {}
    for part in folder_name.split(","):
        if "=" in part:
            k, v = part.lstrip("+").split("=", 1)
            parsed[k] = v
    return parsed


def parse_dataset_labels(spec):
    items = []
    for part in spec.split(","):
        key, label = part.split(":", 1)
        items.append((key, label))
    return tuple(items)


def parse_int_list(spec):
    return tuple(int(part.strip()) for part in spec.split(",") if part.strip())


def safe_load_json(path):
    try:
        with path.open("r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def load_records(experiment, datasets, n_sessions, temperature, random_states):
    base_dir = Path("results") / experiment
    if not base_dir.exists():
        raise FileNotFoundError(f"Results folder not found: {base_dir}")

    wanted_datasets = {key for key, _ in datasets}
    wanted_random_states = set(random_states)
    records = []

    for path in base_dir.rglob("ips_results.json"):
        parsed = parse_folder_name(path.parent.name)
        dataset = parsed.get("data")
        ips_model = parsed.get("ips.model")
        propensity_model = parsed.get("propensity_model")
        raw_n_sessions = parsed.get("ips.n_sessions")
        raw_random_state = parsed.get("random_state")
        raw_temperature = parsed.get("policy_temperature")

        if dataset not in wanted_datasets or ips_model not in IPS_MODELS:
            continue
        if propensity_model not in PROP_ORDER:
            continue
        if raw_n_sessions is None or int(raw_n_sessions) != n_sessions:
            continue
        if raw_random_state is None or int(raw_random_state) not in wanted_random_states:
            continue
        if raw_temperature is None or float(raw_temperature) != temperature:
            continue
        if parsed.get("ips.position_bias.source") != "estimate":
            continue
        if parsed.get("ips.position_bias.estimator") != "global_all_pairs":
            continue

        payload = safe_load_json(path)
        if not payload:
            continue
        metrics = payload.get("results", {}).get("metrics", {})
        if "NDCG" not in metrics:
            continue

        records.append(
            {
                "dataset": dataset,
                "ips_model": ips_model,
                "propensity_model": propensity_model,
                "random_state": int(raw_random_state),
                "value": float(metrics["NDCG"]),
            }
        )

    return records


def build_table_data(records, datasets, random_states):
    values = defaultdict(list)
    paired = defaultdict(dict)

    for record in records:
        key = (record["dataset"], record["propensity_model"], record["ips_model"])
        values[key].append(record["value"])

        pair_key = (record["dataset"], record["ips_model"], record["random_state"])
        paired[pair_key][record["propensity_model"]] = record["value"]

    table_data = defaultdict(lambda: defaultdict(dict))
    diagnostics = []
    dataset_keys = {dataset for dataset, _ in datasets}

    for dataset in dataset_keys:
        for ips_model in IPS_MODELS:
            candidate_props = [
                prop
                for prop in PROP_ORDER
                if prop != "true_propensity" and values.get((dataset, prop, ips_model), [])
            ]
            best_prop = (
                max(
                    candidate_props,
                    key=lambda prop: (
                        sum(values[(dataset, prop, ips_model)]) / len(values[(dataset, prop, ips_model)]),
                        -PROP_ORDER.index(prop),
                    ),
                )
                if candidate_props
                else None
            )

            base_values = []
            for random_state in random_states:
                pair = paired.get((dataset, ips_model, random_state), {})
                if "frequency-based" in pair:
                    base_values.append(pair["frequency-based"])

            for prop in PROP_ORDER:
                prop_values = values.get((dataset, prop, ips_model), [])
                if not prop_values:
                    continue

                mean_value = sum(prop_values) / len(prop_values)
                std_value = sample_std(prop_values)
                is_best_non_oracle = prop == best_prop

                if prop == "frequency-based":
                    cell = format_summary_cell(mean_value, std_value, "", bold=is_best_non_oracle)
                else:
                    matched_values = []
                    matched_base_values = []
                    for random_state in random_states:
                        pair = paired.get((dataset, ips_model, random_state), {})
                        if prop in pair and "frequency-based" in pair:
                            matched_values.append(pair[prop])
                            matched_base_values.append(pair["frequency-based"])

                    valid_pairs = get_valid_pairs(matched_values, matched_base_values)
                    p_value = paired_test_pvalue(
                        [x for x, _ in valid_pairs],
                        [y for _, y in valid_pairs],
                    )
                    base_mean = sum(base_values) / len(base_values) if base_values else mean_value
                    symbol = get_sig_symbol(mean_value, base_mean, p_value)
                    cell = format_summary_cell(mean_value, std_value, symbol, bold=is_best_non_oracle)
                    if base_values and len(valid_pairs) < 2:
                        diagnostics.append(
                            {
                                "dataset": dataset,
                                "ips_model": ips_model,
                                "propensity_model": prop,
                                "matched_pairs": len(valid_pairs),
                            }
                        )

                table_data[dataset][prop][ips_model] = cell

    return table_data, diagnostics


def render_table(table_data, datasets, output_prefix, n_sessions, temperature):
    available_datasets = [(ds_key, ds_label) for ds_key, ds_label in datasets if ds_key in table_data]
    col_spec = "llccc"
    header = " & ".join(["\\textbf{Dataset}", "\\textbf{Model}"] + [f"\\textbf{{{IPS_MODEL_NAMES[m]}}}" for m in IPS_MODELS])

    lines = []
    lines.append("\\begin{table}[h!]")
    lines.append(
        "  \\caption{Estimated position-bias results for feature-based propensity estimation. "
        f"Runs use $n={n_sessions:,}$ sessions and policy temperature $T={temperature:g}$. "
        "Columns report objectives. Cells show mean (SD) NDCG. The oracle propensity baseline is included, "
        "and bold marks the best non-oracle value within each dataset-objective block. Significance markers compare "
        "each non-frequency-based method with the frequency-based method using a two-sided Student "
        "t-test: $p < 0.01$ ($\\,^\\blacktriangle \\text{ and } ^\\blacktriangledown$), "
        "$p < 0.05$ ($\\,^\\triangle \\text{ and } ^\\triangledown$).}"
    )
    lines.append("  \\centering")
    lines.append("  \\resizebox{0.85\\textwidth}{!}{")
    lines.append(f"    \\begin{{tabular}}{{{col_spec}}}")
    lines.append("      \\toprule")
    lines.append(f"      {header} \\\\")
    lines.append("      \\midrule")

    last_col = 2 + len(IPS_MODELS)
    for ds_idx, (dataset_key, dataset_label) in enumerate(available_datasets):
        lines.append(f"      \\multirow{{{len(PROP_ORDER)}}}{{*}}{{\\textbf{{{dataset_label}}}}}")
        for prop in PROP_ORDER:
            row = f"      & {PROP_NAMES[prop]}"
            for ips_model in IPS_MODELS:
                row += f" & {table_data[dataset_key][prop].get(ips_model, '-')}"
            row += " \\\\"
            lines.append(row)
            if prop == "true_propensity":
                lines.append(f"      \\cmidrule(lr){{2-{last_col}}}")
        if ds_idx != len(available_datasets) - 1:
            lines.append("      \\midrule")

    lines.append("      \\bottomrule")
    lines.append("    \\end{tabular}")
    lines.append("  }")
    lines.append(f"  \\label{{tab:{output_prefix}_objective_comparison}}")
    lines.append("\\end{table}")
    return "\n".join(lines)


def print_diagnostics(table_name, diagnostics):
    if not diagnostics:
        print(f"{table_name}: all significance comparisons had at least two matched pairs.")
        return

    print(f"{table_name}: significance not computed for the following cells due to fewer than two matched pairs:")
    for item in diagnostics:
        print(
            "  - "
            f"Dataset={item['dataset']}, "
            f"Objective={item['ips_model'].upper()}, "
            f"Propensity={PROP_NAMES[item['propensity_model']]}, "
            f"matched_pairs={item['matched_pairs']}"
        )


def main():
    args = parse_args()
    datasets = parse_dataset_labels(args.datasets)
    random_states = parse_int_list(args.random_states)

    records = load_records(
        experiment=args.experiment,
        datasets=datasets,
        n_sessions=args.n_sessions,
        temperature=args.temperature,
        random_states=random_states,
    )
    if not records:
        raise RuntimeError("No matching records found for the requested configuration.")

    table_data, diagnostics = build_table_data(records, datasets, random_states)
    table_str = render_table(
        table_data=table_data,
        datasets=datasets,
        output_prefix=args.output_prefix,
        n_sessions=args.n_sessions,
        temperature=args.temperature,
    )

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    THESIS_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    filename = TABLES_DIR / f"tabel_{args.output_prefix}_objective_comparison.txt"
    thesis_filename = THESIS_TABLES_DIR / filename.name
    filename.write_text(table_str)
    thesis_filename.write_text(table_str)
    print(f"Wrote {filename}")
    print_diagnostics(filename.name, diagnostics)


if __name__ == "__main__":
    main()
