import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

from table_significance import format_summary_cell, get_sig_symbol, get_valid_pairs, paired_test_pvalue, sample_std


current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
os.chdir(project_root)


METRICS = ("NDCG", "RCTR")
DEFAULT_DATASETS = (("mslr30k", "MSLR-WEB30K"), ("istella", "Istella-S"))
TABLES_DIR = Path("notebooks/thesis_tables")
THESIS_TABLES_DIR = Path(__file__).resolve().parents[3] / "Thesis" / "tables"
PROP_NAMES = {
    "frequency-based": "Frequency-Based",
    "MLPregression": "MLP",
    "cosine": "Cosine-Sim",
    "knn": "KNN",
    "kmeans": "K-means",
    "true_propensity": "Oracle Propensity",
}


def format_n_sessions(n_sessions):
    exponent_map = {
        1_000: 3,
        10_000: 4,
        100_000: 5,
        1_000_000: 6,
    }
    if n_sessions in exponent_map:
        return rf"\sampleN{{{exponent_map[n_sessions]}}}"
    return f"{n_sessions:,}"


def format_n_sessions_header(n_sessions):
    exponent_map = {
        1_000: 3,
        10_000: 4,
        100_000: 5,
        1_000_000: 6,
    }
    if n_sessions in exponent_map:
        return rf"\sampleNheader{{{exponent_map[n_sessions]}}}"
    return f"\\textbf{{{format_n_sessions(n_sessions)}}}"


def parse_args():
    parser = argparse.ArgumentParser(description="Generate thesis LaTeX tables from experiment results.")
    parser.add_argument("--experiment", default="mul-two-tower", help="Experiment folder under results/.")
    parser.add_argument(
        "--n-list",
        default="1000,10000,100000,1000000",
        help="Comma-separated session counts to include.",
    )
    parser.add_argument(
        "--datasets",
        default="mslr30k:MSLR-WEB30K,istella:Istella-S",
        help="Comma-separated dataset_key:Dataset Label pairs.",
    )
    parser.add_argument(
        "--random-states",
        default="40,41,42,43,44,45,46,47,48,49",
        help="Comma-separated random states to include.",
    )
    parser.add_argument(
        "--temperatures",
        default="0.5",
        help="Comma-separated policy temperatures to include.",
    )
    parser.add_argument(
        "--output-prefix",
        default="dual",
        help="Prefix used in generated thesis_tables filenames.",
    )
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


def parse_number_list(spec, cast):
    return tuple(cast(part.strip()) for part in spec.split(",") if part.strip())


def safe_load_json(path):
    try:
        with path.open("r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def load_records(experiment, datasets, n_list, random_states, temperatures):
    base_dir = Path("results") / experiment
    if not base_dir.exists():
        raise FileNotFoundError(f"Results folder not found: {base_dir}")

    wanted_datasets = {key for key, _ in datasets}
    wanted_n = set(n_list)
    wanted_random_states = set(random_states)
    wanted_temperatures = set(temperatures)
    wanted_models = {"ips", "dm", "dr"}
    wanted_props = {
        "frequency-based",
        "MLPregression",
        "true_propensity",
        "cosine",
        "knn",
        "kmeans",
    }

    records = []
    for path in base_dir.rglob("ips_results.json"):
        parsed = parse_folder_name(path.parent.name)
        dataset = parsed.get("data")
        ips_model = parsed.get("ips.model")
        propensity_model = parsed.get("propensity_model")
        raw_n_sessions = parsed.get("ips.n_sessions")
        raw_random_state = parsed.get("random_state")
        raw_temperature = parsed.get("policy_temperature")
        filt_str = parsed.get("ips.filter_single_display_pairs", "False")

        if dataset not in wanted_datasets or ips_model not in wanted_models:
            continue
        if propensity_model not in wanted_props:
            continue
        if raw_n_sessions is None or int(raw_n_sessions) not in wanted_n:
            continue
        if raw_random_state is None or int(raw_random_state) not in wanted_random_states:
            continue
        if raw_temperature is None or float(raw_temperature) not in wanted_temperatures:
            continue

        plot_filter = "merged" if ips_model == "ips" else (filt_str.lower() == "true")
        if ips_model == "ips" and plot_filter != "merged":
            continue
        if ips_model in {"dm", "dr"} and plot_filter is not False:
            continue

        payload = safe_load_json(path)
        if not payload:
            continue
        metrics = payload.get("results", {}).get("metrics", {})
        for metric in METRICS:
            if metric not in metrics:
                continue
            records.append(
                {
                    "dataset": dataset,
                    "ips_model": ips_model,
                    "propensity_model": propensity_model,
                    "n_sessions": int(raw_n_sessions),
                    "random_state": int(raw_random_state),
                    "tmp": float(raw_temperature),
                    "metric": metric,
                    "value": float(metrics[metric]),
                }
            )
    return records


def build_table_data(records, ips_model, prop_order, n_list, temperatures, random_states):
    values = defaultdict(list)
    paired = defaultdict(dict)

    for record in records:
        if record["ips_model"] != ips_model:
            continue
        key = (
            record["dataset"],
            record["propensity_model"],
            record["n_sessions"],
            record["metric"],
        )
        values[key].append(record["value"])

        pair_key = (
            record["dataset"],
            record["n_sessions"],
            record["metric"],
            record["random_state"],
            record["tmp"],
        )
        paired[pair_key][record["propensity_model"]] = record["value"]

    table_data = defaultdict(lambda: {metric: {n: {} for n in n_list} for metric in METRICS})
    diagnostics = []
    best_prop_map = {}

    summary_keys = {(dataset_key, n_sessions, metric) for dataset_key, n_sessions, metric, _, _ in paired.keys()}
    for dataset_key, n_sessions, metric in summary_keys:
        if n_sessions not in n_list or metric not in METRICS:
            continue
        base_values = []
        for random_state in random_states:
            for tmp in temperatures:
                pair = paired.get((dataset_key, n_sessions, metric, random_state, tmp), {})
                if "frequency-based" in pair:
                    base_values.append(pair["frequency-based"])

        candidate_props = [
            prop
            for prop in prop_order
            if prop != "true_propensity" and values.get((dataset_key, prop, n_sessions, metric), [])
        ]
        best_prop_map[(dataset_key, n_sessions, metric)] = (
            max(
                candidate_props,
                key=lambda prop: (
                    sum(values[(dataset_key, prop, n_sessions, metric)])
                    / len(values[(dataset_key, prop, n_sessions, metric)]),
                    -prop_order.index(prop),
                ),
            )
            if candidate_props
            else None
        )

        for prop in prop_order:
            prop_values = values.get((dataset_key, prop, n_sessions, metric), [])
            if not prop_values:
                continue

            mean_value = sum(prop_values) / len(prop_values)
            std_value = sample_std(prop_values)
            is_best_non_oracle = prop == best_prop_map[(dataset_key, n_sessions, metric)]

            if prop == "frequency-based":
                cell = format_summary_cell(mean_value, std_value, "", bold=is_best_non_oracle)
            else:
                matched_values = []
                matched_base_values = []
                for random_state in random_states:
                    for tmp in temperatures:
                        pair = paired.get((dataset_key, n_sessions, metric, random_state, tmp), {})
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
                matched_count = len(valid_pairs)
                cell = format_summary_cell(mean_value, std_value, symbol, bold=is_best_non_oracle)
                if base_values and matched_count < 2:
                    diagnostics.append(
                        {
                            "dataset": dataset_key,
                            "ips_model": ips_model,
                            "propensity_model": prop,
                            "n_sessions": n_sessions,
                            "metric": metric,
                            "matched_pairs": matched_count,
                        }
                    )

            table_data[dataset_key][metric][n_sessions][prop] = cell

    return table_data, diagnostics


def render_combined_table(ips_model, include_distance, dataset_tables, datasets, n_list, output_prefix):
    if include_distance:
        prop_order = ["true_propensity", "frequency-based", "MLPregression", "cosine", "knn", "kmeans"]
        toggle_str = "included"
    else:
        prop_order = ["true_propensity", "frequency-based", "MLPregression"]
        toggle_str = "excluded"
    col_spec = "ll" + "cc" * len(n_list)
    cmidrules = "".join(
        f"\\cmidrule(lr){{{3 + 2 * idx}-{4 + 2 * idx}}}" for idx in range(len(n_list))
    )
    header_groups = " ".join(
        f"& \\multicolumn{{2}}{{c}}{{{format_n_sessions_header(n)}}}" for n in n_list
    )
    metric_header = " & ".join(["\\textbf{Dataset}", "\\textbf{Model}"] + ["NDCG", "RCTR"] * len(n_list))

    lines = []
    lines.append("\\begin{table}[h!]")
    lines.append(
        f"  \\caption{{{ips_model.upper()} results for feature-based propensity estimation across sample counts. "
        "Columns are grouped by \\textbf{sample count}, with two \\textbf{metrics} "
        "(NDCG, RCTR) reported for each. The oracle propensity baseline appears first for each dataset, and "
        "bold marks the best non-oracle value within each dataset-sample-metric block. Cells show mean (SD). "
        "Significance markers compare each non-frequency-based method with the frequency-based method "
        "using a two-sided Student t-test: $p < 0.01$ ($\\,^\\blacktriangle \\text{ and } ^\\blacktriangledown$), "
        "$p < 0.05$ ($\\,^\\triangle \\text{ and } ^\\triangledown$).}"
    )
    lines.append("  \\centering")
    lines.append("  \\setlength{\\tabcolsep}{4pt}")
    lines.append("  \\resizebox{\\textwidth}{!}{")
    lines.append(f"    \\begin{{tabular}}{{{col_spec}}}")
    lines.append("      \\toprule")
    lines.append(f"      & {header_groups} \\\\")
    lines.append(f"      {cmidrules}")
    lines.append(f"      {metric_header} \\\\")
    lines.append("      \\midrule")

    available_datasets = [(ds_key, ds_label) for ds_key, ds_label in datasets if ds_key in dataset_tables]
    last_col = 2 + 2 * len(n_list)
    for ds_idx, (ds_key, ds_label) in enumerate(available_datasets):
        lines.append(f"      \\multirow{{{len(prop_order)}}}{{*}}{{\\textbf{{{ds_label}}}}}")
        for prop in prop_order:
            row = f"      & {PROP_NAMES[prop]}"
            for n_sessions in n_list:
                for metric in METRICS:
                    row += f" & {dataset_tables[ds_key][metric][n_sessions].get(prop, '-')}"
            row += " \\\\"
            lines.append(row)
            if prop == "true_propensity":
                lines.append(f"      \\cmidrule(lr){{2-{last_col}}}")
        if ds_idx != len(available_datasets) - 1:
            lines.append("      \\midrule")

    lines.append("      \\bottomrule")
    lines.append("  \\end{tabular}}")
    lines.append(f"  \\label{{tab:{ips_model}_means_{output_prefix}_{toggle_str}}}")
    lines.append("\\end{table}")
    return "\n".join(lines), toggle_str


def render_column_table(
    ips_model,
    include_distance,
    dataset_key,
    dataset_label,
    dataset_table,
    n_list,
    output_prefix,
):
    if include_distance:
        prop_order = ["true_propensity", "frequency-based", "MLPregression", "cosine", "knn", "kmeans"]
        toggle_str = "included"
    else:
        prop_order = ["true_propensity", "frequency-based", "MLPregression"]
        toggle_str = "excluded"
    prop_order = [
        prop
        for prop in prop_order
        if any(dataset_table["NDCG"][n_sessions].get(prop) for n_sessions in n_list)
    ]

    col_spec = "l" + "c" * len(n_list)
    header = " & ".join(["\\textbf{Model}"] + [format_n_sessions_header(n) for n in n_list])

    lines = []
    lines.append("\\begin{table}[h!]")
    lines.append(
        f"  \\caption{{{ips_model.upper()} results for feature-based propensity estimation on "
        f"{dataset_label} across sample counts. Columns report sample counts. Cells show mean (SD) "
        "NDCG. The oracle propensity baseline is included, and bold "
        "marks the best non-oracle value within each sample-count block. Significance markers compare "
        "each non-frequency-based method with the frequency-based method using a two-sided Student "
        "t-test: $p < 0.01$ ($\\,^\\blacktriangle \\text{ and } ^\\blacktriangledown$), "
        "$p < 0.05$ ($\\,^\\triangle \\text{ and } ^\\triangledown$).}"
    )
    lines.append("  \\centering")
    lines.append("  \\resizebox{0.8\\textwidth}{!}{")
    lines.append(f"    \\begin{{tabular}}{{{col_spec}}}")
    lines.append("      \\toprule")
    lines.append(f"      {header} \\\\")
    lines.append("      \\midrule")

    for prop in prop_order:
        row = f"      {PROP_NAMES[prop]}"
        for n_sessions in n_list:
            row += f" & {dataset_table['NDCG'][n_sessions].get(prop, '-')}"
        row += " \\\\"
        lines.append(row)
        if prop == "true_propensity":
            lines.append(f"      \\cmidrule(lr){{1-{1 + len(n_list)}}}")

    lines.append("      \\bottomrule")
    lines.append("    \\end{tabular}")
    lines.append("  }")
    lines.append(f"  \\label{{tab:{ips_model}_means_{output_prefix}_{dataset_key}_{toggle_str}_columns}}")
    lines.append("\\end{table}")
    return "\n".join(lines), toggle_str


def render_all_datasets_column_table(
    ips_model,
    include_distance,
    dataset_tables,
    datasets,
    n_list,
    output_prefix,
):
    if include_distance:
        prop_order = ["true_propensity", "frequency-based", "MLPregression", "cosine", "knn", "kmeans"]
        toggle_str = "included"
    else:
        prop_order = ["true_propensity", "frequency-based", "MLPregression"]
        toggle_str = "excluded"

    available_datasets = [(ds_key, ds_label) for ds_key, ds_label in datasets if ds_key in dataset_tables]
    prop_order = [
        prop
        for prop in prop_order
        if any(
            dataset_tables[ds_key]["NDCG"][n_sessions].get(prop)
            for ds_key, _ in available_datasets
            for n_sessions in n_list
        )
    ]

    col_spec = "ll" + "c" * len(n_list)
    header = " & ".join(["\\textbf{Dataset}", "\\textbf{Model}"] + [format_n_sessions_header(n) for n in n_list])

    lines = []
    lines.append("\\begin{table}[h!]")
    lines.append(
        f"  \\caption{{{ips_model.upper()} results for feature-based propensity estimation across sample counts. "
        "Columns report sample counts. Cells show mean (SD) NDCG. The oracle propensity baseline is included, "
        "and bold marks the best non-oracle value within each dataset-sample block. Significance markers compare "
        "each non-frequency-based method with the frequency-based method using a two-sided Student "
        "t-test: $p < 0.01$ ($\\,^\\blacktriangle \\text{ and } ^\\blacktriangledown$), "
        "$p < 0.05$ ($\\,^\\triangle \\text{ and } ^\\triangledown$).}"
    )
    lines.append("  \\centering")
    lines.append("  \\resizebox{\\textwidth}{!}{")
    lines.append(f"    \\begin{{tabular}}{{{col_spec}}}")
    lines.append("      \\toprule")
    lines.append(f"      {header} \\\\")
    lines.append("      \\midrule")

    last_col = 2 + len(n_list)
    for ds_idx, (ds_key, ds_label) in enumerate(available_datasets):
        lines.append(f"      \\multirow{{{len(prop_order)}}}{{*}}{{\\textbf{{{ds_label}}}}}")
        for prop in prop_order:
            row = f"      & {PROP_NAMES[prop]}"
            for n_sessions in n_list:
                row += f" & {dataset_tables[ds_key]['NDCG'][n_sessions].get(prop, '-')}"
            row += " \\\\"
            lines.append(row)
            if prop == "true_propensity":
                lines.append(f"      \\cmidrule(lr){{2-{last_col}}}")
        if ds_idx != len(available_datasets) - 1:
            lines.append("      \\midrule")

    lines.append("      \\bottomrule")
    lines.append("    \\end{tabular}")
    lines.append("  }")
    lines.append(f"  \\label{{tab:{ips_model}_means_{output_prefix}_{toggle_str}_columns}}")
    lines.append("\\end{table}")
    return "\n".join(lines), toggle_str


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
            f"Sessions={item['n_sessions']}, "
            f"Metric={item['metric']}, "
            f"matched_pairs={item['matched_pairs']}"
        )


def preserve_existing_numeric_cells(table_str, *paths):
    return table_str


def main():
    args = parse_args()
    n_list = parse_number_list(args.n_list, int)
    datasets = parse_dataset_labels(args.datasets)
    random_states = parse_number_list(args.random_states, int)
    temperatures = parse_number_list(args.temperatures, float)

    records = load_records(
        experiment=args.experiment,
        datasets=datasets,
        n_list=n_list,
        random_states=random_states,
        temperatures=temperatures,
    )
    if not records:
        raise RuntimeError("No matching records found for the requested configuration.")

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    THESIS_TABLES_DIR.mkdir(parents=True, exist_ok=True)

    for ips_model in ("ips", "dr", "dm"):
        for include_distance in (False, True):
            prop_order = (
                ["true_propensity", "frequency-based", "MLPregression", "cosine", "knn", "kmeans"]
                if include_distance
                else ["true_propensity", "frequency-based", "MLPregression"]
            )
            dataset_tables, diagnostics = build_table_data(
                records=records,
                ips_model=ips_model,
                prop_order=prop_order,
                n_list=n_list,
                temperatures=temperatures,
                random_states=random_states,
            )
            table_str, toggle_str = render_combined_table(
                ips_model=ips_model,
                include_distance=include_distance,
                dataset_tables=dataset_tables,
                datasets=datasets,
                n_list=n_list,
                output_prefix=args.output_prefix,
            )
            filename = TABLES_DIR / f"tabel_{args.output_prefix}_{ips_model}_{toggle_str}.txt"
            thesis_filename = THESIS_TABLES_DIR / filename.name
            table_str = preserve_existing_numeric_cells(table_str, filename, thesis_filename)
            with filename.open("w") as f:
                f.write(table_str)
            thesis_filename.write_text(table_str)
            print(f"Wrote {filename}")
            print_diagnostics(filename.name, diagnostics)

            column_table_str, _ = render_all_datasets_column_table(
                ips_model=ips_model,
                include_distance=include_distance,
                dataset_tables=dataset_tables,
                datasets=datasets,
                n_list=n_list,
                output_prefix=args.output_prefix,
            )
            column_filename = TABLES_DIR / f"tabel_{args.output_prefix}_{ips_model}_{toggle_str}_columns.txt"
            column_thesis_filename = THESIS_TABLES_DIR / column_filename.name
            column_table_str = preserve_existing_numeric_cells(
                column_table_str,
                column_filename,
                column_thesis_filename,
            )
            column_filename.write_text(column_table_str)
            column_thesis_filename.write_text(column_table_str)
            print(f"Wrote {column_filename}")

            for dataset_key, dataset_label in datasets:
                if dataset_key not in dataset_tables:
                    continue
                column_table_str, _ = render_column_table(
                    ips_model=ips_model,
                    include_distance=include_distance,
                    dataset_key=dataset_key,
                    dataset_label=dataset_label,
                    dataset_table=dataset_tables[dataset_key],
                    n_list=n_list,
                    output_prefix=args.output_prefix,
                )
                column_filename = (
                    TABLES_DIR
                    / f"tabel_{args.output_prefix}_{dataset_key}_{ips_model}_{toggle_str}_columns.txt"
                )
                column_thesis_filename = THESIS_TABLES_DIR / column_filename.name
                column_table_str = preserve_existing_numeric_cells(
                    column_table_str,
                    column_filename,
                    column_thesis_filename,
                )
                column_filename.write_text(column_table_str)
                column_thesis_filename.write_text(column_table_str)
                print(f"Wrote {column_filename}")


if __name__ == "__main__":
    main()
