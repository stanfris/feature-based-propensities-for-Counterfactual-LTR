import argparse
from collections import defaultdict
from pathlib import Path

from generate_latex_table import (
    METRICS,
    PROP_NAMES,
    TABLES_DIR,
    format_n_sessions,
    parse_dataset_labels,
    parse_folder_name,
    parse_number_list,
    safe_load_json,
)
from table_significance import format_summary_cell, get_sig_symbol, get_valid_pairs, paired_test_pvalue, sample_std


WANTED_MODELS = {"ips", "dm", "dr"}
WANTED_PROPS = {
    "frequency-based",
    "MLPregression",
    "true_propensity",
    "cosine",
    "knn",
    "kmeans",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate LaTeX temperature tables from real_targets results."
    )
    parser.add_argument("--experiment", default="real_targets", help="Experiment folder under results/.")
    parser.add_argument(
        "--n-sessions",
        type=int,
        default=10000,
        help="Fixed session count to include in the temperature comparison.",
    )
    parser.add_argument(
        "--datasets",
        default="mslr30k:MSLR-WEB30K,istella:Istella-S,yahoo:Yahoo!",
        help="Comma-separated dataset_key:Dataset Label pairs.",
    )
    parser.add_argument(
        "--random-states",
        default="40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59",
        help="Comma-separated random states to include.",
    )
    parser.add_argument(
        "--temperatures",
        default="0,0.25,0.5,0.75,1.0",
        help="Comma-separated policy temperatures to include.",
    )
    parser.add_argument(
        "--output-prefix",
        default="real_targets_temperature",
        help="Prefix used in generated table filenames.",
    )
    return parser.parse_args()


def load_records(experiment, datasets, n_sessions, random_states, temperatures):
    base_dir = Path("results") / experiment
    if not base_dir.exists():
        raise FileNotFoundError(f"Results folder not found: {base_dir}")

    wanted_datasets = {key for key, _ in datasets}
    wanted_random_states = set(random_states)
    wanted_temperatures = set(temperatures)

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

        if dataset not in wanted_datasets or ips_model not in WANTED_MODELS:
            continue
        if propensity_model not in WANTED_PROPS:
            continue
        if raw_n_sessions is None or int(raw_n_sessions) != n_sessions:
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


def build_table_data(records, ips_model, prop_order, temperatures, random_states):
    values = defaultdict(list)
    paired = defaultdict(dict)

    for record in records:
        if record["ips_model"] != ips_model:
            continue

        value_key = (
            record["dataset"],
            record["propensity_model"],
            record["tmp"],
            record["metric"],
        )
        values[value_key].append(record["value"])

        pair_key = (
            record["dataset"],
            record["tmp"],
            record["metric"],
            record["random_state"],
        )
        paired[pair_key][record["propensity_model"]] = record["value"]

    table_data = defaultdict(lambda: {metric: {tmp: {} for tmp in temperatures} for metric in METRICS})
    diagnostics = []
    best_prop_map = {}

    summary_keys = {(dataset, tmp, metric) for dataset, tmp, metric, _ in paired.keys()}
    for dataset_key, temperature, metric in summary_keys:
        if temperature not in temperatures or metric not in METRICS:
            continue

        base_values = []
        for random_state in random_states:
            pair = paired.get((dataset_key, temperature, metric, random_state), {})
            if "frequency-based" in pair:
                base_values.append(pair["frequency-based"])

        candidate_props = [
            prop
            for prop in prop_order
            if prop != "true_propensity" and values.get((dataset_key, prop, temperature, metric), [])
        ]
        best_prop_map[(dataset_key, temperature, metric)] = (
            max(
                candidate_props,
                key=lambda prop: (
                    sum(values[(dataset_key, prop, temperature, metric)])
                    / len(values[(dataset_key, prop, temperature, metric)]),
                    -prop_order.index(prop),
                ),
            )
            if candidate_props
            else None
        )

        for prop in prop_order:
            prop_values = values.get((dataset_key, prop, temperature, metric), [])
            if not prop_values:
                continue

            mean_value = sum(prop_values) / len(prop_values)
            std_value = sample_std(prop_values)
            is_best_non_oracle = prop == best_prop_map[(dataset_key, temperature, metric)]

            if prop == "frequency-based":
                cell = format_summary_cell(mean_value, std_value, "", bold=is_best_non_oracle)
            else:
                matched_values = []
                matched_base_values = []
                for random_state in random_states:
                    pair = paired.get((dataset_key, temperature, metric, random_state), {})
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
                            "temperature": temperature,
                            "metric": metric,
                            "matched_pairs": matched_count,
                        }
                    )

            table_data[dataset_key][metric][temperature][prop] = cell

    return table_data, diagnostics


def format_temperature_label(temperature):
    if abs(temperature - 0.0) < 1e-12:
        return "0"
    if abs(temperature - 0.25) < 1e-12 or abs(temperature - 0.75) < 1e-12:
        return f"{temperature:.2f}"
    if abs(temperature - 0.5) < 1e-12 or abs(temperature - 1.0) < 1e-12:
        return f"{temperature:.1f}"
    return f"{temperature:.2f}".rstrip("0").rstrip(".")


def render_combined_table(
    ips_model,
    include_distance,
    dataset_tables,
    datasets,
    temperatures,
    output_prefix,
    n_sessions,
):
    if include_distance:
        prop_order = ["true_propensity", "frequency-based", "MLPregression", "cosine", "knn", "kmeans"]
        toggle_str = "included"
    else:
        prop_order = ["true_propensity", "frequency-based", "MLPregression"]
        toggle_str = "excluded"
    col_spec = "ll" + "cc" * len(temperatures)
    cmidrules = "".join(
        f"\\cmidrule(lr){{{3 + 2 * idx}-{4 + 2 * idx}}}" for idx in range(len(temperatures))
    )
    header_groups = " ".join(
        f"& \\multicolumn{{2}}{{c}}{{\\textbf{{{format_temperature_label(temp)}}}}}"
        for temp in temperatures
    )
    metric_header = " & ".join(["\\textbf{Dataset}", "\\textbf{Model}"] + ["NDCG", "RCTR"] * len(temperatures))

    lines = []
    lines.append("\\begin{table}[h!]")
    lines.append(
        f"  \\caption{{{ips_model.upper()} results for feature-based propensity estimation on "
        "\\texttt{real\\_targets} across policy temperatures. Columns are grouped by "
        "\\textbf{temperature}, with two \\textbf{metrics} (NDCG, RCTR) reported for each. "
        "The oracle propensity baseline appears first for each dataset, and bold marks the best "
        "non-oracle value within each dataset-temperature-metric block. Cells show mean (SD). "
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
    last_col = 2 + 2 * len(temperatures)
    for ds_idx, (ds_key, ds_label) in enumerate(available_datasets):
        lines.append(f"      \\multirow{{{len(prop_order)}}}{{*}}{{\\textbf{{{ds_label}}}}}")
        for prop in prop_order:
            row = f"      & {PROP_NAMES[prop]}"
            for temperature in temperatures:
                for metric in METRICS:
                    row += f" & {dataset_tables[ds_key][metric][temperature].get(prop, '-')}"
            row += " \\\\"
            lines.append(row)
            if prop == "true_propensity":
                lines.append(f"      \\cmidrule(lr){{2-{last_col}}}")
        if ds_idx != len(available_datasets) - 1:
            lines.append("      \\midrule")

    lines.append("      \\bottomrule")
    lines.append("  \\end{tabular}}")
    lines.append(f"  \\label{{tab:{ips_model}_temperature_means_{output_prefix}_{toggle_str}}}")
    lines.append("\\end{table}")
    return "\n".join(lines), toggle_str


def render_column_table(
    ips_model,
    include_distance,
    dataset_key,
    dataset_label,
    dataset_table,
    temperatures,
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
        if any(dataset_table["NDCG"][temperature].get(prop) for temperature in temperatures)
    ]

    col_spec = "l" + "c" * len(temperatures)
    header = " & ".join(
        ["\\textbf{Model}"] + [f"\\textbf{{{format_temperature_label(temp)}}}" for temp in temperatures]
    )

    lines = []
    lines.append("\\begin{table}[h!]")
    lines.append(
        f"  \\caption{{{ips_model.upper()} results for feature-based propensity estimation on "
        f"{dataset_label} across policy temperatures. Columns report policy temperatures. Cells show "
        "mean (SD) NDCG. The oracle propensity baseline is included, "
        "and bold marks the best non-oracle value within each temperature block. Significance markers "
        "compare each non-frequency-based method with the frequency-based method using a two-sided "
        "Student t-test: $p < 0.01$ ($\\,^\\blacktriangle \\text{ and } ^\\blacktriangledown$), "
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
        for temperature in temperatures:
            row += f" & {dataset_table['NDCG'][temperature].get(prop, '-')}"
        row += " \\\\"
        lines.append(row)
        if prop == "true_propensity":
            lines.append(f"      \\cmidrule(lr){{1-{1 + len(temperatures)}}}")

    lines.append("      \\bottomrule")
    lines.append("    \\end{tabular}")
    lines.append("  }")
    lines.append(
        f"  \\label{{tab:{ips_model}_temperature_means_{output_prefix}_{dataset_key}_{toggle_str}_columns}}"
    )
    lines.append("\\end{table}")
    return "\n".join(lines), toggle_str


def render_all_datasets_column_table(
    ips_model,
    include_distance,
    dataset_tables,
    datasets,
    temperatures,
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
            dataset_tables[ds_key]["NDCG"][temperature].get(prop)
            for ds_key, _ in available_datasets
            for temperature in temperatures
        )
    ]

    col_spec = "ll" + "c" * len(temperatures)
    header = " & ".join(
        ["\\textbf{Dataset}", "\\textbf{Model}"]
        + [f"\\textbf{{{format_temperature_label(temp)}}}" for temp in temperatures]
    )

    lines = []
    lines.append("\\begin{table}[h!]")
    lines.append(
        f"  \\caption{{{ips_model.upper()} results for feature-based propensity estimation across policy temperatures. "
        "Columns report policy temperatures. Cells show mean (SD) NDCG. The oracle propensity baseline is included, "
        "and bold marks the best non-oracle value within each dataset-temperature block. Significance markers compare "
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

    last_col = 2 + len(temperatures)
    for ds_idx, (ds_key, ds_label) in enumerate(available_datasets):
        lines.append(f"      \\multirow{{{len(prop_order)}}}{{*}}{{\\textbf{{{ds_label}}}}}")
        for prop in prop_order:
            row = f"      & {PROP_NAMES[prop]}"
            for temperature in temperatures:
                row += f" & {dataset_tables[ds_key]['NDCG'][temperature].get(prop, '-')}"
            row += " \\\\"
            lines.append(row)
            if prop == "true_propensity":
                lines.append(f"      \\cmidrule(lr){{2-{last_col}}}")
        if ds_idx != len(available_datasets) - 1:
            lines.append("      \\midrule")

    lines.append("      \\bottomrule")
    lines.append("    \\end{tabular}")
    lines.append("  }")
    lines.append(f"  \\label{{tab:{ips_model}_temperature_means_{output_prefix}_{toggle_str}_columns}}")
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
            f"Temperature={item['temperature']}, "
            f"Metric={item['metric']}, "
            f"matched_pairs={item['matched_pairs']}"
        )


def preserve_existing_numeric_cells(table_str, *paths):
    return table_str


def main():
    args = parse_args()
    datasets = parse_dataset_labels(args.datasets)
    temperatures = parse_number_list(args.temperatures, float)

    records = load_records(
        experiment=args.experiment,
        datasets=datasets,
        n_sessions=args.n_sessions,
        random_states=parse_number_list(args.random_states, int) if args.random_states.strip() else tuple(range(0, 10_000)),
        temperatures=temperatures,
    )
    if not records:
        raise RuntimeError("No matching records found for the requested configuration.")

    if args.random_states.strip():
        random_states = parse_number_list(args.random_states, int)
    else:
        random_states = tuple(sorted({record["random_state"] for record in records}))

    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    for ips_model in ("ips", "dr", "dm"):
        for include_distance in (False,):
            prop_order = (
                ["true_propensity", "frequency-based", "MLPregression", "cosine", "knn", "kmeans"]
                if include_distance
                else ["true_propensity", "frequency-based", "MLPregression"]
            )
            dataset_tables, diagnostics = build_table_data(
                records=records,
                ips_model=ips_model,
                prop_order=prop_order,
                temperatures=temperatures,
                random_states=random_states,
            )
            table_str, toggle_str = render_combined_table(
                ips_model=ips_model,
                include_distance=include_distance,
                dataset_tables=dataset_tables,
                datasets=datasets,
                temperatures=temperatures,
                output_prefix=args.output_prefix,
                n_sessions=args.n_sessions,
            )
            filename = TABLES_DIR / f"tabel_{args.output_prefix}_{ips_model}_{toggle_str}.txt"
            table_str = preserve_existing_numeric_cells(table_str, filename)
            with filename.open("w") as f:
                f.write(table_str)
            print(f"Wrote {filename}")
            print_diagnostics(filename.name, diagnostics)

            column_table_str, _ = render_all_datasets_column_table(
                ips_model=ips_model,
                include_distance=include_distance,
                dataset_tables=dataset_tables,
                datasets=datasets,
                temperatures=temperatures,
                output_prefix=args.output_prefix,
            )
            column_filename = TABLES_DIR / f"tabel_{args.output_prefix}_{ips_model}_{toggle_str}_columns.txt"
            column_table_str = preserve_existing_numeric_cells(
                column_table_str,
                column_filename,
            )
            column_filename.write_text(column_table_str)
            print(f"Wrote {column_filename}")


if __name__ == "__main__":
    main()
