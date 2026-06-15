import os
from pathlib import Path

from generate_two_tower_bias_comparison_table import (
    load_records,
    parse_random_states,
)
from table_significance import get_sig_symbol, paired_test_pvalue, sample_std


current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
os.chdir(project_root)


NOTEBOOK_TABLES_DIR = Path("notebooks/thesis_tables")
THESIS_TABLES_DIR = Path("..") / "Thesis" / "tables"
OUTPUT_NAME = "tabel_two_tower_bias_mlp_minus_frequency_all_datasets.txt"
TABLE_LABEL = "tab:two_tower_bias_mlp_minus_frequency_all_datasets"
DATASET_ORDER = (
    ("mslr30k", r"\DatasetMSLR"),
    ("yahoo", r"\DatasetYahoo"),
    ("istella", r"\DatasetIstella"),
)
OBJECTIVE_ORDER = ("ips", "dm", "dr")
SESSION_ORDER = (10000, 100000)
SOURCE_ORDER = (
    "real-targets",
    "add-two-tower",
    "mul-two-tower",
    "pb-adjacent_chain",
    "pb-ctr",
    "pb-global_all_pairs",
    "pb-pivot_one",
)
SOURCE_DISPLAY_NAMES = {
    "real-targets": r"True Bias$^{\mathrm{oracle}}$",
    "add-two-tower": r"Add.\ Two-Tower Bias",
    "mul-two-tower": r"Mul.\ Two-Tower Bias",
    "pb-adjacent_chain": "Adjacent Chain",
    "pb-ctr": "CTR",
    "pb-global_all_pairs": "Global All Pairs",
    "pb-pivot_one": "Pivot One",
}
RANDOM_STATES = tuple(sorted(parse_random_states("40,41,42")))


def build_grouped_pairs(records):
    grouped = {}
    for record in records:
        key = (
            record.get("dataset"),
            record["objective"],
            record["source"],
            record["n_sessions"],
            record["random_state"],
            record["propensity_model"],
        )
        grouped[key] = record["metric_values"]["NDCG"]
    return grouped


def compute_cell_data(records_lookup, dataset, objective, source, n_sessions):
    freq_values = []
    mlp_values = []

    for random_state in RANDOM_STATES:
        freq = records_lookup.get((dataset, objective, source, n_sessions, random_state, "frequency-based"))
        mlp = records_lookup.get((dataset, objective, source, n_sessions, random_state, "MLPregression"))
        if freq is None or mlp is None:
            continue
        freq_values.append(freq)
        mlp_values.append(mlp)

    if not freq_values or not mlp_values:
        return None

    deltas = [mlp - freq for mlp, freq in zip(mlp_values, freq_values)]
    mean_delta = sum(deltas) / len(deltas)
    std_delta = sample_std(deltas)
    freq_mean = sum(freq_values) / len(freq_values)
    mlp_mean = sum(mlp_values) / len(mlp_values)
    p_value = paired_test_pvalue(mlp_values, freq_values)
    symbol = get_sig_symbol(mlp_mean, freq_mean, p_value)
    return {
        "cell": f"{mean_delta:+.3f} ({std_delta:.3f}){symbol}",
        "best_absolute_mean": max(mlp_mean, freq_mean),
    }


def render_table(records_lookup):
    cell_data = {}
    best_absolute_map = {}
    for dataset_key, _ in DATASET_ORDER:
        for objective in OBJECTIVE_ORDER:
            for n_sessions in SESSION_ORDER:
                best_absolute = None
                for source in SOURCE_ORDER:
                    data = compute_cell_data(records_lookup, dataset_key, objective, source, n_sessions)
                    cell_data[(dataset_key, objective, source, n_sessions)] = data
                    if data is None or source == "real-targets":
                        continue
                    best_absolute = (
                        data["best_absolute_mean"]
                        if best_absolute is None
                        else max(best_absolute, data["best_absolute_mean"])
                    )
                best_absolute_map[(dataset_key, objective, n_sessions)] = best_absolute

    lines = []
    lines.append("\\begin{table}[h!]")
    lines.append(
        "  \\caption{Paired difference between the feature-based MLP propensity estimator and the "
        "frequency-based propensity estimator on MSLR-WEB30K, Yahoo!, and Istella-S for the matched "
        "two-tower bias experiments. Rows are grouped by dataset, and columns are grouped first by "
        "counterfactual learning objective and then by sample count. Cells show "
        "$\\Delta=\\text{MLP}-\\text{frequency-based}$ in NDCG, with the standard deviation of paired "
        "differences in parentheses. Bold marks the row whose highest absolute mean NDCG across the two "
        "propensity estimators is largest within each dataset-objective-sample block. Significance markers "
        "use a paired two-sided Student t-test against zero difference: $p < 0.01$ "
        "($\\,^\\blacktriangle \\text{ and } ^\\blacktriangledown$), $p < 0.05$ "
        "($\\,^\\triangle \\text{ and } ^\\triangledown$).}"
    )
    lines.append("  \\centering")
    lines.append("  \\setlength{\\tabcolsep}{4pt}")
    lines.append("  \\resizebox{\\textwidth}{!}{")
    lines.append("    \\begin{tabular}{llcccccc}")
    lines.append("      \\toprule")
    lines.append("      & & \\multicolumn{2}{c}{\\textbf{IPS}} & \\multicolumn{2}{c}{\\textbf{DM}} & \\multicolumn{2}{c}{\\textbf{DR}} \\\\")
    lines.append("      \\cmidrule(lr){3-4}\\cmidrule(lr){5-6}\\cmidrule(lr){7-8}")
    lines.append(
        "      \\textbf{Dataset} & \\textbf{Bias Source} & \\sampleNheader{4} & \\sampleNheader{5} & "
        "\\sampleNheader{4} & \\sampleNheader{5} & \\sampleNheader{4} & \\sampleNheader{5} \\\\"
    )
    lines.append("      \\midrule")

    for dataset_idx, (dataset_key, dataset_label) in enumerate(DATASET_ORDER):
        lines.append(f"      \\multirow{{{len(SOURCE_ORDER)}}}{{*}}{{{dataset_label}}}")
        for source_idx, source in enumerate(SOURCE_ORDER):
            row = f"      & {SOURCE_DISPLAY_NAMES[source]}"
            for objective in OBJECTIVE_ORDER:
                for n_sessions in SESSION_ORDER:
                    data = cell_data[(dataset_key, objective, source, n_sessions)]
                    if data is None:
                        row += " & -"
                        continue
                    best_absolute = best_absolute_map[(dataset_key, objective, n_sessions)]
                    is_best = (
                        source != "real-targets"
                        and best_absolute is not None
                        and abs(data["best_absolute_mean"] - best_absolute) < 1e-12
                    )
                    cell_text = data["cell"]
                    if is_best:
                        cell_text = f"\\textbf{{{cell_text}}}"
                    row += f" & {cell_text}"
            row += " \\\\"
            lines.append(row)
            if source_idx == 0:
                lines.append("      \\cmidrule(lr){2-8}")
        if dataset_idx != len(DATASET_ORDER) - 1:
            lines.append("      \\midrule")

    lines.append("      \\bottomrule")
    lines.append("  \\end{tabular}}")
    lines.append(f"  \\label{{{TABLE_LABEL}}}")
    lines.append("\\end{table}")
    return "\n".join(lines)


def write_outputs(table_str):
    for output_dir in (NOTEBOOK_TABLES_DIR, THESIS_TABLES_DIR):
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / OUTPUT_NAME
        with output_path.open("w") as handle:
            handle.write(table_str)
        print(f"Wrote {output_path}")


def main():
    records_with_dataset = []
    for dataset, _ in DATASET_ORDER:
        for n_sessions in SESSION_ORDER:
            for record in load_records(
                dataset=dataset,
                n_sessions=n_sessions,
                temperature=0.5,
                random_states=set(RANDOM_STATES),
            ):
                enriched = dict(record)
                enriched["dataset"] = dataset
                records_with_dataset.append(enriched)

    if not records_with_dataset:
        raise RuntimeError("No matching records found for the MLP-minus-frequency table.")

    records_lookup = build_grouped_pairs(records_with_dataset)
    table_str = render_table(records_lookup)
    write_outputs(table_str)


if __name__ == "__main__":
    main()
