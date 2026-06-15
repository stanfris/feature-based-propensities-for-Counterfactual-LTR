import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from .config import ExperimentConfig, NO_PROPENSITY_IN_FOLDER


SUMMARY_COLUMNS = [
    "mean_value",
    "std_value",
    "ci_low_value",
    "ci_high_value",
    "n_runs",
]


def _empty_summary_df(*columns: str) -> pd.DataFrame:
    return pd.DataFrame(columns=[*columns, *SUMMARY_COLUMNS])


def _quantile(q: float):
    return lambda values: values.quantile(q)


def safe_load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with path.open("r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def parse_folder_name(folder_name: str) -> Dict[str, str]:
    """Parse comma-separated key=value pairs into a dict."""
    parts = folder_name.split(",")
    parsed = {"folder_name": folder_name}
    for part in parts:
        if "=" in part:
            k, v = part.lstrip("+").split("=", 1)
            parsed[k] = v
    return parsed


def load_long_metrics(cfg: ExperimentConfig) -> pd.DataFrame:
    """
    Scans the base paths for ips_results.json, parses the folder name to
    extract hyperparams, filters them based on the config, and returns
    a flattened (long) DataFrame of all metrics.
    """
    records = []

    def gather_runs_from_dir(base_path: str, is_two_tower: bool):
        base_dir = Path(base_path)
        if not base_dir.exists():
            return
            
        for path in base_dir.rglob("ips_results.json"):
            folder_name = path.parent.name
            parsed = parse_folder_name(folder_name)
            
            ips_model = parsed.get("ips.model")
            if not ips_model or ips_model not in cfg.ips_models:
                continue

            dataset = parsed.get("data")
            if dataset != cfg.dataset_name:
                continue
                
            filt_str = parsed.get("ips.filter_single_display_pairs", "False")
            filt = True if filt_str.lower() == "true" else False
            if filt not in cfg.filter_single_display_pairs_values:
                continue
                
            raw_session_pct = parsed.get("ips.n_session_percentage")
            n_session_percentage = (
                float(raw_session_pct) if raw_session_pct is not None else np.nan
            )
            raw_n_sessions = parsed.get("ips.n_sessions")
            n_sessions = int(raw_n_sessions) if raw_n_sessions is not None else 0

            if cfg.n_session_percentage_list is not None:
                if np.isnan(n_session_percentage):
                    continue
                if not any(
                    np.isclose(n_session_percentage, pct, atol=1e-5)
                    for pct in cfg.n_session_percentage_list
                ):
                    continue
            else:
                if n_sessions not in cfg.n_sessions_list:
                    continue
                
            tmp = float(parsed.get("policy_temperature", 0.0))
            if not any(np.isclose(tmp, t, atol=1e-5) for t in cfg.temperatures):
                continue
                
            strength = float(parsed.get("policy_strength", 1.0))
            if not any(np.isclose(strength, s, atol=1e-5) for s in cfg.policy_strengths):
                continue
                
            # Random state
            rs = int(parsed.get("random_state", 0))
            if rs not in cfg.random_states:
                continue

            # Propensity model
            prop_model = parsed.get("propensity_model")
            if ips_model in NO_PROPENSITY_IN_FOLDER and not is_two_tower:
                prop_model = None
                
            if prop_model is not None and prop_model not in cfg.propensity_models:
                continue
                
            # Load the actual metrics
            payload = safe_load_json(path)
            if not payload:
                continue
                
            metrics = payload.get("results", {}).get("metrics", {})
            if not metrics:
                continue
            n_sessions_used = int(payload.get("n_sessions_used", n_sessions))
                
            prop_label = prop_model if prop_model else ips_model
            plot_filter = "merged" if ips_model in {"ips", "naive", "naive-ho"} else filt
            
            for metric, value in metrics.items():
                records.append({
                    "ips_model": ips_model,
                    "ips_filter_single_display_pairs": filt,
                    "plot_filter": plot_filter,
                    "n_sessions": n_sessions,
                    "n_sessions_used": n_sessions_used,
                    "n_session_percentage": n_session_percentage,
                    "propensity_model_raw": prop_model,
                    "propensity_model": prop_label,
                    "tmp": tmp,
                    "policy_strength": strength,
                    "random_state": rs,
                    "metric": metric,
                    "value": value,
                    "folder": folder_name,
                    "base_path": base_path,
                    "json_path": str(path),
                })

    gather_runs_from_dir(cfg.base_path_policy_models, is_two_tower=False)
    gather_runs_from_dir(cfg.base_path_two_tower, is_two_tower=True)
    
    if not records:
        raise ValueError("No matching JSON results found evaluating the current ExperimentConfig constraints.")
        
    return pd.DataFrame(records)


def aggregate_mean_std(df_long: pd.DataFrame) -> pd.DataFrame:
    return (
        df_long.groupby(
            [
                "ips_model",
                "plot_filter",
                "n_sessions",
                "n_sessions_used",
                "n_session_percentage",
                "propensity_model",
                "tmp",
                "metric",
                "policy_strength",
            ],
            as_index=False,
            dropna=False,
        )
        .agg(
            mean_value=("value", "mean"),
            std_value=("value", "std"),
            ci_low_value=("value", _quantile(0.05)),
            ci_high_value=("value", _quantile(0.95)),
            n_runs=("value", "size"),
        )
        .sort_values(
            [
                "ips_model",
                "plot_filter",
                "metric",
                "propensity_model",
                "tmp",
                "policy_strength",
                "n_session_percentage",
                "n_sessions",
                "n_sessions_used",
            ]
        )
        .reset_index(drop=True)
    )


def metric_only_baselines(
    agg: pd.DataFrame,
    baseline_models=("max-score", "logging-policy"),
) -> pd.DataFrame:
    b = agg[agg["ips_model"].isin(baseline_models)].copy()
    if b.empty:
        return _empty_summary_df("baseline_name", "metric")

    return (
        b.assign(baseline_name=b["ips_model"])
        .groupby(["baseline_name", "metric"], as_index=False)
        .agg(
            mean_value=("mean_value", "mean"),
            std_value=("std_value", "mean"),
            ci_low_value=("ci_low_value", "mean"),
            ci_high_value=("ci_high_value", "mean"),
            n_runs=("n_runs", "sum"),
        )
        .sort_values(["baseline_name", "metric"])
        .reset_index(drop=True)
    )

def load_baselines_from_folder(
    cfg: ExperimentConfig,
    metrics_to_plot=("RCTR", "NDCG"),
    max_score_n_sessions: Optional[int] = 500000,
    baselines_path: str = "results/baselines",
):
    rows = []
    base_dir = Path(baselines_path)
    if not base_dir.exists():
        return _empty_summary_df("baseline_name", "metric")
        
    for path in base_dir.rglob("ips_results.json"):
        folder = path.parent.name
        parsed = parse_folder_name(folder)
        if parsed.get("data") != cfg.dataset_name:
            continue

        ips_model = parsed.get("ips.model")
        baseline_name = {
            "logging_policy": "logging-policy",
            "logging-policy": "logging-policy",
            "max-score": "max-score",
        }.get(ips_model)
        if baseline_name is None:
            continue

        if baseline_name == "max-score" and max_score_n_sessions is not None:
            try:
                n_sessions = int(parsed.get("ips.n_sessions", "0"))
            except ValueError:
                continue
            if n_sessions != max_score_n_sessions:
                continue
            
        data = safe_load_json(path)
        if not data:
            continue
        metrics = data.get("results", {}).get("metrics", {})
        for m, v in metrics.items():
            if m in metrics_to_plot:
                rows.append({"baseline_name": baseline_name, "metric": m, "value": v})
                
    if not rows:
        return _empty_summary_df("baseline_name", "metric")
    
    df = pd.DataFrame(rows)
    agg = df.groupby(["baseline_name", "metric"], as_index=False).agg(
        mean_value=("value", "mean"),
        std_value=("value", "std"),
        ci_low_value=("value", _quantile(0.05)),
        ci_high_value=("value", _quantile(0.95)),
        n_runs=("value", "size"),
    )
    agg["std_value"] = agg["std_value"].fillna(0.0)
    return agg
