from __future__ import annotations

import importlib
import json
import logging
import sys
from pathlib import Path

import jax
import numpy as np
import pandas as pd
from hydra.utils import get_original_cwd
from omegaconf import DictConfig

from feature_based_propensities_for_ULTR.data.runtime import resolve_click_bundle_cutoff
from feature_based_propensities_for_ULTR.prebuilt import is_prebuilt_click_mode
from feature_based_propensities_for_ULTR.simulation.session_sampling import (
    build_query_sampling_state,
    sample_session_indices,
)

from .config_resolver import (
    resolve_aggregation_session_budget,
    resolve_dataset_label,
    resolve_ips_config,
    split_sessions,
)
from .data_pipeline import prepare_data_bundle

logger = logging.getLogger(__name__)

POSITION_BIAS_CSV_COLUMNS = [
    "dataset",
    "top_x",
    "n_sessions_requested",
    "n_sessions_used",
    "policy_temperature",
    "estimator",
    "position",
    "examination",
    "relative_position_bias_logit",
]

POSITION_BIAS_CSV_KEY_COLUMNS = [
    "dataset",
    "top_x",
    "n_sessions_requested",
    "n_sessions_used",
    "policy_temperature",
    "estimator",
    "position",
]


def _workspace_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _import_ultr_bias_module(module_name: str):
    try:
        return importlib.import_module(module_name)
    except ImportError as first_error:
        toolkit_root = _workspace_root() / "ultr-bias-toolkit"
        if toolkit_root.exists():
            toolkit_root_str = str(toolkit_root)
            if toolkit_root_str not in sys.path:
                sys.path.insert(0, toolkit_root_str)
            try:
                return importlib.import_module(module_name)
            except ImportError:
                pass
        raise ImportError(
            "Could not import ultr_bias_toolkit. Install the package or ensure the "
            f"local checkout exists at '{toolkit_root}'."
        ) from first_error


def build_ultr_estimator_registry() -> dict[str, object]:
    naive_module = _import_ultr_bias_module("ultr_bias_toolkit.bias.naive")
    intervention_module = _import_ultr_bias_module(
        "ultr_bias_toolkit.bias.intervention_harvesting"
    )
    return {
        "ctr": naive_module.NaiveCtrEstimator(),
        "pivot_one": intervention_module.PivotEstimator(pivot_rank=1),
        "adjacent_chain": intervention_module.AdjacentChainEstimator(),
        "global_all_pairs": intervention_module.AllPairsEstimator(),
    }


def _resolve_output_path(config: DictConfig) -> Path:
    output_path = Path(str(config.ips.position_bias.export_path)).expanduser()
    if not output_path.is_absolute():
        output_path = _resolve_relative_to_original_cwd(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path.resolve()


def _position_bias_base_epsilon(config: DictConfig) -> float:
    position_bias_cfg = getattr(getattr(config, "ips", None), "position_bias", None)
    raw_value = (
        getattr(position_bias_cfg, "base_epsilon", 0.001)
        if position_bias_cfg is not None
        else 0.001
    )
    epsilon = float(raw_value)
    if not np.isfinite(epsilon) or epsilon <= 0.0 or epsilon >= 1.0:
        raise ValueError(
            f"ips.position_bias.base_epsilon must be finite and in (0, 1), got {raw_value}."
        )
    return epsilon


def _resolve_relative_to_original_cwd(path: Path) -> Path:
    if path.is_absolute():
        return path
    try:
        return Path(get_original_cwd()) / path
    except Exception:
        return Path.cwd() / path


def _sanitize_position_bias_probabilities(
    probabilities: np.ndarray,
    *,
    base_epsilon: float,
) -> tuple[np.ndarray, int, int]:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    invalid_probability = ~(np.isfinite(probabilities) & (probabilities > 0.0))
    below_floor = np.isfinite(probabilities) & (probabilities > 0.0) & (probabilities < base_epsilon)
    sanitized = np.where(invalid_probability, base_epsilon, probabilities)
    sanitized = np.maximum(sanitized, base_epsilon)
    return sanitized, int(np.sum(invalid_probability)), int(np.sum(below_floor))


def _write_position_bias_csv(output_df: pd.DataFrame, output_path: Path) -> None:
    output_df = output_df[POSITION_BIAS_CSV_COLUMNS].copy()
    if output_path.exists():
        existing_df = pd.read_csv(output_path)
        missing_columns = [
            col for col in POSITION_BIAS_CSV_COLUMNS if col not in existing_df.columns
        ]
        if missing_columns:
            raise ValueError(
                "Existing position-bias CSV is missing required columns for append/update: "
                f"{', '.join(missing_columns)}"
            )
        existing_df = existing_df[POSITION_BIAS_CSV_COLUMNS]
        output_df = pd.concat([existing_df, output_df], ignore_index=True)
        output_df = output_df.drop_duplicates(
            subset=POSITION_BIAS_CSV_KEY_COLUMNS,
            keep="last",
        )

    output_df = output_df.sort_values(POSITION_BIAS_CSV_KEY_COLUMNS).reset_index(drop=True)
    output_df.to_csv(output_path, index=False)


def _resolve_session_budget(
    config: DictConfig,
    *,
    train_len: int,
    val_len: int,
) -> tuple[int | None, int, int, int]:
    requested_n_sessions = resolve_aggregation_session_budget(config, train_len, val_len)
    if requested_n_sessions is None:
        n_train = int(train_len)
        n_val = int(val_len)
        total_requested = int(train_len + val_len)
        return None, n_train, n_val, total_requested

    _, n_train, n_val = split_sessions(
        int(requested_n_sessions),
        int(train_len),
        int(val_len),
    )
    return int(requested_n_sessions), int(n_train), int(n_val), int(requested_n_sessions)


def _sample_click_dataset_indices(
    click_dataset,
    *,
    n_sessions: int | None,
    rng_key,
    use_query_doc_id_mapping: bool,
) -> np.ndarray:
    query_state = build_query_sampling_state(
        np.asarray(click_dataset.query),
        use_query_doc_id_mapping=use_query_doc_id_mapping,
    )
    sample_idx, _ = sample_session_indices(
        total_sessions=len(click_dataset),
        n_sessions=n_sessions,
        use_query_doc_id_mapping=use_query_doc_id_mapping,
        query_group_idx_per_row=query_state.query_group_idx_per_row,
        query_out=query_state.query_out,
        rng_key=rng_key,
        debug=False,
    )
    return np.asarray(sample_idx, dtype=np.int32)


def _extract_click_log_frame(click_dataset, sample_idx: np.ndarray) -> pd.DataFrame:
    if sample_idx.size == 0:
        return pd.DataFrame(columns=["query_id", "doc_id", "position", "click"])

    sample_idx = np.asarray(sample_idx, dtype=np.int64)
    session_rows = np.asarray(click_dataset.sessions, dtype=np.int64)[sample_idx]
    query_ids = np.asarray(click_dataset.query).reshape(-1)[session_rows]
    query_doc_ids = np.asarray(click_dataset.query_doc_ids)[session_rows]
    mask = np.asarray(click_dataset.mask, dtype=bool)[session_rows]
    clicks = np.asarray(click_dataset.clicks, dtype=np.int8)[sample_idx]
    positions_raw = np.asarray(click_dataset.positions, dtype=np.int64)[sample_idx]

    if getattr(click_dataset, "positions_are_dense_identity", False):
        position_idx = np.broadcast_to(
            np.arange(clicks.shape[1], dtype=np.int64)[None, :],
            clicks.shape,
        )
        displayed_mask = mask
    else:
        position_idx = np.asarray(
            click_dataset._normalize_position_indices(positions_raw),
            dtype=np.int64,
        )
        valid_idx = (position_idx >= 0) & (position_idx < query_doc_ids.shape[1])
        safe_position_idx = np.where(valid_idx, position_idx, 0)
        displayed_mask = np.take_along_axis(mask, safe_position_idx, axis=1) & valid_idx
        position_idx = safe_position_idx

    displayed_doc_ids = np.take_along_axis(query_doc_ids, position_idx, axis=1)
    rank_positions = np.broadcast_to(
        np.arange(clicks.shape[1], dtype=np.int64)[None, :] + 1,
        clicks.shape,
    )
    valid = displayed_mask & (displayed_doc_ids >= 0)
    if not np.any(valid):
        return pd.DataFrame(columns=["query_id", "doc_id", "position", "click"])

    return pd.DataFrame(
        {
            "query_id": np.broadcast_to(query_ids[:, None], clicks.shape)[valid],
            "doc_id": displayed_doc_ids[valid],
            "position": rank_positions[valid],
            "click": clicks[valid].astype(np.int8, copy=False),
        }
    )


def _estimate_single_method(
    *,
    click_log_df: pd.DataFrame,
    estimator_name: str,
    estimator,
    cutoff: int,
    metadata: dict[str, object],
    base_epsilon: float,
) -> pd.DataFrame:
    examination_df = estimator(click_log_df, query_col="query_id", doc_col="doc_id")
    examination_df = examination_df[["position", "examination"]].copy()
    examination_df["position"] = examination_df["position"].astype(np.int64) - 1
    examination_df = examination_df[
        (examination_df["position"] >= 0) & (examination_df["position"] < cutoff)
    ]
    examination_df = (
        examination_df.groupby("position", as_index=False)["examination"]
        .mean()
        .sort_values("position")
    )

    full_df = pd.DataFrame({"position": np.arange(cutoff, dtype=np.int64)})
    full_df = full_df.merge(examination_df, on="position", how="left")
    full_df["examination"] = full_df["examination"].fillna(0.0).astype(np.float64)
    clipped, _, _ = _sanitize_position_bias_probabilities(
        full_df["examination"].to_numpy(dtype=np.float64),
        base_epsilon=base_epsilon,
    )
    full_df["examination"] = clipped
    relative_logits = np.log(clipped)
    relative_logits -= float(relative_logits[0])

    for key, value in metadata.items():
        full_df[key] = value
    full_df["estimator"] = estimator_name
    full_df["relative_position_bias_logit"] = relative_logits
    return full_df[POSITION_BIAS_CSV_COLUMNS]


def _position_bias_estimator_name(config: DictConfig) -> str:
    position_bias_cfg = getattr(getattr(config, "ips", None), "position_bias", None)
    raw_value = (
        getattr(position_bias_cfg, "estimator", "pivot_one")
        if position_bias_cfg is not None
        else "pivot_one"
    )
    estimator_name = str(raw_value).strip()
    if not estimator_name:
        raise ValueError("ips.position_bias.estimator must be non-empty when source='estimate'.")
    return estimator_name


def _sample_position_bias_click_log(
    *,
    config: DictConfig,
    resolved,
    click_bundle,
) -> tuple[pd.DataFrame, int, int]:
    _, n_train, n_val, total_requested = _resolve_session_budget(
        config,
        train_len=len(click_bundle.train),
        val_len=len(click_bundle.val),
    )

    rng = jax.random.PRNGKey(resolved.seed)
    rng_train, rng_val, _ = jax.random.split(rng, 3)
    use_query_doc_id_mapping = is_prebuilt_click_mode(config)

    train_idx = _sample_click_dataset_indices(
        click_bundle.train,
        n_sessions=n_train,
        rng_key=rng_train,
        use_query_doc_id_mapping=use_query_doc_id_mapping,
    )
    val_idx = _sample_click_dataset_indices(
        click_bundle.val,
        n_sessions=n_val,
        rng_key=rng_val,
        use_query_doc_id_mapping=use_query_doc_id_mapping,
    )
    n_sessions_used = int(train_idx.shape[0] + val_idx.shape[0])

    click_log_df = pd.concat(
        [
            _extract_click_log_frame(click_bundle.train, train_idx),
            _extract_click_log_frame(click_bundle.val, val_idx),
        ],
        ignore_index=True,
    )
    if click_log_df.empty:
        raise ValueError("No sampled click rows were available for position-bias estimation.")

    return click_log_df, n_sessions_used, int(total_requested)


def _position_bias_frame_to_values(
    position_bias_df: pd.DataFrame,
    *,
    base_epsilon: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    probabilities = pd.to_numeric(
        position_bias_df["examination"],
        errors="coerce",
    ).to_numpy(dtype=np.float64)
    probabilities, n_invalid, n_below_floor = _sanitize_position_bias_probabilities(
        probabilities,
        base_epsilon=base_epsilon,
    )
    if n_invalid > 0 or n_below_floor > 0:
        logger.warning(
            "Applying base epsilon=%s to %d estimated position-bias values "
            "(invalid=%d, below_floor=%d).",
            base_epsilon,
            n_invalid + n_below_floor,
            n_invalid,
            n_below_floor,
        )

    alpha_logits = np.log(probabilities)
    alpha_logits -= float(alpha_logits[0])
    alpha = 1.0 / (1.0 + np.exp(-alpha_logits))
    return alpha_logits, alpha, probabilities


def _json_safe_float_list(values: np.ndarray) -> list[float]:
    return [float(value) for value in np.asarray(values, dtype=np.float64).reshape(-1)]


def write_estimated_position_bias_json(
    *,
    output_path: Path,
    position_bias_df: pd.DataFrame,
    alpha_logits: np.ndarray,
    alpha: np.ndarray,
    examination: np.ndarray,
    source: str = "estimate",
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = position_bias_df.sort_values("position").reset_index(drop=True)
    if df.empty:
        raise ValueError("Cannot write empty estimated position-bias curve.")

    first_row = df.iloc[0]
    output = {
        "source": source,
        "dataset": str(first_row["dataset"]),
        "top_x": int(first_row["top_x"]),
        "n_sessions_requested": int(first_row["n_sessions_requested"]),
        "n_sessions_used": int(first_row["n_sessions_used"]),
        "policy_temperature": float(first_row["policy_temperature"]),
        "estimator": str(first_row["estimator"]),
        "position": [int(value) for value in df["position"].to_numpy(dtype=np.int64)],
        "examination": _json_safe_float_list(examination),
        "relative_position_bias_logit": _json_safe_float_list(alpha_logits),
        "alpha": _json_safe_float_list(alpha),
    }
    with output_path.open("w") as f:
        json.dump(output, f)
    logger.info("Saved estimated position-bias curve to %s", output_path)
    return output_path


def estimate_position_bias_values(
    *,
    config: DictConfig,
    click_bundle,
    cutoff: int,
    dataset_label: str,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, np.ndarray, str]:
    resolved = resolve_ips_config(config)
    base_epsilon = _position_bias_base_epsilon(config)
    estimator_name = _position_bias_estimator_name(config)
    estimators = build_ultr_estimator_registry()
    if estimator_name not in estimators:
        raise ValueError(
            "Unknown ips.position_bias.estimator "
            f"'{estimator_name}'. Expected one of: {', '.join(sorted(estimators))}."
        )

    click_log_df, n_sessions_used, total_requested = _sample_position_bias_click_log(
        config=config,
        resolved=resolved,
        click_bundle=click_bundle,
    )
    metadata = {
        "dataset": dataset_label,
        "top_x": int(cutoff),
        "n_sessions_requested": int(total_requested),
        "n_sessions_used": int(n_sessions_used),
        "policy_temperature": float(getattr(config, "policy_temperature", 0.0)),
    }
    position_bias_df = _estimate_single_method(
        click_log_df=click_log_df,
        estimator_name=estimator_name,
        estimator=estimators[estimator_name],
        cutoff=int(cutoff),
        metadata=metadata,
        base_epsilon=base_epsilon,
    )
    alpha_logits, alpha, examination = _position_bias_frame_to_values(
        position_bias_df,
        base_epsilon=base_epsilon,
    )
    return alpha_logits, alpha, position_bias_df, examination, estimator_name


def estimate_position_bias(config: DictConfig) -> Path:
    resolved = resolve_ips_config(config)
    dataset_label = resolve_dataset_label(config)
    base_epsilon = _position_bias_base_epsilon(config)
    data_bundle = prepare_data_bundle(config, resolved, force_click_datasets=True)
    click_bundle = data_bundle.clicks
    if click_bundle is None:
        raise RuntimeError("Position-bias estimation requires loaded click datasets.")

    cutoff = int(
        data_bundle.aggregated.cutoff
        if data_bundle.aggregated is not None
        else resolve_click_bundle_cutoff(config, click_bundle)
    )
    click_log_df, n_sessions_used, total_requested = _sample_position_bias_click_log(
        config=config,
        resolved=resolved,
        click_bundle=click_bundle,
    )

    metadata = {
        "dataset": dataset_label,
        "top_x": int(cutoff),
        "n_sessions_requested": int(total_requested),
        "n_sessions_used": int(n_sessions_used),
        "policy_temperature": float(getattr(config, "policy_temperature", 0.0)),
    }

    estimator_frames = [
        _estimate_single_method(
            click_log_df=click_log_df,
            estimator_name=name,
            estimator=estimator,
            cutoff=cutoff,
            metadata=metadata,
            base_epsilon=base_epsilon,
        )
        for name, estimator in build_ultr_estimator_registry().items()
    ]
    output_df = pd.concat(estimator_frames, ignore_index=True)

    output_path = _resolve_output_path(config)
    _write_position_bias_csv(output_df, output_path)
    logger.info("Saved ULTR position-bias estimates to %s", output_path)
    return output_path


def _require_csv_config(config: DictConfig) -> tuple[Path, str]:
    csv_cfg = getattr(getattr(config.ips, "position_bias", None), "csv", None)
    if csv_cfg is None:
        raise ValueError("ips.position_bias.csv must be configured when source='csv'.")

    raw_path = getattr(csv_cfg, "path", None)
    if raw_path in (None, ""):
        raise ValueError("ips.position_bias.csv.path is required when source='csv'.")

    raw_method = getattr(csv_cfg, "method", None)
    if raw_method in (None, ""):
        raise ValueError("ips.position_bias.csv.method is required when source='csv'.")

    path = Path(str(raw_path)).expanduser()
    if not path.is_absolute():
        path = _resolve_relative_to_original_cwd(path)
    return path.resolve(), str(raw_method)


def load_position_bias_from_csv(
    *,
    config: DictConfig,
    dataset_label: str,
    cutoff: int,
    n_sessions_used: int,
) -> tuple[np.ndarray, np.ndarray, Path, str]:
    csv_path, method = _require_csv_config(config)
    base_epsilon = _position_bias_base_epsilon(config)
    if not csv_path.exists():
        raise FileNotFoundError(f"Position-bias CSV does not exist: {csv_path}")

    df = pd.read_csv(csv_path)
    missing_columns = [col for col in POSITION_BIAS_CSV_COLUMNS if col not in df.columns]
    if missing_columns:
        raise ValueError(
            "Position-bias CSV is missing required columns: "
            f"{', '.join(missing_columns)}"
        )

    policy_temperature = float(getattr(config, "policy_temperature", 0.0))
    mask = (
        (df["dataset"].astype(str) == str(dataset_label))
        & (df["top_x"].astype(int) == int(cutoff))
        & (df["n_sessions_used"].astype(int) == int(n_sessions_used))
        & (df["estimator"].astype(str) == method)
        & np.isclose(df["policy_temperature"].astype(float), policy_temperature)
    )
    matched = df.loc[mask].copy()
    if matched.empty:
        raise ValueError(
            "No matching position-bias rows found in CSV for "
            f"dataset='{dataset_label}', top_x={int(cutoff)}, n_sessions_used={int(n_sessions_used)}, "
            f"policy_temperature={policy_temperature}, estimator='{method}'."
        )

    if matched["position"].duplicated().any():
        raise ValueError("Position-bias CSV contains duplicate rows for at least one position.")

    matched = matched.sort_values("position").reset_index(drop=True)
    observed_positions = matched["position"].astype(int).to_numpy()
    expected_positions = np.arange(int(cutoff), dtype=np.int64)
    if not np.array_equal(observed_positions, expected_positions):
        raise ValueError(
            "Position-bias CSV must contain exactly one row per position in "
            f"0..{int(cutoff) - 1}; observed positions={observed_positions.tolist()}."
        )

    examination = pd.to_numeric(matched["examination"], errors="coerce").to_numpy(dtype=np.float64)
    relative_logits_raw = pd.to_numeric(
        matched["relative_position_bias_logit"],
        errors="coerce",
    ).to_numpy(dtype=np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        examination_from_logits = np.exp(relative_logits_raw)
    use_direct_examination = np.isfinite(examination) & (examination > 0.0)
    probabilities = np.where(use_direct_examination, examination, examination_from_logits)
    probabilities, n_invalid, n_below_floor = _sanitize_position_bias_probabilities(
        probabilities,
        base_epsilon=base_epsilon,
    )
    if n_invalid > 0 or n_below_floor > 0:
        logger.warning(
            "Applying base epsilon=%s to %d position-bias values while reading %s "
            "(invalid=%d, below_floor=%d).",
            base_epsilon,
            n_invalid + n_below_floor,
            csv_path,
            n_invalid,
            n_below_floor,
        )

    alpha_logits = np.log(probabilities)
    alpha_logits -= float(alpha_logits[0])
    alpha = 1.0 / (1.0 + np.exp(-alpha_logits))
    return alpha_logits, alpha, csv_path, method


__all__ = [
    "POSITION_BIAS_CSV_COLUMNS",
    "build_ultr_estimator_registry",
    "estimate_position_bias",
    "estimate_position_bias_values",
    "load_position_bias_from_csv",
    "write_estimated_position_bias_json",
]
