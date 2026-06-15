"""Schema contracts and validators for prebuilt click datasets."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from feature_based_propensities_for_ULTR.prebuilt import (
    prefer_existing_cutoff_variant,
    prebuilt_manifest_path,
    prebuilt_split_artifact_path,
)


QUERY_DOC_FEATURE_NAMES = [
    "bm25",
    "bm25_title",
    "bm25_abstract",
    "tf_idf",
    "tf",
    "idf",
    "ql_jelinek_mercer_short",
    "ql_jelinek_mercer_long",
    "ql_dirichlet",
    "query_length",
    "document_length",
    "title_length",
    "abstract_length",
]

LP_FEATURE_NAMES = [
    "position",
    "media_type",
    "displayed_time",
    "serp_height",
    "slipoff_count_after_click",
]

REQUIRED_CLICK_DATASET_KEYS = {
    "sessions",
    "clicks",
    "positions",
    "sessions_per_query",
    "query",
    "query_doc_features",
    "lp_query_doc_features",
    "query_doc_ids",
    "labels",
    "mask",
    "n",
}

OPTIONAL_CLICK_DATASET_KEYS = {
    "sessions_per_doc_pos",
    "sessions_per_doc_pos_diag",
    "has_diagonal_only_sessions_per_doc_pos",
    "positions_are_dense_identity",
    "positions_are_one_based",
}


def _require_dtype_kind(
    split_name: str,
    name: str,
    value: np.ndarray,
    *,
    allowed_kinds: str,
) -> None:
    dtype = np.asarray(value).dtype
    if dtype.kind not in allowed_kinds:
        raise ValueError(
            f"[{split_name}] {name} has invalid dtype {dtype}; expected kind in {list(allowed_kinds)}."
        )


def _validate_dtypes(split_name: str, dataset: dict[str, np.ndarray]) -> None:
    _require_dtype_kind(split_name, "sessions", dataset["sessions"], allowed_kinds="iu")
    _require_dtype_kind(split_name, "clicks", dataset["clicks"], allowed_kinds="iu")
    _require_dtype_kind(split_name, "positions", dataset["positions"], allowed_kinds="iu")
    _require_dtype_kind(
        split_name,
        "sessions_per_query",
        dataset["sessions_per_query"],
        allowed_kinds="iu",
    )
    count_key = "sessions_per_doc_pos_diag" if "sessions_per_doc_pos_diag" in dataset else "sessions_per_doc_pos"
    _require_dtype_kind(
        split_name,
        count_key,
        dataset[count_key],
        allowed_kinds="iu",
    )
    _require_dtype_kind(split_name, "query", dataset["query"], allowed_kinds="iu")
    _require_dtype_kind(split_name, "query_doc_features", dataset["query_doc_features"], allowed_kinds="f")
    _require_dtype_kind(
        split_name,
        "lp_query_doc_features",
        dataset["lp_query_doc_features"],
        allowed_kinds="f",
    )
    _require_dtype_kind(split_name, "query_doc_ids", dataset["query_doc_ids"], allowed_kinds="iu")
    _require_dtype_kind(split_name, "labels", dataset["labels"], allowed_kinds="f")
    _require_dtype_kind(split_name, "mask", dataset["mask"], allowed_kinds="b")
    _require_dtype_kind(split_name, "n", dataset["n"], allowed_kinds="iu")


def _positions_are_dense_identity(positions: np.ndarray, max_len: int) -> bool:
    """Accept dense rank identities in either 0-based or 1-based form."""
    zero_based = np.broadcast_to(
        np.arange(max_len, dtype=positions.dtype)[None, :],
        positions.shape,
    )
    one_based = np.broadcast_to(
        np.arange(1, max_len + 1, dtype=positions.dtype)[None, :],
        positions.shape,
    )
    return bool(np.array_equal(positions, zero_based) or np.array_equal(positions, one_based))


@dataclass
class SplitStats:
    name: str
    files: list[str]
    rows_total: int = 0
    rows_kept: int = 0
    rows_dropped_due_to_max_len: int = 0
    rows_dropped_invalid_position: int = 0
    rows_dropped_duplicate_position: int = 0
    sessions: int = 0
    max_docs_per_session_observed: int = 0


# ---------------------------------------------------------------------------
# Shared shape-validation helper (used by both parse-time and load-time checks)
# ---------------------------------------------------------------------------


def _validate_payload_shapes(
    split: str,
    data: dict[str, np.ndarray],
    max_len: int,
) -> dict[str, int]:
    """Validate shapes of all arrays in a click-dataset payload.

    Returns a summary dict suitable for comparison against the manifest.
    Raises ``ValueError`` on any shape or contract violation.
    """
    keys = set(data.keys())
    missing = REQUIRED_CLICK_DATASET_KEYS - keys
    if missing:
        raise ValueError(f"[{split}] Missing keys: {sorted(missing)}")
    if "sessions_per_doc_pos" not in keys and "sessions_per_doc_pos_diag" not in keys:
        raise ValueError(
            f"[{split}] Missing one of required count tensors: "
            "['sessions_per_doc_pos', 'sessions_per_doc_pos_diag']"
        )

    _validate_dtypes(split, data)

    sessions = np.asarray(data["sessions"])
    n_sessions = int(sessions.shape[0])
    if n_sessions <= 0:
        raise ValueError(f"[{split}] Expected positive number of sessions.")

    expected_1d = (n_sessions,)
    for name in ("sessions", "sessions_per_query", "query", "n"):
        if np.asarray(data[name]).shape != expected_1d:
            raise ValueError(
                f"[{split}] {name} shape mismatch: "
                f"{np.asarray(data[name]).shape} vs {expected_1d}"
            )

    expected_2d = (n_sessions, max_len)
    for name in ("clicks", "positions", "mask", "labels", "query_doc_ids"):
        if np.asarray(data[name]).shape != expected_2d:
            raise ValueError(
                f"[{split}] {name} shape mismatch: "
                f"{np.asarray(data[name]).shape} vs {expected_2d}"
            )

    if np.asarray(data["query_doc_features"]).shape[:2] != expected_2d:
        raise ValueError(
            f"[{split}] query_doc_features shape mismatch: "
            f"{np.asarray(data['query_doc_features']).shape[:2]} vs {expected_2d}"
        )
    if np.asarray(data["lp_query_doc_features"]).shape[:2] != expected_2d:
        raise ValueError(
            f"[{split}] lp_query_doc_features shape mismatch: "
            f"{np.asarray(data['lp_query_doc_features']).shape[:2]} vs {expected_2d}"
        )

    if "sessions_per_doc_pos_diag" in data:
        sessions_per_doc_pos = np.asarray(data["sessions_per_doc_pos_diag"])
        expected_counts = (n_sessions, max_len)
        if sessions_per_doc_pos.shape != expected_counts:
            raise ValueError(
                f"[{split}] sessions_per_doc_pos_diag shape mismatch: "
                f"{sessions_per_doc_pos.shape} vs {expected_counts}"
            )
    else:
        sessions_per_doc_pos = np.asarray(data["sessions_per_doc_pos"])
        if sessions_per_doc_pos.shape != (n_sessions, max_len, max_len):
            raise ValueError(
                f"[{split}] sessions_per_doc_pos shape mismatch: "
                f"{sessions_per_doc_pos.shape} vs {(n_sessions, max_len, max_len)}"
            )

    positions = np.asarray(data["positions"])
    if not _positions_are_dense_identity(positions, max_len):
        raise ValueError(
            f"[{split}] positions must be dense identity ranks in 0-based or 1-based form."
        )

    mask = np.asarray(data["mask"], dtype=bool)
    n = np.asarray(data["n"])
    valid_docs = int(mask.sum())
    if int(n.sum()) != valid_docs:
        raise ValueError(
            f"[{split}] n.sum() != mask.sum() ({int(n.sum())} vs {valid_docs})."
        )

    clicks = np.asarray(data["clicks"])
    return {
        "sessions": n_sessions,
        "rows_kept": valid_docs,
        "max_docs_per_session": int(np.max(n)),
        "total_clicks": int(np.sum(clicks)),
    }


# ---------------------------------------------------------------------------
# Parse-time validator (called immediately after parsing, before saving)
# ---------------------------------------------------------------------------


def validate_dataset_payload(split_name: str, dataset: dict[str, np.ndarray], *, max_len: int) -> None:
    """Validate a freshly-parsed payload dict.

    Raises ``ValueError`` on structural or semantic errors.
    """
    # Shape / dtype checks (shared)
    _validate_payload_shapes(split_name, dataset, max_len)

    # Additional semantic checks not covered by _validate_payload_shapes:
    query_doc_ids = np.asarray(dataset["query_doc_ids"])
    mask = np.asarray(dataset["mask"], dtype=bool)
    sessions_per_query = np.asarray(dataset["sessions_per_query"])
    n = np.asarray(dataset["n"])

    if np.any(query_doc_ids[mask] < 0):
        raise ValueError(f"[{split_name}] query_doc_ids contains invalid ids on masked docs.")
    if np.any(sessions_per_query < 1):
        raise ValueError(f"[{split_name}] sessions_per_query contains values < 1.")
    if np.any(n < 0) or np.any(n > max_len):
        raise ValueError(f"[{split_name}] n contains invalid values outside [0, {max_len}].")

    qdoc = np.asarray(dataset["query_doc_features"])
    lp = np.asarray(dataset["lp_query_doc_features"])
    labels = np.asarray(dataset["labels"])
    if not np.isfinite(qdoc).all():
        raise ValueError(f"[{split_name}] query_doc_features contains non-finite values.")
    if not np.isfinite(lp).all():
        raise ValueError(f"[{split_name}] lp_query_doc_features contains non-finite values.")
    if not np.isfinite(labels).all():
        raise ValueError(f"[{split_name}] labels contains non-finite values.")


# ---------------------------------------------------------------------------
# Load-time validators (called when reading existing NPZ files)
# ---------------------------------------------------------------------------


def _load_manifest(processed_dir: Path, max_len: int | None = None) -> dict:
    manifest_path = prefer_existing_cutoff_variant(
        prebuilt_manifest_path(processed_dir, None),
        max_len,
    )
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")
    return json.loads(manifest_path.read_text())


def _validate_against_manifest(
    split: str,
    observed: dict[str, int],
    manifest: dict,
) -> None:
    stats = manifest.get("stats", {}).get(split, {})
    if not stats:
        raise ValueError(f"[{split}] Missing manifest stats entry.")

    for key in ("sessions", "rows_kept"):
        expected = int(stats.get(key, -1))
        got = int(observed[key])
        if expected != got:
            raise ValueError(f"[{split}] manifest mismatch for {key}: expected {expected}, got {got}")


def _validate_split_specific(split: str, data: dict[str, np.ndarray]) -> None:
    clicks = np.asarray(data["clicks"])
    labels = np.asarray(data["labels"])
    mask = np.asarray(data["mask"], dtype=bool)
    lp = np.asarray(data["lp_query_doc_features"])

    if split in {"train", "val", "test_clicks"}:
        if not np.array_equal(labels[mask], clicks[mask].astype(labels.dtype)):
            raise ValueError(f"[{split}] labels must equal clicks on valid docs.")
        return

    if split != "test":
        raise ValueError(f"Unknown split: {split}")

    if np.any(clicks != 0):
        raise ValueError("[test] clicks should be zero for label-based test dataset.")
    if np.all(labels[mask] == 0):
        raise ValueError("[test] labels appear to be all zero; expected relevance labels.")

    test_pos = lp[..., 0]
    expected_pos = np.broadcast_to(
        np.arange(1, lp.shape[1] + 1, dtype=test_pos.dtype)[None, :],
        test_pos.shape,
    )
    if not np.all(test_pos[mask] == expected_pos[mask]):
        raise ValueError("[test] lp position feature is expected to be synthetic rank+1.")
    if not np.all(test_pos[~mask] == 0):
        raise ValueError("[test] lp position should be zero for padded docs.")
    if not np.all(lp[..., 1:] == 0):
        raise ValueError("[test] non-position lp features are expected to be zero-filled.")


def validate_prebuilt_click_dataset(
    processed_dir: Path,
    *,
    max_len: int | None = None,
) -> dict[str, dict[str, int]]:
    processed_dir = processed_dir.expanduser().resolve()
    manifest = _load_manifest(processed_dir, max_len=max_len)
    max_len = int(manifest.get("max_len", 20))

    summary: dict[str, dict[str, int]] = {}
    splits = ["train", "val", "test"]
    if "test_clicks" in manifest.get("stats", {}):
        splits.append("test_clicks")

    for split in splits:
        path = prefer_existing_cutoff_variant(
            prebuilt_split_artifact_path(processed_dir, split, None),
            max_len,
        ) if split != "test_clicks" else prefer_existing_cutoff_variant(
            processed_dir / "test_clicks_click_dataset.npz",
            max_len,
        )
        if not path.exists():
            raise FileNotFoundError(f"Missing split NPZ: {path}")

        with np.load(path, allow_pickle=True) as data_npz:
            data = {k: data_npz[k] for k in data_npz.files}

        observed = _validate_payload_shapes(split, data, max_len)
        _validate_split_specific(split, data)
        _validate_against_manifest(split, observed, manifest)
        summary[split] = observed

    return summary


__all__ = [
    "LP_FEATURE_NAMES",
    "QUERY_DOC_FEATURE_NAMES",
    "REQUIRED_CLICK_DATASET_KEYS",
    "SplitStats",
    "validate_dataset_payload",
    "validate_prebuilt_click_dataset",
]
