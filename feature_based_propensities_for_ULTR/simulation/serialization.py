"""NPZ serialization helpers for simulated click datasets."""

from __future__ import annotations

from typing import Dict, Optional

import jax.numpy as jnp
import numpy as np


def _compact_array(value):
    arr = np.asarray(value)
    if arr.dtype == np.bool_ or not np.issubdtype(arr.dtype, np.integer):
        return arr
    if arr.size == 0:
        return arr

    min_value = int(arr.min())
    max_value = int(arr.max())
    if min_value >= 0:
        for dtype in (np.uint8, np.uint16, np.uint32):
            info = np.iinfo(dtype)
            if max_value <= info.max:
                return arr.astype(dtype, copy=False)
        return arr.astype(np.uint64, copy=False)

    for dtype in (np.int8, np.int16, np.int32):
        info = np.iinfo(dtype)
        if min_value >= info.min and max_value <= info.max:
            return arr.astype(dtype, copy=False)
    return arr.astype(np.int64, copy=False)


def save_click_dataset_npz(dataset, file_path: str) -> None:
    """Save a ClickDataset to a compressed NPZ file."""
    payload = {
        "sessions": _compact_array(dataset.sessions),
        "clicks": _compact_array(dataset.clicks),
        "positions": _compact_array(dataset.positions),
        "sessions_per_query": _compact_array(dataset.sessions_per_query),
        "query": _compact_array(dataset.query),
        "query_doc_features": np.asarray(dataset.query_doc_features),
        "query_doc_ids": _compact_array(dataset.query_doc_ids),
        "labels": np.asarray(dataset.labels),
        "mask": np.asarray(dataset.mask),
        "n": _compact_array(dataset.n),
        "has_diagonal_only_sessions_per_doc_pos": np.asarray(
            dataset.has_diagonal_only_sessions_per_doc_pos
        ),
        "positions_are_dense_identity": np.asarray(dataset.positions_are_dense_identity),
        "positions_are_one_based": np.asarray(dataset.positions_are_one_based),
    }
    if dataset.has_diagonal_only_sessions_per_doc_pos:
        payload["sessions_per_doc_pos_diag"] = _compact_array(dataset.sessions_per_doc_pos_diag)
    else:
        payload["sessions_per_doc_pos"] = _compact_array(dataset.sessions_per_doc_pos)
    if dataset.has_separate_lp_query_doc_features:
        payload["lp_query_doc_features"] = np.asarray(dataset.lp_query_doc_features)

    np.savez_compressed(file_path, **payload)
    print(f"Dataset saved to {file_path}")


def save_aggregated_click_dataset_npz(
    dataset,
    file_path: str,
    *,
    extra_metadata: Optional[Dict] = None,
) -> None:
    """Save an AggregatedClickDataset to a compressed NPZ file."""
    payload = {
        "query": _compact_array(dataset.query),
        "feature_matrix": np.asarray(dataset.feature_matrix),
        "label_vector": np.asarray(dataset.label_vector),
        "doc_id_vector": _compact_array(dataset.doc_id_vector),
        "doclist_ranges": _compact_array(dataset.doclist_ranges),
        "doc_id_map": _compact_array(dataset.doc_id_map),
        "clicks": _compact_array(dataset.clicks),
        "displays": _compact_array(dataset.displays),
        "clicks_per_doc": _compact_array(dataset.clicks_per_doc),
        "displays_per_doc": _compact_array(dataset.displays_per_doc),
        "query_freq": _compact_array(dataset.query_freq),
        "sessions": _compact_array(dataset.sessions),
        "cutoff": np.asarray(dataset.cutoff),
        "n_sessions": np.asarray(len(dataset.sessions)),
    }
    if dataset.has_separate_lp_feature_matrix:
        payload["lp_feature_matrix"] = np.asarray(dataset.lp_feature_matrix)
    payload["query_index_per_doc"] = (
        _compact_array(dataset.query_index_per_doc)
        if getattr(dataset, "query_index_per_doc", None) is not None
        else np.array([], dtype=np.int32)
    )
    metadata = dict(dataset.metadata)
    if extra_metadata:
        metadata.update(extra_metadata)
    for key, value in metadata.items():
        payload[f"meta_{key}"] = np.asarray(value)

    np.savez_compressed(file_path, **payload)
    print(f"Aggregated dataset saved to {file_path}")


def load_aggregated_click_dataset_npz(file_path: str):
    """Load an AggregatedClickDataset from a compressed NPZ file."""
    data = np.load(file_path, allow_pickle=True)

    def _as_jnp(key: str) -> jnp.ndarray:
        return jnp.asarray(data[key])

    def _as_int(key: str) -> int:
        return int(np.asarray(data[key]).item())

    metadata = {}
    for key in data.files:
        if key.startswith("meta_"):
            value = data[key]
            if np.asarray(value).shape == ():
                metadata[key[5:]] = np.asarray(value).item()
            else:
                metadata[key[5:]] = value

    from .datasets import AggregatedClickDataset

    return AggregatedClickDataset(
        query=_as_jnp("query"),
        feature_matrix=_as_jnp("feature_matrix"),
        lp_feature_matrix=(
            _as_jnp("lp_feature_matrix") if "lp_feature_matrix" in data.files else None
        ),
        label_vector=_as_jnp("label_vector"),
        doc_id_vector=_as_jnp("doc_id_vector"),
        doclist_ranges=_as_jnp("doclist_ranges"),
        doc_id_map=_as_jnp("doc_id_map"),
        query_index_per_doc=(
            _as_jnp("query_index_per_doc")
            if "query_index_per_doc" in data.files and np.asarray(data["query_index_per_doc"]).size > 0
            else None
        ),
        clicks=_as_jnp("clicks"),
        displays=_as_jnp("displays"),
        clicks_per_doc=_as_jnp("clicks_per_doc"),
        displays_per_doc=_as_jnp("displays_per_doc"),
        query_freq=_as_jnp("query_freq"),
        sessions=_as_jnp("sessions"),
        cutoff=_as_int("cutoff"),
        metadata=metadata,
        query_doc_ids=(
            _as_jnp("query_doc_ids") if "query_doc_ids" in data.files else None
        ),
        query_doc_features=(
            _as_jnp("query_doc_features") if "query_doc_features" in data.files else None
        ),
        lp_query_doc_features=(
            _as_jnp("lp_query_doc_features") if "lp_query_doc_features" in data.files else None
        ),
        labels=(
            _as_jnp("labels") if "labels" in data.files else None
        ),
        mask=(
            _as_jnp("mask") if "mask" in data.files else None
        ),
        n=(
            _as_jnp("n") if "n" in data.files else None
        ),
    )
