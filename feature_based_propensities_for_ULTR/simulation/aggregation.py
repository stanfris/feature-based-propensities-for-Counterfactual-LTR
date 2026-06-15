"""Aggregation utilities for building document-level click datasets."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from feature_based_propensities_for_ULTR.data.base import RatingDataset

from .datasets import AggregatedClickDataset, ClickDataset
from .session_sampling import build_query_sampling_state, sample_session_indices


@dataclass(frozen=True)
class _AggregationInputs:
    query: jnp.ndarray
    query_out: jnp.ndarray
    query_group_idx_per_row: jnp.ndarray
    doclist_ranges: jnp.ndarray
    doc_id_map: jnp.ndarray
    query_index_per_doc: jnp.ndarray | None
    feature_matrix: jnp.ndarray
    lp_feature_matrix: jnp.ndarray | None
    label_vector: jnp.ndarray
    doc_id_vector: jnp.ndarray


def _build_doc_index_mapping(mask: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    mask = mask.astype(bool)
    n_docs_per_query = jnp.sum(mask, axis=1, dtype=jnp.int32)
    doclist_ranges = jnp.concatenate(
        [
            jnp.array([0], dtype=jnp.int32),
            jnp.cumsum(n_docs_per_query, dtype=jnp.int32),
        ]
    )
    valid_rank = jnp.cumsum(mask, axis=1, dtype=jnp.int32) - 1
    doc_id_map = jnp.where(
        mask,
        doclist_ranges[:-1, None] + valid_rank,
        jnp.array(-1, dtype=jnp.int32),
    )
    return doclist_ranges, doc_id_map.astype(jnp.int32)


def _build_doc_index_mapping_shared(
    query: jnp.ndarray,
    query_doc_ids: jnp.ndarray,
    mask: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Build a document mapping that shares document ids across sessions using
    (query, query_doc_id) keys.
    """
    query_np = np.asarray(query).reshape(-1)
    query_doc_ids_np = np.asarray(query_doc_ids)
    mask_np = np.asarray(mask, dtype=bool)

    if query_doc_ids_np.shape != mask_np.shape:
        raise ValueError(
            "query_doc_ids and mask must have the same shape for shared-doc mapping. "
            f"Got {query_doc_ids_np.shape} vs {mask_np.shape}"
        )

    n_rows, n_pos = mask_np.shape
    query_values, query_group_idx = np.unique(query_np, return_inverse=True)

    row_idx = np.repeat(np.arange(n_rows, dtype=np.int32), n_pos)
    flat_doc_ids = query_doc_ids_np.reshape(-1).astype(np.int64, copy=False)
    flat_valid = mask_np.reshape(-1) & (flat_doc_ids >= 0)
    valid_flat_idx = np.flatnonzero(flat_valid)
    if valid_flat_idx.size == 0:
        raise ValueError("No valid documents found while building shared-doc mapping.")

    flat_query_group = query_group_idx[row_idx]
    pair_dtype = np.dtype([("q", np.int64), ("d", np.int64)])
    pair_keys = np.empty(valid_flat_idx.shape[0], dtype=pair_dtype)
    pair_keys["q"] = flat_query_group[flat_valid].astype(np.int64, copy=False)
    pair_keys["d"] = flat_doc_ids[flat_valid]

    unique_pairs, inverse = np.unique(pair_keys, return_inverse=True)
    query_index_per_doc = unique_pairs["q"].astype(np.int32, copy=False)
    n_query_groups = int(query_values.shape[0])
    counts = np.bincount(query_index_per_doc, minlength=n_query_groups).astype(np.int32, copy=False)
    doclist_ranges = np.concatenate(
        [np.array([0], dtype=np.int32), np.cumsum(counts, dtype=np.int32)]
    )

    doc_id_map_flat = np.full((n_rows * n_pos,), -1, dtype=np.int32)
    doc_id_map_flat[valid_flat_idx] = inverse.astype(np.int32, copy=False)
    doc_id_map = doc_id_map_flat.reshape(n_rows, n_pos)

    return (
        jnp.asarray(doclist_ranges),
        jnp.asarray(doc_id_map),
        jnp.asarray(query_group_idx, dtype=jnp.int32),
        jnp.asarray(query_index_per_doc, dtype=jnp.int32),
        jnp.asarray(query_values),
    )


def _infer_positions_are_ranks_jax(
    positions: jnp.ndarray,
    *,
    n_positions: int,
    sample_size: int = 2000,
    rng_key: Optional[jax.Array] = None,
) -> bool:
    if positions.ndim != 2 or positions.size == 0:
        return False

    n_rows = positions.shape[0]
    if rng_key is None:
        rng_key = jax.random.PRNGKey(0)

    if n_rows > sample_size:
        idx = jax.random.choice(rng_key, n_rows, shape=(sample_size,), replace=False)
        sample = positions[idx]
    else:
        sample = positions

    rank_row_0 = jnp.arange(n_positions)
    rank_row_1 = jnp.arange(1, n_positions + 1)
    match_0 = jnp.mean(jnp.all(sample == rank_row_0, axis=1))
    match_1 = jnp.mean(jnp.all(sample == rank_row_1, axis=1))

    return bool(jnp.maximum(match_0, match_1) >= 0.9)


def _compute_query_freq(
    *,
    use_query_doc_id_mapping: bool,
    query_out: jnp.ndarray,
    query_group_idx_per_row: jnp.ndarray,
    sessions: jnp.ndarray,
    doclist_ranges: jnp.ndarray,
) -> jnp.ndarray:
    if use_query_doc_id_mapping:
        n_queries = int(query_out.shape[0])
        sampled_query_groups = query_group_idx_per_row[sessions]
        return jnp.bincount(sampled_query_groups, length=n_queries).astype(jnp.int32)

    n_queries = int(doclist_ranges.shape[0] - 1)
    return jnp.bincount(sessions, length=n_queries).astype(jnp.int32)


def _prepare_aggregation_inputs(
    rating_dataset: RatingDataset,
    *,
    use_query_doc_id_mapping: bool,
) -> _AggregationInputs:
    query = jnp.asarray(rating_dataset.query)
    query_doc_ids = jnp.asarray(rating_dataset.query_doc_ids)
    query_doc_features = jnp.asarray(rating_dataset.query_doc_features)
    has_separate_lp_query_doc_features = rating_dataset.has_separate_lp_query_doc_features
    lp_query_doc_features = (
        jnp.asarray(rating_dataset.lp_query_doc_features)
        if has_separate_lp_query_doc_features
        else None
    )
    labels = jnp.asarray(rating_dataset.labels)
    mask = jnp.asarray(rating_dataset.mask)
    query_sampling_state = build_query_sampling_state(
        query,
        use_query_doc_id_mapping=use_query_doc_id_mapping,
    )
    query_group_idx_per_row = query_sampling_state.query_group_idx_per_row
    query_out = query_sampling_state.query_out
    query_index_per_doc = None
    if use_query_doc_id_mapping:
        (
            doclist_ranges,
            doc_id_map,
            _query_group_idx_per_row_unused,
            query_index_per_doc,
            _query_values_per_group_unused,
        ) = _build_doc_index_mapping_shared(query, query_doc_ids, mask)
    else:
        doclist_ranges, doc_id_map = _build_doc_index_mapping(mask)
    total_docs = int(doclist_ranges[-1])

    flat_doc_ids = doc_id_map.reshape(-1)
    valid_flat = flat_doc_ids >= 0

    flat_features = query_doc_features.reshape(-1, query_doc_features.shape[2])
    flat_labels = labels.reshape(-1)
    flat_query_doc_ids = query_doc_ids.reshape(-1)

    valid_doc_ids = flat_doc_ids[valid_flat]
    valid_features = flat_features[valid_flat]
    valid_labels = flat_labels[valid_flat]
    valid_query_doc_ids = flat_query_doc_ids[valid_flat]

    feature_matrix = jnp.zeros((total_docs, query_doc_features.shape[2]), dtype=query_doc_features.dtype)
    lp_feature_matrix = None
    label_vector = jnp.zeros((total_docs,), dtype=labels.dtype)
    doc_id_vector = jnp.zeros((total_docs,), dtype=query_doc_ids.dtype)

    feature_matrix = feature_matrix.at[valid_doc_ids].set(valid_features)
    if has_separate_lp_query_doc_features:
        assert lp_query_doc_features is not None
        flat_lp_features = lp_query_doc_features.reshape(-1, lp_query_doc_features.shape[2])
        valid_lp_features = flat_lp_features[valid_flat]
        lp_feature_matrix = jnp.zeros((total_docs, lp_query_doc_features.shape[2]), dtype=lp_query_doc_features.dtype)
        lp_feature_matrix = lp_feature_matrix.at[valid_doc_ids].set(valid_lp_features)
    label_vector = label_vector.at[valid_doc_ids].set(valid_labels)
    doc_id_vector = doc_id_vector.at[valid_doc_ids].set(valid_query_doc_ids)

    return _AggregationInputs(
        query=query,
        query_out=query_out,
        query_group_idx_per_row=query_group_idx_per_row,
        doclist_ranges=doclist_ranges,
        doc_id_map=doc_id_map,
        query_index_per_doc=query_index_per_doc,
        feature_matrix=feature_matrix,
        lp_feature_matrix=lp_feature_matrix,
        label_vector=label_vector,
        doc_id_vector=doc_id_vector,
    )


def _aggregate_click_dataset(
    *,
    rating_dataset: RatingDataset,
    click_dataset: ClickDataset,
    n_sessions: Optional[int],
    cutoff: Optional[int],
    positions_are_ranks: Optional[bool],
    display_histogram_max: int,
    display_histogram_path: Optional[str],
    rng_key: Optional[jax.Array],
    use_query_doc_id_mapping: bool = False,
    debug: bool = False,
) -> AggregatedClickDataset:
    base = _prepare_aggregation_inputs(
        rating_dataset,
        use_query_doc_id_mapping=use_query_doc_id_mapping,
    )
    total_docs = int(base.doclist_ranges[-1])

    sessions_all = jnp.asarray(click_dataset.sessions)
    positions_all = jnp.asarray(click_dataset.positions)
    clicks_all = jnp.asarray(click_dataset.clicks)

    total_sessions = int(sessions_all.shape[0])
    sample_idx, sampling_mode = sample_session_indices(
        total_sessions=total_sessions,
        n_sessions=n_sessions,
        use_query_doc_id_mapping=use_query_doc_id_mapping,
        query_group_idx_per_row=base.query_group_idx_per_row,
        query_out=base.query_out,
        rng_key=rng_key,
        debug=debug,
    )

    sessions = sessions_all[sample_idx]
    positions = positions_all[sample_idx]
    session_clicks = clicks_all[sample_idx]

    if cutoff is None:
        cutoff = int(
            min(
                int(base.doc_id_map.shape[1]),
                int(positions.shape[1]),
                int(session_clicks.shape[1]),
            )
        )
    else:
        cutoff = int(cutoff)

    if positions_are_ranks is None:
        positions_are_ranks = _infer_positions_are_ranks_jax(
            positions,
            n_positions=int(positions.shape[1]),
            rng_key=rng_key,
        )
    positions_are_ranks = bool(positions_are_ranks)

    if not positions_are_ranks and positions.size > 0:
        pos_min = int(jnp.min(positions))
        pos_max = int(jnp.max(positions))
        if pos_min >= 1 and pos_max == int(positions.shape[1]):
            positions = positions - 1

    clicks_per_doc = jnp.zeros((total_docs, cutoff), dtype=jnp.int32)
    displays_per_doc = jnp.zeros((total_docs, cutoff), dtype=jnp.int32)

    max_rank = min(cutoff, int(positions.shape[1]), int(session_clicks.shape[1]))
    global_doc = None
    if max_rank > 0 and sessions.size > 0:
        if positions_are_ranks:
            doc_idx = jnp.broadcast_to(
                jnp.arange(max_rank, dtype=jnp.int32)[None, :],
                (sessions.shape[0], max_rank),
            )
        else:
            doc_idx = positions[:, :max_rank]

        valid_pos = (doc_idx >= 0) & (doc_idx < base.doc_id_map.shape[1])
        q_idx = jnp.broadcast_to(sessions[:, None], doc_idx.shape)
        safe_doc_idx = jnp.where(valid_pos, doc_idx, 0)
        global_doc = base.doc_id_map[q_idx, safe_doc_idx]
        global_doc = jnp.where(valid_pos, global_doc, -1)

        valid_doc = global_doc >= 0
        flat_valid = valid_doc.reshape(-1)
        flat_global_doc = global_doc.reshape(-1)
        flat_r = jnp.broadcast_to(
            jnp.arange(max_rank, dtype=jnp.int32)[None, :],
            doc_idx.shape,
        ).reshape(-1)
        flat_clicks = session_clicks[:, :max_rank].reshape(-1)

        flat_global_doc_safe = jnp.where(flat_valid, flat_global_doc, 0)
        flat_display = flat_valid.astype(jnp.int32)
        flat_clicks = flat_clicks * flat_display

        displays_per_doc = displays_per_doc.at[(flat_global_doc_safe, flat_r)].add(flat_display)
        clicks_per_doc = clicks_per_doc.at[(flat_global_doc_safe, flat_r)].add(flat_clicks)
    query_freq = _compute_query_freq(
        use_query_doc_id_mapping=use_query_doc_id_mapping,
        query_out=base.query_out,
        query_group_idx_per_row=base.query_group_idx_per_row,
        sessions=sessions,
        doclist_ranges=base.doclist_ranges,
    )

    clicks_per_doc_total = jnp.sum(clicks_per_doc, axis=1)
    displays_per_doc_total = jnp.sum(displays_per_doc, axis=1)

    display_counts = np.asarray(displays_per_doc_total)
    max_display_log = int(max(display_histogram_max, 0))
    if max_display_log > 0:
        hist = [
            (k, int(np.sum(display_counts == k)))
            for k in range(1, max_display_log + 1)
        ]
        hist_str = ", ".join([f"{k}:{v}" for k, v in hist])
        print(f"Display count histogram (1..{max_display_log}): {hist_str}")

        if display_histogram_path:
            payload = {
                "max_count": max_display_log,
                "total_docs": int(display_counts.shape[0]),
                "histogram": {str(k): v for k, v in hist},
            }
            hist_path = Path(display_histogram_path).expanduser()
            hist_path.parent.mkdir(parents=True, exist_ok=True)
            hist_path.write_text(json.dumps(payload, indent=2))

    if debug:
        print(
            "sampling_mode=%s requested_sessions=%s actual_sessions=%d"
            % (
                sampling_mode,
                "all" if n_sessions is None else int(n_sessions),
                int(sample_idx.shape[0]),
            )
        )
        print(
            "shape of doclist ranges and doc_id_map",
            base.doclist_ranges.shape,
            base.doc_id_map.shape,
        )
        print("shape of session clicks", session_clicks.shape)
        print("shape of doc id map", base.doc_id_map.shape)
        if global_doc is not None:
            print("shape of global doc", global_doc.shape)
        print("shape of clicks_per_doc", clicks_per_doc.shape)
        print("shape of displays_per_doc", displays_per_doc.shape)
        print("shape of total displays_per_doc", displays_per_doc_total.shape)

    return AggregatedClickDataset(
        query=base.query_out,
        feature_matrix=base.feature_matrix,
        lp_feature_matrix=base.lp_feature_matrix,
        label_vector=base.label_vector,
        doc_id_vector=base.doc_id_vector,
        doclist_ranges=base.doclist_ranges,
        doc_id_map=base.doc_id_map,
        query_index_per_doc=base.query_index_per_doc,
        clicks=clicks_per_doc,
        displays=displays_per_doc,
        clicks_per_doc=clicks_per_doc_total,
        displays_per_doc=displays_per_doc_total,
        query_freq=query_freq,
        sessions=sessions,
        cutoff=cutoff,
        metadata={
            "use_query_doc_id_mapping": bool(use_query_doc_id_mapping),
            "has_reconstructible_query_tensors": True,
            "has_separate_lp_feature_matrix": bool(base.lp_feature_matrix is not None),
        },
    )
