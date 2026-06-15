"""Shared helpers for selecting click sessions consistently across pipelines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np


@dataclass(frozen=True)
class QuerySamplingState:
    query_out: jnp.ndarray
    query_group_idx_per_row: jnp.ndarray


def build_query_sampling_state(
    query: np.ndarray | jnp.ndarray,
    *,
    use_query_doc_id_mapping: bool,
) -> QuerySamplingState:
    query = jnp.asarray(query).reshape(-1)
    if not use_query_doc_id_mapping:
        return QuerySamplingState(
            query_out=query,
            query_group_idx_per_row=jnp.arange(query.shape[0], dtype=jnp.int32),
        )

    query_np = np.asarray(query).reshape(-1)
    query_values, query_group_idx = np.unique(query_np, return_inverse=True)
    return QuerySamplingState(
        query_out=jnp.asarray(query_values),
        query_group_idx_per_row=jnp.asarray(query_group_idx, dtype=jnp.int32),
    )


def sample_session_indices(
    *,
    total_sessions: int,
    n_sessions: Optional[int],
    use_query_doc_id_mapping: bool,
    query_group_idx_per_row: jnp.ndarray,
    query_out: jnp.ndarray,
    rng_key: Optional[jax.Array],
    debug: bool,
) -> tuple[jnp.ndarray, str]:
    if n_sessions is None or n_sessions >= total_sessions:
        return jnp.arange(total_sessions, dtype=jnp.int32), "all_sessions"

    if rng_key is None:
        rng_key = jax.random.PRNGKey(0)

    if not use_query_doc_id_mapping:
        perm = jax.random.permutation(rng_key, total_sessions)
        return perm[:n_sessions], "session_uniform"

    query_group_idx_np = np.asarray(query_group_idx_per_row, dtype=np.int32)
    n_query_groups = int(query_out.shape[0])
    group_sizes = np.bincount(query_group_idx_np, minlength=n_query_groups).astype(np.int64, copy=False)
    group_perm = np.asarray(jax.random.permutation(rng_key, n_query_groups), dtype=np.int32)
    group_cumsum = np.cumsum(group_sizes[group_perm], dtype=np.int64)
    keep_groups = int(np.searchsorted(group_cumsum, int(n_sessions), side="right"))
    if keep_groups <= 0:
        keep_groups = 1

    selected_groups = group_perm[:keep_groups]
    keep_query_group = np.zeros((n_query_groups,), dtype=bool)
    keep_query_group[selected_groups] = True
    sample_idx_np = np.flatnonzero(keep_query_group[query_group_idx_np]).astype(np.int32, copy=False)
    sample_idx = jnp.asarray(sample_idx_np, dtype=jnp.int32)

    if debug:
        print(
            "Session sampling (query_family): requested=%d actual=%d selected_queries=%d/%d"
            % (
                int(n_sessions),
                int(sample_idx_np.shape[0]),
                int(keep_groups),
                int(n_query_groups),
            )
        )
    return sample_idx, "query_family"


__all__ = [
    "QuerySamplingState",
    "build_query_sampling_state",
    "sample_session_indices",
]
