"""Dataset wrappers for simulated click data."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import jax.numpy as jnp
import numpy as np
from torch.utils.data import Dataset

from feature_based_propensities_for_ULTR.data.base import RatingDataset
from feature_based_propensities_for_ULTR.data.collate import stack_dict_collate


class ClickDataset(Dataset):
    def __init__(
        self,
        rating_dataset: RatingDataset,
        sessions: np.ndarray,
        clicks: np.ndarray,
        positions: np.ndarray,
        sessions_per_query: np.ndarray,
        sessions_per_doc_pos: np.ndarray,
    ):
        self.sessions = np.asarray(sessions, dtype=np.int32)
        self.clicks = np.asarray(clicks, dtype=np.int8)
        self.positions = np.asarray(positions, dtype=np.int32)
        self.query = rating_dataset.query
        self.query_doc_features = rating_dataset.query_doc_features
        self.query_doc_ids = rating_dataset.query_doc_ids
        self._lp_query_doc_features = (
            np.asarray(rating_dataset.lp_query_doc_features)
            if rating_dataset.has_separate_lp_query_doc_features
            else None
        )
        self.labels = rating_dataset.labels
        self.mask = rating_dataset.mask
        self.n = rating_dataset.n
        self.positions_are_dense_identity = self._positions_are_dense_identity(self.positions)
        self.positions_are_one_based = self._positions_are_one_based(self.positions)
        self.positions_are_zero_based = self.positions_are_dense_identity and not self.positions_are_one_based
        self.sessions_per_query = np.asarray(sessions_per_query, dtype=np.int32)
        (
            self._sessions_per_doc_pos_diag,
            self._sessions_per_doc_pos_full,
        ) = self._prepare_sessions_per_doc_pos_storage(
            sessions_per_doc_pos,
            compact=self.positions_are_dense_identity,
        )
        self.has_diagonal_only_sessions_per_doc_pos = self._sessions_per_doc_pos_diag is not None
        self._identity_positions_cache = None
        if self.positions_are_dense_identity:
            width = int(self.positions.shape[1]) if self.positions.ndim == 2 else 0
            self._identity_positions_cache = np.arange(width, dtype=np.int64)

    @staticmethod
    def _prepare_sessions_per_doc_pos_storage(
        sessions_per_doc_pos: np.ndarray,
        *,
        compact: bool,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        counts = np.asarray(sessions_per_doc_pos)
        if counts.ndim == 2:
            return counts.astype(np.int32, copy=False), None
        if counts.ndim != 3:
            raise ValueError(
                "sessions_per_doc_pos must have shape (N, P) or (N, P, P); "
                f"got {counts.shape}."
            )
        counts = counts.astype(np.int32, copy=False)
        if compact:
            diag_idx = np.arange(counts.shape[1], dtype=np.int64)
            diag = counts[:, diag_idx, diag_idx].astype(np.int32, copy=False)
            return diag, None
        return None, counts

    @staticmethod
    def _positions_are_dense_identity(positions: np.ndarray) -> bool:
        if positions.ndim != 2 or positions.size == 0:
            return False
        width = int(positions.shape[1])
        zero_based = np.broadcast_to(
            np.arange(width, dtype=positions.dtype)[None, :],
            positions.shape,
        )
        if np.array_equal(positions, zero_based):
            return True
        one_based = np.broadcast_to(
            np.arange(1, width + 1, dtype=positions.dtype)[None, :],
            positions.shape,
        )
        return bool(np.array_equal(positions, one_based))

    @staticmethod
    def _positions_are_one_based(positions: np.ndarray) -> bool:
        if positions.ndim != 2 or positions.size == 0:
            return False
        width = int(positions.shape[1])
        one_based = np.broadcast_to(
            np.arange(1, width + 1, dtype=positions.dtype)[None, :],
            positions.shape,
        )
        return bool(np.array_equal(positions, one_based))

    def _normalize_position_indices(self, positions: np.ndarray) -> np.ndarray:
        """
        Normalize position arrays to zero-based gather indices.

        Prebuilt artifacts may use 1-based dense ranks; simulated data is usually
        already zero-based. This helper keeps both formats compatible.
        """
        pos = np.asarray(positions, dtype=np.int64)
        if pos.size == 0:
            return pos

        n_docs = int(self.query_doc_features.shape[1])
        pos_min = int(np.min(pos))
        pos_max = int(np.max(pos))
        if pos_min >= 1 and pos_max <= n_docs:
            return pos - 1
        return pos

    def __getitem__(self, idx):
        session_idx = self.sessions[idx]
        if self.positions_are_dense_identity and self._identity_positions_cache is not None:
            position_idx = self._identity_positions_cache
            mask = self.mask[session_idx]
            query_doc_features = self.query_doc_features[session_idx]
            lp_query_doc_features = self.lp_query_doc_features[session_idx]
            query_doc_ids = self.query_doc_ids[session_idx]
            labels = self.labels[session_idx]
            if self._sessions_per_doc_pos_diag is None:
                raise RuntimeError("Dense-identity dataset is missing compact count storage.")
            sessions_per_doc_pos = self._sessions_per_doc_pos_diag[session_idx]
        else:
            position_raw = self.positions[idx]
            position_idx = self._normalize_position_indices(position_raw)
            doc_idx = np.arange(len(position_idx), dtype=np.int64)
            mask = self.mask[session_idx][position_idx]
            safe_position_idx = np.where(mask, position_idx, 0)
            query_doc_features = self.query_doc_features[session_idx][safe_position_idx]
            lp_query_doc_features = self.lp_query_doc_features[session_idx][safe_position_idx]
            query_doc_ids = self.query_doc_ids[session_idx][safe_position_idx]
            labels = self.labels[session_idx][safe_position_idx]
            if self._sessions_per_doc_pos_full is not None:
                sessions_per_doc_pos = self._sessions_per_doc_pos_full[
                    session_idx, doc_idx, safe_position_idx
                ]
            elif self._sessions_per_doc_pos_diag is not None:
                sessions_per_doc_pos = self._sessions_per_doc_pos_diag[
                    session_idx, safe_position_idx
                ]
            else:
                raise RuntimeError("ClickDataset is missing per-position session counts.")

        # Clear padded slots so masked items contribute no features, ids, or labels.
        query_doc_features = np.where(mask[:, None], query_doc_features, 0.0)
        lp_query_doc_features = np.where(mask[:, None], lp_query_doc_features, 0.0)
        query_doc_ids = np.where(mask, query_doc_ids, -1)
        labels = np.where(mask, labels, 0.0)

        sessions_per_query = max(int(self.sessions_per_query[session_idx]), 1)
        with np.errstate(divide="ignore", invalid="ignore"):
            propensities = sessions_per_doc_pos / sessions_per_query
        # Missing/padded docs are masked from loss. Force safe weights there.
        propensities = np.where(mask, propensities, 1.0).astype(np.float32)

        return {
            "query": self.query[session_idx],
            "query_doc_features": query_doc_features,
            "lp_query_doc_features": lp_query_doc_features,
            "query_doc_ids": query_doc_ids,
            "labels": labels,
            "propensities": propensities,
            "clicks": self.clicks[idx],
            "positions": np.arange(len(position_idx), dtype=np.int64),
            "mask": mask,
            "n": self.n[session_idx],
        }

    def __len__(self):
        return len(self.sessions)

    @property
    def n_queries(self) -> int:
        return int(self.query.shape[0])

    @property
    def n_positions(self) -> int:
        return int(self.positions.shape[1])

    @property
    def n_features(self) -> int:
        return int(self.query_doc_features.shape[2])

    @property
    def n_logging_policy_features(self) -> int:
        return int(self.lp_query_doc_features.shape[2])

    @property
    def lp_query_doc_features(self) -> np.ndarray:
        if self._lp_query_doc_features is None:
            return self.query_doc_features
        return self._lp_query_doc_features

    @property
    def sessions_per_doc_pos_diag(self) -> np.ndarray:
        if self._sessions_per_doc_pos_diag is not None:
            return self._sessions_per_doc_pos_diag
        if self._sessions_per_doc_pos_full is None:
            raise RuntimeError("ClickDataset is missing per-position session counts.")
        diag = np.arange(self._sessions_per_doc_pos_full.shape[1], dtype=np.int64)
        return self._sessions_per_doc_pos_full[:, diag, diag]

    @property
    def sessions_per_doc_pos(self) -> np.ndarray:
        if self._sessions_per_doc_pos_full is None:
            diag_counts = self.sessions_per_doc_pos_diag
            n_sessions, width = diag_counts.shape
            full = np.zeros((n_sessions, width, width), dtype=diag_counts.dtype)
            diag = np.arange(width, dtype=np.int64)
            full[:, diag, diag] = diag_counts
            self._sessions_per_doc_pos_full = full
        return self._sessions_per_doc_pos_full

    @property
    def has_separate_lp_query_doc_features(self) -> bool:
        return self._lp_query_doc_features is not None

    def sample_features(
        self, n_samples: int, random_state: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed=random_state)
        sample_idx = rng.choice(len(self), size=n_samples)
        session_idx = self.sessions[sample_idx]
        if self.positions_are_dense_identity:
            width = int(self.positions.shape[1])
            position_idx = np.broadcast_to(
                np.arange(width, dtype=np.int64)[None, :],
                (len(sample_idx), width),
            )
        else:
            position_idx = self._normalize_position_indices(self.positions[sample_idx])

        x = self.query_doc_features[session_idx]
        mask = self.mask[session_idx]

        x = np.take_along_axis(x, position_idx[:, :, None], axis=1)
        mask = np.take_along_axis(mask, position_idx, axis=1)
        x = np.where(mask[:, :, None], x, 0.0)

        return x, mask

    def save_to_npz(self, file_path: str):
        """
        Save the dataset to a .npz file.
        """
        from .serialization import save_click_dataset_npz

        save_click_dataset_npz(self, file_path)

    @staticmethod
    def collate_fn(batch):
        return stack_dict_collate(batch)


class AggregatedClickDataset:
    """
    Aggregated, document-level dataset built from a ClickDataset + RatingDataset.

    This mirrors the DataSplitAdapter API while storing a compact document-level
    representation. Query-shaped tensors are reconstructed lazily when needed.
    """

    def __init__(
        self,
        *,
        query: jnp.ndarray,
        feature_matrix: jnp.ndarray,
        lp_feature_matrix: jnp.ndarray | None,
        label_vector: jnp.ndarray,
        doc_id_vector: jnp.ndarray,
        doclist_ranges: jnp.ndarray,
        doc_id_map: jnp.ndarray,
        query_index_per_doc: jnp.ndarray | None = None,
        clicks: jnp.ndarray,
        displays: jnp.ndarray,
        clicks_per_doc: jnp.ndarray,
        displays_per_doc: jnp.ndarray,
        query_freq: jnp.ndarray,
        sessions: jnp.ndarray,
        cutoff: int,
        metadata: Optional[Dict] = None,
        query_doc_ids: jnp.ndarray | None = None,
        query_doc_features: jnp.ndarray | None = None,
        lp_query_doc_features: jnp.ndarray | None = None,
        labels: jnp.ndarray | None = None,
        mask: jnp.ndarray | None = None,
        n: jnp.ndarray | None = None,
    ):
        self.query = query
        self._query_doc_ids = query_doc_ids
        self._query_doc_features = query_doc_features
        self._lp_query_doc_features = lp_query_doc_features
        self._labels = labels
        self._mask = mask
        self._n = n

        self.feature_matrix = feature_matrix
        self._lp_feature_matrix = lp_feature_matrix
        self.label_vector = label_vector
        self.doc_id_vector = doc_id_vector
        self.doclist_ranges = doclist_ranges
        self.doc_id_map = doc_id_map
        self.query_index_per_doc = query_index_per_doc

        self.clicks = clicks
        self.displays = displays
        self.clicks_per_doc = clicks_per_doc
        self.displays_per_doc = displays_per_doc

        self.query_freq = query_freq
        self.sessions = sessions
        self.cutoff = int(cutoff)
        self.metadata = metadata or {}

        self._query_doc_gather_idx = None
        self._query_index_per_document_vector = None

    def __len__(self) -> int:
        return self.num_docs()

    @property
    def lp_query_doc_features(self):
        if self._lp_query_doc_features is None:
            source = self.lp_feature_matrix
            if source is self.feature_matrix:
                return self.query_doc_features
            self._lp_query_doc_features = self._materialize_query_feature_tensor(source)
        return self._lp_query_doc_features

    @property
    def has_separate_lp_query_doc_features(self) -> bool:
        return self._lp_feature_matrix is not None

    @property
    def lp_feature_matrix(self):
        if self._lp_feature_matrix is None:
            return self.feature_matrix
        return self._lp_feature_matrix

    @property
    def has_separate_lp_feature_matrix(self) -> bool:
        return self._lp_feature_matrix is not None

    @property
    def query_doc_ids(self):
        if self._query_doc_ids is None:
            self._query_doc_ids = self._materialize_query_vector_tensor(
                self.doc_id_vector,
                fill_value=-1,
            )
        return self._query_doc_ids

    @property
    def query_doc_features(self):
        if self._query_doc_features is None:
            self._query_doc_features = self._materialize_query_feature_tensor(self.feature_matrix)
        return self._query_doc_features

    @property
    def labels(self):
        if self._labels is None:
            self._labels = self._materialize_query_vector_tensor(self.label_vector, fill_value=0)
        return self._labels

    @property
    def mask(self):
        if self._mask is None:
            self._mask = jnp.asarray(self.doc_id_map >= 0)
        return self._mask

    @property
    def n(self):
        if self._n is None:
            self._n = jnp.sum(self.mask.astype(jnp.int32), axis=1)
        return self._n

    def __getitem__(self, idx: int):
        q_idx = self.query_index_per_document()[idx]
        return {
            "query": self.query[q_idx],
            "query_index": q_idx,
            "query_doc_ids": self.doc_id_vector[idx],
            "query_doc_features": self.feature_matrix[idx],
            "lp_query_doc_features": self.lp_feature_matrix[idx],
            "labels": self.label_vector[idx],
            "clicks": self.clicks[idx],
            "displays": self.displays[idx],
            "clicks_per_doc": self.clicks_per_doc[idx],
            "displays_per_doc": self.displays_per_doc[idx],
        }

    def save_to_npz(self, file_path: str, *, extra_metadata: Optional[Dict] = None) -> None:
        """
        Save the aggregated dataset to a .npz file.
        """
        from .serialization import save_aggregated_click_dataset_npz

        save_aggregated_click_dataset_npz(
            self,
            file_path,
            extra_metadata=extra_metadata,
        )

    @staticmethod
    def load_from_npz(file_path: str) -> "AggregatedClickDataset":
        from .serialization import load_aggregated_click_dataset_npz

        return load_aggregated_click_dataset_npz(file_path)

    @staticmethod
    def collate_fn(batch):
        keys = batch[0].keys()
        return {key: np.stack([sample[key] for sample in batch]) for key in keys}

    def num_queries(self) -> int:
        return int(self.doclist_ranges.shape[0] - 1)

    def num_docs(self) -> int:
        return int(self.feature_matrix.shape[0])

    def _query_doc_gather_indices(self) -> jnp.ndarray:
        if self._query_doc_gather_idx is None:
            n_docs = self.num_docs()
            self._query_doc_gather_idx = jnp.where(
                jnp.asarray(self.doc_id_map) >= 0,
                jnp.asarray(self.doc_id_map),
                jnp.full_like(jnp.asarray(self.doc_id_map), n_docs),
            )
        return self._query_doc_gather_idx

    def _materialize_query_feature_tensor(self, source: jnp.ndarray) -> jnp.ndarray:
        source = jnp.asarray(source)
        pad = jnp.zeros((1, source.shape[1]), dtype=source.dtype)
        gathered = jnp.concatenate([source, pad], axis=0)[self._query_doc_gather_indices()]
        return jnp.where(self.mask[..., None], gathered, 0.0)

    def _materialize_query_vector_tensor(self, source: jnp.ndarray, *, fill_value) -> jnp.ndarray:
        source = jnp.asarray(source)
        fill = jnp.asarray([fill_value], dtype=source.dtype)
        gathered = jnp.concatenate([source, fill], axis=0)[self._query_doc_gather_indices()]
        return jnp.where(self.mask, gathered, fill_value)

    def query_index_per_document(self) -> jnp.ndarray:
        if self.query_index_per_doc is not None:
            return jnp.asarray(self.query_index_per_doc, dtype=jnp.int32)
        if self._query_index_per_document_vector is None:
            q_idx = jnp.zeros(self.num_docs(), dtype=jnp.int32)
            if self.num_queries() > 1 and self.num_docs() > 0:
                start_idx = self.doclist_ranges[1:-1]
                start_idx = start_idx[start_idx < self.num_docs()]
                q_idx = q_idx.at[start_idx].set(1)
            q_idx = jnp.cumsum(q_idx, dtype=jnp.int32)
            self._query_index_per_document_vector = q_idx
        return self._query_index_per_document_vector

    def query_range(self, query_index: int) -> Tuple[int, int]:
        s_i = int(self.doclist_ranges[query_index])
        e_i = int(self.doclist_ranges[query_index + 1])
        return s_i, e_i

    def query_size(self, query_index: int) -> int:
        s_i, e_i = self.query_range(query_index)
        return int(e_i - s_i)

    def query_sizes(self) -> jnp.ndarray:
        return self.doclist_ranges[1:] - self.doclist_ranges[:-1]

    def max_query_size(self) -> int:
        return int(jnp.max(self.query_sizes()))

    def query_values_from_vector(self, qid: int, vector: jnp.ndarray) -> jnp.ndarray:
        s_i, e_i = self.query_range(qid)
        return vector[s_i:e_i]

    def query_feat(self, query_index: int) -> jnp.ndarray:
        s_i, e_i = self.query_range(query_index)
        return self.feature_matrix[s_i:e_i, :]

    def query_labels(self, query_index: int) -> jnp.ndarray:
        s_i, e_i = self.query_range(query_index)
        return self.label_vector[s_i:e_i]
