from __future__ import annotations

from typing import Optional

import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax import Array

from feature_based_propensities_for_ULTR.models.towers import EmbeddingBiasTower
from .base import BasePropensityModel


class UnsupervisedGroupingPropensity(BasePropensityModel):
    """
    Unsupervised propensity model that groups documents by feature similarity
    globally across queries. Clustering can be k-means or distance-based.
    """

    def __init__(
        self,
        query_doc_features: int,
        positions: int,
        num_groups: Optional[int] = None,
        kmeans_iters: int = 5,
        method: str = "kmeans",
        knn_k: int = 50,
        cosine_threshold: float = 1.0,
        euclidean_threshold: float = np.inf,
        min_cluster_size: int = 2,
        deterministic_groups: bool = False,
        *,
        rngs: nnx.Rngs,
        bias_tower: EmbeddingBiasTower = None,
        bias_value_array: Array = None,
        **kwargs,
    ):
        super().__init__()

        self.positions = positions
        self.method = str(method).lower()
        self.knn_k = int(knn_k)
        self.cosine_threshold = float(cosine_threshold)
        self.euclidean_threshold = float(euclidean_threshold)
        self.min_cluster_size = int(min_cluster_size)
        self.deterministic_groups = bool(deterministic_groups)
        if self.min_cluster_size < 1:
            raise ValueError("min_cluster_size must be >= 1")
        self.num_groups = int(num_groups) if num_groups is not None else max(2, positions // 2)
        self.kmeans_iters = int(kmeans_iters)

        if bias_tower is not None:
            bias_tower_output = bias_tower({"positions": jnp.arange(positions)}).squeeze()
            full_bias = bias_tower_output - bias_tower_output[0]
        elif bias_value_array is not None:
            full_bias = jnp.asarray(bias_value_array)
        else:
            full_bias = jnp.zeros((positions,), dtype=jnp.float32)

        if full_bias.shape[0] != self.num_groups:
            idx = jnp.round(jnp.linspace(0, full_bias.shape[0] - 1, self.num_groups)).astype(jnp.int32)
            group_bias = full_bias[idx]
        else:
            group_bias = full_bias

        self.group_bias_values = nnx.Variable(
            group_bias,
            collection="constants",
        )

    def assign_groups_global(
        self,
        features: Array,
        mask: Array,
        doc_ids: Array | None = None,
    ) -> Array:
        features_np = jnp.asarray(features)
        mask_np = jnp.asarray(mask).astype(bool)
        group_ids = -jnp.ones(mask_np.shape, dtype=jnp.int32)

        flat_mask = np.asarray(mask_np).reshape(-1)
        valid_idx = np.where(flat_mask)[0]
        if valid_idx.size == 0:
            return group_ids

        flat_features = np.asarray(features_np).reshape((-1, features_np.shape[-1]))
        valid_features = flat_features[valid_idx]

        if self.deterministic_groups and doc_ids is not None:
            flat_doc_ids = np.asarray(jnp.asarray(doc_ids)).reshape(-1)
            doc_ids_valid = flat_doc_ids[valid_idx]
            order = np.argsort(doc_ids_valid, kind="stable")
            inv_order = np.empty_like(order)
            inv_order[order] = np.arange(order.size)
            valid_features = valid_features[order]
        else:
            inv_order = None

        if self.method == "kmeans":
            assignments = self._kmeans_assign_numpy(valid_features)
        elif self.method == "cosine":
            assignments = self._cosine_assign_numpy(valid_features)
        elif self.method == "euclidean":
            assignments = self._euclidean_assign_numpy(valid_features)
        elif self.method == "knn":
            assignments = self._knn_assign_numpy(
                valid_features,
                apply_threshold=False,
                metric="cosine",
                threshold=self.cosine_threshold,
            )
        else:
            raise ValueError(f"Unknown grouping method: {self.method}")

        if inv_order is not None:
            assignments = np.asarray(assignments)[inv_order]

        group_ids_flat = -np.ones((flat_mask.size,), dtype=np.int32)
        group_ids_flat[valid_idx] = np.asarray(assignments, dtype=np.int32)
        return jnp.asarray(group_ids_flat.reshape(mask_np.shape), dtype=jnp.int32)

    def _kmeans_assign_numpy(self, features: Array) -> Array:
        feats = np.asarray(jnp.asarray(features), dtype=np.float32)
        n_docs = int(feats.shape[0])
        if n_docs == 0:
            return jnp.asarray([], dtype=jnp.int32)

        k_requested = int(self.num_groups)
        k = min(max(1, k_requested), n_docs)

        sample_multiplier = 10
        min_sample = 1000
        sample_size = min(n_docs, max(k * sample_multiplier, min_sample))
        use_subsample = sample_size < n_docs
        train_iters = min(max(int(self.kmeans_iters), 1), 5) if use_subsample else max(int(self.kmeans_iters), 1)

        if use_subsample:
            rng = np.random.default_rng(0)
            sample_idx = rng.choice(n_docs, size=sample_size, replace=False)
            feats_train = feats[sample_idx]
        else:
            feats_train = feats

        centers = self._init_kmeans_centers_numpy(feats_train, k)
        assignments = np.full((feats_train.shape[0],), -1, dtype=np.int64)

        for it in range(train_iters):
            dist = self._pairwise_sq_dist_numpy(feats_train, centers)
            new_assignments = np.argmin(dist, axis=1)
            if np.array_equal(new_assignments, assignments):
                break
            assignments = new_assignments
            centers = self._recompute_centers_numpy(feats_train, assignments, centers, dist)

        if use_subsample:
            assignments = self._assign_clusters_chunked_numpy(feats, centers)

        if self.min_cluster_size > 1 and k > 1:
            assignments = self._merge_small_clusters_numpy(
                feats,
                assignments,
                min_size=int(self.min_cluster_size),
            )

        unique = np.unique(assignments)
        assignments = np.searchsorted(unique, assignments)
        return jnp.asarray(assignments, dtype=jnp.int32)

    @staticmethod
    def _pairwise_sq_dist_numpy(feats: np.ndarray, centers: np.ndarray) -> np.ndarray:
        diffs = feats[:, None, :] - centers[None, :, :]
        return np.sum(diffs * diffs, axis=-1)

    @staticmethod
    def _assign_clusters_chunked_numpy(
        feats: np.ndarray,
        centers: np.ndarray,
        *,
        chunk_size: int = 10000,
    ) -> np.ndarray:
        n_docs = feats.shape[0]
        assignments = np.empty((n_docs,), dtype=np.int64)
        for start in range(0, n_docs, chunk_size):
            end = min(start + chunk_size, n_docs)
            dist = UnsupervisedGroupingPropensity._pairwise_sq_dist_numpy(
                feats[start:end],
                centers,
            )
            assignments[start:end] = np.argmin(dist, axis=1)
        return assignments

    @staticmethod
    def _init_kmeans_centers_numpy(feats: np.ndarray, k: int) -> np.ndarray:
        n_docs, n_dim = feats.shape
        centers = np.empty((k, n_dim), dtype=feats.dtype)
        centers[0] = feats[0]
        selected = np.zeros((n_docs,), dtype=bool)
        selected[0] = True
        closest = np.sum((feats - centers[0]) ** 2, axis=1)
        for i in range(1, k):
            scores = np.where(selected, -np.inf, closest)
            if np.all(~np.isfinite(scores)):
                idx = int(np.argmax(~selected))
            else:
                idx = int(np.argmax(scores))
            centers[i] = feats[idx]
            selected[idx] = True
            dist = np.sum((feats - centers[i]) ** 2, axis=1)
            closest = np.minimum(closest, dist)
        return centers

    @staticmethod
    def _recompute_centers_numpy(
        feats: np.ndarray,
        assignments: np.ndarray,
        centers: np.ndarray,
        dist: np.ndarray,
    ) -> np.ndarray:
        k = centers.shape[0]
        for i in range(k):
            mask = assignments == i
            if np.any(mask):
                centers[i] = feats[mask].mean(axis=0)
            else:
                farthest = int(np.argmax(np.min(dist, axis=1)))
                centers[i] = feats[farthest]
                assignments[farthest] = i
        return centers

    @staticmethod
    def _merge_small_clusters_numpy(
        feats: np.ndarray,
        assignments: np.ndarray,
        *,
        min_size: int,
    ) -> np.ndarray:
        max_passes = max(int(np.unique(assignments).size), 1)
        for _ in range(max_passes):
            unique, counts = np.unique(assignments, return_counts=True)
            if unique.size <= 1:
                break
            small = unique[counts < min_size]
            if small.size == 0:
                break
            large = unique[counts >= min_size]
            if large.size == 0:
                assignments[:] = unique[0]
                break

            centers = np.zeros((unique.size, feats.shape[1]), dtype=feats.dtype)
            id_to_idx = {cid: idx for idx, cid in enumerate(unique)}
            for cid in unique:
                centers[id_to_idx[cid]] = feats[assignments == cid].mean(axis=0)

            large_idx = np.array([id_to_idx[cid] for cid in large], dtype=np.int64)
            for cid in small:
                src_idx = id_to_idx[cid]
                diffs = centers[large_idx] - centers[src_idx]
                dists = np.sum(diffs * diffs, axis=1)
                target = large[int(np.argmin(dists))]
                assignments[assignments == cid] = target
        return assignments

    def _cosine_assign_numpy(self, features: Array) -> Array:
        return self._knn_assign_numpy(
            features,
            apply_threshold=True,
            metric="cosine",
            threshold=self.cosine_threshold,
        )

    def _euclidean_assign_numpy(self, features: Array) -> Array:
        return self._knn_assign_numpy(
            features,
            apply_threshold=True,
            metric="euclidean",
            threshold=self.euclidean_threshold,
        )

    def _knn_assign_numpy(
        self,
        features: Array,
        *,
        apply_threshold: bool,
        metric: str,
        threshold: float,
    ) -> Array:
        feats = np.asarray(jnp.asarray(features), dtype=np.float32)
        n_docs = int(feats.shape[0])
        if n_docs == 0:
            return jnp.asarray([], dtype=jnp.int32)

        if n_docs >= 50000:
            print(f"[knn] computing neighbors: n_docs={n_docs}, k={self.knn_k}")

        metric = str(metric).lower()
        if metric == "cosine":
            norms = np.linalg.norm(feats, axis=1, keepdims=True)
            feats = feats / np.maximum(norms, 1e-8)
        elif metric != "euclidean":
            raise ValueError(f"Unknown neighbor metric: {metric}")

        k = min(max(int(self.knn_k), 1), n_docs)
        from sklearn.neighbors import NearestNeighbors

        # Exact cosine neighbors use normalized vectors and Euclidean distance.
        nn = NearestNeighbors(
            n_neighbors=k,
            metric="euclidean",
            algorithm="brute",
            n_jobs=-1,
        )
        nn.fit(feats)
        distances, indices = nn.kneighbors(feats, return_distance=True)

        if apply_threshold:
            distances = distances.astype(np.float32, copy=False)
            if metric == "cosine":
                cosine_sim = 1.0 - 0.5 * np.square(distances)
                cosine_sim = np.clip(cosine_sim, -1.0, 1.0)
                keep = cosine_sim >= float(threshold)
            else:
                keep = distances <= float(threshold)
            indices = np.where(keep, indices, -1)

        rows = np.repeat(np.arange(n_docs, dtype=np.int64), indices.shape[1])
        cols = indices.reshape(-1).astype(np.int64, copy=False)
        valid = (cols >= 0) & (cols < n_docs) & (rows != cols)
        if not np.any(valid):
            return jnp.arange(n_docs, dtype=jnp.int32)

        rows = rows[valid]
        cols = cols[valid]

        labels = self._connected_components_numpy(n_docs, rows, cols)
        return jnp.asarray(labels, dtype=jnp.int32).ravel()

    @staticmethod
    def _connected_components_numpy(
        n_nodes: int,
        rows: np.ndarray,
        cols: np.ndarray,
    ) -> np.ndarray:
        from scipy.sparse import coo_matrix  # type: ignore
        from scipy.sparse.csgraph import connected_components  # type: ignore

        data = np.ones(rows.shape[0], dtype=np.uint8)
        graph = coo_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes))
        _, labels = connected_components(graph, directed=False, connection="weak")
        _, relabeled = np.unique(labels, return_inverse=True)
        return relabeled.astype(np.int32)

    def __call__(self, batch: dict) -> Array:
        return self.compute_output(batch)

    def compute_loss(self, output) -> Array:
        return jnp.asarray(0.0, dtype=getattr(output, "dtype", jnp.float32))

    def compute_output(self, batch: dict) -> Array:
        features = batch["query_doc_features"]
        if features.ndim == 2:
            features = features[None, ...]
        mask = batch.get("mask", jnp.ones(features.shape[:2], dtype=bool))
        if mask.ndim == 1:
            mask = mask[None, ...]
        doc_ids = batch.get("doc_ids", None)
        if doc_ids is not None and doc_ids.ndim == 1:
            doc_ids = doc_ids[None, ...]
        return self.assign_groups_global(features, mask, doc_ids=doc_ids)

    @property
    def requires_alpha(self) -> bool:
        return True

    @property
    def uses_global_groups(self) -> bool:
        return True
