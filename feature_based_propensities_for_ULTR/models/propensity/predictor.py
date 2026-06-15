import os
import time

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from torch.utils.data import DataLoader

from feature_based_propensities_for_ULTR.models.utils import load_model_params
from feature_based_propensities_for_ULTR.trainer import PropensityTrainer
from .base import BasePropensityModel, PropensityModelSpec
from .mlp import ClassifierPropensityMLP


class PropensityPredictor:
    def __init__(
        self,
        *,
        model: BasePropensityModel,
        alpha: np.ndarray | None = None,
        apply_sigmoid: bool = True,
    ):
        self.model = model
        self.alpha = None if alpha is None else np.asarray(alpha)
        self.apply_sigmoid = apply_sigmoid

    @staticmethod
    def _map_query_predictions_to_docs(data, per_query_doc_pred: np.ndarray) -> np.ndarray:
        doc_id_map = np.asarray(data.doc_id_map)
        n_queries, n_positions = per_query_doc_pred.shape[:2]
        doc_id_map = doc_id_map[:n_queries, :n_positions]
        pred_per_doc = np.zeros(data.num_docs(), dtype=np.float64)
        valid = doc_id_map >= 0
        pred_per_doc[doc_id_map[valid]] = per_query_doc_pred[valid]
        return pred_per_doc

    def _predict_classifier_per_doc(
        self,
        data,
        *,
        batch_size: int = 1024,
        show_progress: bool = True,
    ) -> np.ndarray:
        @nnx.jit
        def forward_step(model, features):
            output = model.compute_output({"query_doc_features": features})
            return jax.nn.sigmoid(output) if self.apply_sigmoid else output

        query_doc_features = np.asarray(data.query_doc_features)
        n_queries = query_doc_features.shape[0]
        per_query_doc_pred = []
        batch_indices = list(range(0, n_queries, batch_size))
        if show_progress:
            from tqdm import tqdm
            batch_indices = tqdm(batch_indices, desc="Propensity batches", unit="batch")

        for start in batch_indices:
            end = min(start + batch_size, n_queries)
            batch_features = jnp.asarray(query_doc_features[start:end])
            preds = forward_step(self.model, batch_features)
            per_query_doc_pred.append(np.asarray(jax.device_get(preds)))

        per_query_doc_pred = np.concatenate(per_query_doc_pred, axis=0)
        return self._map_query_predictions_to_docs(data, per_query_doc_pred)

    def _predict_regression_per_doc(
        self,
        data,
        *,
        batch_size: int = 1024,
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        Inference path for models with a `predict_per_document` method.
        Runs on flat per-document features (`data.feature_matrix`) to avoid
        the rank-ordering mismatch that exists in `data.query_doc_features`.
        """
        @nnx.jit
        def forward_step(model, features):
            return model.predict_per_document(features)

        feature_matrix = np.asarray(data.feature_matrix)  # (N, F)
        n_docs = feature_matrix.shape[0]
        preds_list = []
        batch_indices = list(range(0, n_docs, batch_size))
        if show_progress:
            from tqdm import tqdm
            batch_indices = tqdm(batch_indices, desc="Propensity batches", unit="batch")

        for start in batch_indices:
            end = min(start + batch_size, n_docs)
            batch_features = jnp.asarray(feature_matrix[start:end])
            preds = forward_step(self.model, batch_features)
            preds_list.append(np.asarray(jax.device_get(preds)))

        return np.concatenate(preds_list, axis=0)  # (N,) — already per-doc



    def _predict_grouped_per_doc(
        self,
        data,
        *,
        debug_groups: bool = False,
        debug_group_limit: int = 5,
        show_progress: bool = True,
    ) -> np.ndarray:
        query_doc_features = np.asarray(data.query_doc_features)
        query_mask = np.asarray(data.mask)
        doc_id_map = np.asarray(data.doc_id_map)
        displays = np.asarray(data.displays)
        query_freq = np.asarray(data.query_freq)
        sampled_rows = np.asarray(getattr(data, "sessions", []), dtype=np.int64).reshape(-1)
        n_rows = query_doc_features.shape[0]

        if self.model.uses_global_groups or sampled_rows.size == 0:
            assignment_rows = np.arange(n_rows, dtype=np.int64)
        else:
            assignment_rows = np.unique(sampled_rows.astype(np.int64, copy=False))
        row_to_assignment_idx = np.full((n_rows,), -1, dtype=np.int64)
        row_to_assignment_idx[assignment_rows] = np.arange(assignment_rows.shape[0], dtype=np.int64)

        batch = {
            "query_doc_features": jnp.asarray(query_doc_features[assignment_rows]),
            "mask": jnp.asarray(query_mask[assignment_rows]),
            "doc_ids": jnp.asarray(doc_id_map[assignment_rows]),
        }
        if show_progress:
            method_name = getattr(self.model, "method", "grouped")
            print(
                f"[{method_name}] assigning groups for {assignment_rows.shape[0]} queries "
                f"and {query_doc_features.shape[1]} positions"
            )
            start = time.time()
        group_ids_assigned = np.asarray(self.model.compute_output(batch))
        if show_progress:
            elapsed = time.time() - start
            print(f"[{method_name}] group assignment finished in {elapsed:.2f}s")

        cutoff = int(displays.shape[1])
        alpha = np.asarray(self.alpha)[:cutoff]
        pred_per_doc = np.zeros(data.num_docs(), dtype=np.float64)

        if self.model.uses_global_groups:
            if show_progress:
                agg_start = time.time()

            valid = doc_id_map >= 0
            group_id_per_doc = np.full(data.num_docs(), -1, dtype=np.int32)
            group_ids_global = np.full(doc_id_map.shape, -1, dtype=np.int32)
            group_ids_global[assignment_rows] = group_ids_assigned
            group_id_per_doc[doc_id_map[valid]] = group_ids_global[valid]

            valid_docs = group_id_per_doc >= 0
            if not np.any(valid_docs):
                return pred_per_doc

            gids = group_id_per_doc[valid_docs]
            n_groups = int(gids.max()) + 1
            query_index_per_doc = np.asarray(data.query_index_per_document())
            query_idx_valid = query_index_per_doc[valid_docs]

            group_query_freq = np.bincount(
                gids,
                weights=query_freq[query_idx_valid].astype(np.float64, copy=False),
                minlength=n_groups,
            ).astype(np.float64, copy=False)

            group_displays = np.zeros((n_groups, cutoff), dtype=np.float64)
            displays_valid = displays[valid_docs].astype(np.float64, copy=False)
            for r in range(cutoff):
                group_displays[:, r] = np.bincount(
                    gids,
                    weights=displays_valid[:, r],
                    minlength=n_groups,
                )

            display_prob = group_displays / np.maximum(group_query_freq[:, None], 1.0)
            group_prop = display_prob @ alpha
            pred_per_doc[valid_docs] = group_prop[gids]

            if show_progress:
                elapsed = time.time() - agg_start
                print(
                    f"[{getattr(self.model, 'method', 'grouped')}] group aggregation finished in {elapsed:.2f}s"
                )

            if debug_groups:
                uniq, counts = np.unique(gids, return_counts=True)
                total_clusters = int(uniq.size)
                avg_docs = float(counts.mean()) if total_clusters > 0 else 0.0
                singleton_clusters = int(np.sum(counts == 1))
                print(
                    f"[{getattr(self.model, 'method', 'grouped')}] clusters used: total={total_clusters}, "
                    f"avg_docs_per_cluster={avg_docs:.2f}, "
                    f"singletons={singleton_clusters} ({singleton_clusters / max(total_clusters, 1):.1%})"
                )

            return pred_per_doc

        debug_printed = 0
        if debug_groups:
            valid_mask_assigned = query_mask[assignment_rows].astype(bool)
            valid_groups = group_ids_assigned[valid_mask_assigned]
            valid_groups = valid_groups[valid_groups >= 0]
            if valid_groups.size == 0:
                print(
                    f"[{getattr(self.model, 'method', 'grouped')}] clusters used: "
                    "total=0, avg_docs_per_cluster=0.00, singletons=0 (0.0%)"
                )
            else:
                uniq, counts = np.unique(valid_groups, return_counts=True)
                total_clusters = int(uniq.size)
                avg_docs = float(counts.mean())
                singleton_clusters = int(np.sum(counts == 1))
                print(
                    f"[{getattr(self.model, 'method', 'grouped')}] clusters used: total={total_clusters}, "
                    f"avg_docs_per_cluster={avg_docs:.2f}, "
                    f"singletons={singleton_clusters} ({singleton_clusters / total_clusters:.1%})"
                )

        # In shared-doc aggregation mode, query_freq is indexed by query-group id
        # (len == num unique queries), while q here indexes original rows/sessions.
        row_to_query_group = None
        if query_freq.shape[0] != query_doc_features.shape[0]:
            query_index_per_doc = np.asarray(data.query_index_per_document(), dtype=np.int64)
            row_to_query_group = np.full((doc_id_map.shape[0],), -1, dtype=np.int64)
            row_has_doc = np.any(doc_id_map >= 0, axis=1)
            if np.any(row_has_doc):
                first_valid_col = np.argmax(doc_id_map >= 0, axis=1)
                first_doc = doc_id_map[np.arange(doc_id_map.shape[0]), first_valid_col]
                valid_first = row_has_doc & (first_doc >= 0)
                row_to_query_group[valid_first] = query_index_per_doc[first_doc[valid_first]]

        query_iter = sampled_rows if sampled_rows.size > 0 else range(query_doc_features.shape[0])
        if show_progress:
            from tqdm import tqdm
            query_iter = tqdm(query_iter, desc="Aggregating groups", unit="query")

        for q in query_iter:
            q_int = int(q)
            if q_int < 0 or q_int >= query_doc_features.shape[0]:
                continue
            q_local = int(row_to_assignment_idx[q_int])
            if q_local < 0:
                continue
            mask = query_mask[q_int].astype(bool)
            if not np.any(mask):
                continue

            group_ids = group_ids_assigned[q_local]
            valid_positions = np.where(mask)[0]
            if valid_positions.size == 0:
                continue

            global_docs = doc_id_map[q_int, valid_positions]
            valid_doc_mask = global_docs >= 0
            if not np.any(valid_doc_mask):
                continue

            valid_positions = valid_positions[valid_doc_mask]
            global_docs = global_docs[valid_doc_mask]
            group_ids_valid = group_ids[valid_positions]

            if debug_groups and debug_printed < max(debug_group_limit, 0):
                n_valid = int(valid_positions.size)
                n_groups = int(np.unique(group_ids_valid[group_ids_valid >= 0]).size)
                print(
                    f"[{getattr(self.model, 'method', 'grouped')}] q={q} docs={n_valid} groups={n_groups}"
                )
                debug_printed += 1

            for gid in np.unique(group_ids_valid):
                if gid < 0:
                    continue
                doc_idx = global_docs[group_ids_valid == gid]
                if doc_idx.size == 0:
                    continue
                group_displays = displays[doc_idx].sum(axis=0)
                group_size = int(doc_idx.size)
                if row_to_query_group is None:
                    q_freq_idx = q_int
                else:
                    q_freq_idx = int(row_to_query_group[q_int])
                    if q_freq_idx < 0:
                        continue
                if q_freq_idx >= query_freq.shape[0]:
                    continue
                group_query_freq = max(query_freq[q_freq_idx], 1.0) * max(group_size, 1)
                display_prob = group_displays / group_query_freq
                group_prop = float(np.sum(display_prob * alpha))
                pred_per_doc[doc_idx] = group_prop

        return pred_per_doc

    def predict_per_doc(
        self,
        data,
        *,
        batch_size: int = 512,
        debug_groups: bool = False,
        debug_group_limit: int = 5,
        show_progress: bool = True,
    ) -> np.ndarray:
        if self.model.requires_alpha:
            if self.alpha is None:
                raise ValueError("alpha is required for this propensity model")
            return self._predict_grouped_per_doc(
                data,
                debug_groups=debug_groups,
                debug_group_limit=debug_group_limit,
                show_progress=show_progress,
            )
        if hasattr(self.model, "predict_per_document"):
            return self._predict_regression_per_doc(
                data,
                batch_size=batch_size,
                show_progress=show_progress,
            )
        return self._predict_classifier_per_doc(
            data,
            batch_size=batch_size,
            show_progress=show_progress,
        )


def train_propensity_classifier(
    *,
    propensity_model: ClassifierPropensityMLP,
    train_click_dataset,
    val_click_dataset,
    epochs: int = 150,
    patience: int = 3,
    batch_size: int = 512,
    learning_rate: float = 5e-3,
    min_delta: float = 1e-5,
    ckpt_dir: str | None = None,
    dataloader_num_workers: int = 0,
    dataloader_prefetch_factor: int | None = None,
    dataloader_persistent_workers: bool = False,
    dataloader_pin_memory: bool = False,
) -> tuple[ClassifierPropensityMLP, float]:
    def resolve_collate_fn(*datasets):
        for dataset in datasets:
            current = dataset
            visited = set()
            while current is not None and id(current) not in visited:
                visited.add(id(current))
                collate_fn = getattr(current, "collate_fn", None)
                if collate_fn is not None:
                    return collate_fn
                current = getattr(current, "dataset", None)
        raise AttributeError("Could not resolve collate_fn from the provided dataset wrappers")

    num_workers = max(int(dataloader_num_workers), 0)
    prefetch_factor = dataloader_prefetch_factor
    if num_workers > 0:
        prefetch_factor = int(prefetch_factor) if prefetch_factor is not None else 2
    else:
        prefetch_factor = None

    collate_fn = resolve_collate_fn(train_click_dataset, val_click_dataset)
    loader_kwargs = {
        "batch_size": batch_size,
        "collate_fn": collate_fn,
        "num_workers": num_workers,
    }
    if num_workers > 0:
        loader_kwargs.update(
            prefetch_factor=prefetch_factor,
            persistent_workers=bool(dataloader_persistent_workers),
            pin_memory=bool(dataloader_pin_memory),
        )

    train_click_loader = DataLoader(
        train_click_dataset,
        shuffle=True,
        **loader_kwargs,
    )
    val_click_loader = DataLoader(
        val_click_dataset,
        shuffle=False,
        **loader_kwargs,
    )

    if ckpt_dir and os.path.exists(ckpt_dir):
        print(f"Loading propensity model from {ckpt_dir}")
        model = load_model_params(propensity_model, ckpt_dir=ckpt_dir)
        return model, float("nan")

    print(
        "[PropensityEstimator] Starting MLP training "
        f"(train_batches={len(train_click_loader)}, val_batches={len(val_click_loader)}, "
        f"batch_size={batch_size}, lr={learning_rate})",
        flush=True,
    )

    trainer = PropensityTrainer(
        epochs=epochs,
        patience=patience,
        learning_rate=learning_rate,
        min_delta=min_delta,
    )
    propensity_model, best_val_loss = trainer.train(
        propensity_model,
        train_click_loader,
        val_click_loader,
    )
    print("[PropensityEstimator] Finished MLP training", flush=True)
    if ckpt_dir:
        print(f"[PropensityEstimator] Saving checkpoint to {ckpt_dir}", flush=True)
        trainer.save_model_params(propensity_model, ckpt_dir=ckpt_dir)
    return propensity_model, float(best_val_loss)


def prepare_propensity_model(
    *,
    spec: PropensityModelSpec,
    train_click_dataset,
    val_click_dataset,
    ckpt_dir: str | None = None,
    batch_size: int = 512,
    learning_rate: float = 5e-3,
    min_delta: float = 1e-5,
    dataloader_num_workers: int = 0,
    dataloader_prefetch_factor: int | None = None,
    dataloader_persistent_workers: bool = False,
    dataloader_pin_memory: bool = False,
) -> BasePropensityModel:
    if not spec.trainable:
        return spec.model
    model, best_val_loss = train_propensity_classifier(
        propensity_model=spec.model,
        train_click_dataset=train_click_dataset,
        val_click_dataset=val_click_dataset,
        ckpt_dir=ckpt_dir or spec.ckpt_dir,
        batch_size=batch_size,
        learning_rate=learning_rate,
        min_delta=min_delta,
        dataloader_num_workers=dataloader_num_workers,
        dataloader_prefetch_factor=dataloader_prefetch_factor,
        dataloader_persistent_workers=dataloader_persistent_workers,
        dataloader_pin_memory=dataloader_pin_memory,
    )
    if np.isfinite(best_val_loss):
        print(f"Best validation loss: {best_val_loss:.6f}")
    return model


class PropensityEstimator:
    def __init__(
        self,
        *,
        model_spec: PropensityModelSpec,
        alpha: np.ndarray,
        checkpoint_dir: str | None = None,
        apply_sigmoid: bool = True,
        batch_size: int = 512,
        learning_rate: float = 5e-3,
        min_delta: float = 1e-5,
        dataloader_num_workers: int = 0,
        dataloader_prefetch_factor: int | None = None,
        dataloader_persistent_workers: bool = False,
        dataloader_pin_memory: bool = False,
    ):
        self.alpha = np.asarray(alpha)
        self.model_spec = model_spec
        self.checkpoint_dir = checkpoint_dir
        self.apply_sigmoid = apply_sigmoid
        self.batch_size = int(batch_size)
        self.learning_rate = float(learning_rate)
        self.min_delta = float(min_delta)
        self.dataloader_num_workers = int(dataloader_num_workers)
        self.dataloader_prefetch_factor = dataloader_prefetch_factor
        self.dataloader_persistent_workers = bool(dataloader_persistent_workers)
        self.dataloader_pin_memory = bool(dataloader_pin_memory)

    @property
    def model(self) -> BasePropensityModel:
        return self.model_spec.model

    def fit(self, train_click_dataset, val_click_dataset) -> None:
        if not self.model_spec.trainable:
            print(
                f"[PropensityEstimator] No fit required for '{self.model_spec.name}'",
                flush=True,
            )
            return
        print(
            f"[PropensityEstimator] Fitting '{self.model_spec.name}'",
            flush=True,
        )
        model = prepare_propensity_model(
            spec=self.model_spec,
            train_click_dataset=train_click_dataset,
            val_click_dataset=val_click_dataset,
            ckpt_dir=self.checkpoint_dir,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            min_delta=self.min_delta,
            dataloader_num_workers=self.dataloader_num_workers,
            dataloader_prefetch_factor=self.dataloader_prefetch_factor,
            dataloader_persistent_workers=self.dataloader_persistent_workers,
            dataloader_pin_memory=self.dataloader_pin_memory,
        )
        self.model_spec.model = model
        print(
            f"[PropensityEstimator] Fit completed for '{self.model_spec.name}'",
            flush=True,
        )

    def predict_per_doc(
        self,
        data,
        *,
        batch_size: int | None = None,
        debug_groups: bool = False,
        debug_group_limit: int = 5,
        show_progress: bool = True,
    ) -> np.ndarray:
        print(
            f"[PropensityEstimator] Predicting per-doc outputs with '{self.model_spec.name}'",
            flush=True,
        )
        predictor = PropensityPredictor(
            model=self.model_spec.model,
            alpha=self.alpha,
            apply_sigmoid=self.apply_sigmoid,
        )
        predictions = predictor.predict_per_doc(
            data,
            batch_size=batch_size if batch_size is not None else self.batch_size,
            debug_groups=debug_groups,
            debug_group_limit=debug_group_limit,
            show_progress=show_progress,
        )
        print(
            f"[PropensityEstimator] Prediction completed for '{self.model_spec.name}'",
            flush=True,
        )
        return predictions
