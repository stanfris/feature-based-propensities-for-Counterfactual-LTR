from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from flax import nnx
from omegaconf import DictConfig

from .base import PropensityModelSpec
from .frequency import FrequencyPropensityModel
from .mlp import ClassifierPropensityMLP, RegressionPropensityMLP
from .unsupervised import UnsupervisedGroupingPropensity


def _infer_dataset_dims(dataset) -> tuple[int, int]:
    if hasattr(dataset, "n_features") and hasattr(dataset, "n_positions"):
        return int(dataset.n_features), int(dataset.n_positions)
    features = np.asarray(dataset.query_doc_features)
    return int(features.shape[2]), int(features.shape[1])


def build_propensity_model_spec(
    *,
    config: DictConfig,
    train_dataset,
    bias_value_array: jnp.ndarray,
    seed: int,
    method: str | None = None,
) -> PropensityModelSpec:
    method_norm = str(method or config.propensity_model.type).lower().replace("-", "_")
    feature_dim, positions = _infer_dataset_dims(train_dataset)
    deterministic_groups = bool(config.deterministic_groups)
    bias_width = int(np.asarray(bias_value_array).shape[0])

    if bias_width != positions:
        raise ValueError(
            "Propensity model width mismatch: "
            f"dataset positions={positions}, bias vector length={bias_width}. "
            "Align the click dataset width with the effective aggregation cutoff before "
            "building trainable propensity models."
        )

    if method_norm == "frequency_based":
        return PropensityModelSpec(
            name="frequency_based",
            model=FrequencyPropensityModel(),
            trainable=False,
        )

    if method_norm == "propensity_mlp_classifier":
        classifier_cfg = config.propensity_model.classifier
        layers = int(classifier_cfg.layers)
        hidden_units = int(classifier_cfg.hidden_units)
        dropout = float(classifier_cfg.dropout)
        model = ClassifierPropensityMLP(
            query_doc_features=feature_dim,
            layers=layers,
            hidden_units=hidden_units,
            dropout=dropout,
            rngs=nnx.Rngs(seed),
            bias_value_array=bias_value_array,
            positions=positions,
        )
        return PropensityModelSpec(
            name="propensity_mlp_classifier",
            model=model,
            trainable=True,
        )

    if method_norm == "propensity_mlp_regression":
        regression_cfg = config.propensity_model.regression
        layers = int(regression_cfg.layers)
        hidden_units = int(regression_cfg.hidden_units)
        dropout = float(regression_cfg.dropout)
        model = RegressionPropensityMLP(
            query_doc_features=feature_dim,
            layers=layers,
            hidden_units=hidden_units,
            dropout=dropout,
            alpha=bias_value_array,
            rngs=nnx.Rngs(seed),
        )
        return PropensityModelSpec(
            name="propensity_mlp_regression",
            model=model,
            trainable=True,
        )

    if method_norm == "kmeans":
        kmeans_cfg = config.propensity_model.kmeans
        model = UnsupervisedGroupingPropensity(
            query_doc_features=feature_dim,
            positions=positions,
            rngs=nnx.Rngs(seed),
            bias_value_array=bias_value_array,
            method="kmeans",
            num_groups=int(kmeans_cfg.num_groups),
            min_cluster_size=int(kmeans_cfg.min_cluster_size),
            kmeans_iters=int(kmeans_cfg.kmeans_iters),
            deterministic_groups=deterministic_groups,
        )
        return PropensityModelSpec(
            name="unsupervised_kmeans",
            model=model,
            trainable=False,
        )

    if method_norm == "cosine":
        cosine_cfg = config.propensity_model.cosine
        model = UnsupervisedGroupingPropensity(
            query_doc_features=feature_dim,
            positions=positions,
            rngs=nnx.Rngs(seed),
            bias_value_array=bias_value_array,
            method="cosine",
            knn_k=int(cosine_cfg.knn_k),
            cosine_threshold=float(cosine_cfg.cosine_threshold),
            deterministic_groups=deterministic_groups,
        )
        return PropensityModelSpec(
            name="unsupervised_cosine",
            model=model,
            trainable=False,
        )

    if method_norm == "euclidean":
        euclidean_cfg = config.propensity_model.euclidean
        model = UnsupervisedGroupingPropensity(
            query_doc_features=feature_dim,
            positions=positions,
            rngs=nnx.Rngs(seed),
            bias_value_array=bias_value_array,
            method="euclidean",
            knn_k=int(euclidean_cfg.knn_k),
            euclidean_threshold=float(euclidean_cfg.euclidean_threshold),
            deterministic_groups=deterministic_groups,
        )
        return PropensityModelSpec(
            name="unsupervised_euclidean",
            model=model,
            trainable=False,
        )

    if method_norm == "knn":
        knn_cfg = config.propensity_model.knn
        model = UnsupervisedGroupingPropensity(
            query_doc_features=feature_dim,
            positions=positions,
            rngs=nnx.Rngs(seed),
            bias_value_array=bias_value_array,
            method="knn",
            knn_k=int(knn_cfg.knn_k),
            deterministic_groups=deterministic_groups,
        )
        return PropensityModelSpec(
            name="unsupervised_knn",
            model=model,
            trainable=False,
        )

    raise ValueError(f"Unknown propensity model type: {method_norm}")
