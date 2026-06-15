from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax import Array

from feature_based_propensities_for_ULTR.ranking import plackettluce as pl


class PolicyModelBase(nnx.Module):
    name: str = "base"

    def __init__(
        self,
        *,
        scorer: nnx.Module | None = None,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.scorer = scorer
        self.eps = eps

    def _require_scorer(self) -> nnx.Module:
        if self.scorer is None:
            raise ValueError(f"{self.__class__.__name__} requires a scorer for ranking.")
        return self.scorer

    def compute_loss(
        self,
        scores: Array,
        doc_weights: Array,
        rank_weights: Array,
        n_samples: int,
    ) -> Array:
        grad = pl.gradient_based_on_samples(
            rank_weights,
            doc_weights,
            scores,
            n_samples=n_samples,
        )
        grad = jax.lax.stop_gradient(grad)
        return -jnp.sum(scores * grad)

    def compute_weights(
        self,
        data,
        *,
        propensity: np.ndarray | None = None,
        propensity_clipping: float | None = None,
        regression_pred: np.ndarray | None = None,
    ) -> np.ndarray:
        raise NotImplementedError

    def score_query(self, query_doc_features: Array) -> Array:
        scorer = self._require_scorer()
        return scorer({"query_doc_features": query_doc_features})

    def score_feature_matrix(self, feature_matrix: Array) -> Array:
        scorer = self._require_scorer()
        return scorer({"query_doc_features": feature_matrix})


def _to_float64(values) -> np.ndarray:
    return np.asarray(values, dtype=np.float64)


def ctr_per_doc(data) -> np.ndarray:
    clicks = getattr(data, "clicks_per_doc", None)
    if clicks is None:
        clicks = np.sum(np.asarray(data.clicks), axis=1)
    else:
        clicks = np.asarray(clicks)
    q_idx = np.asarray(data.query_index_per_document())
    query_freq = np.asarray(data.query_freq)
    safe_query_freq = np.maximum(query_freq, 1)
    return _to_float64(clicks) / safe_query_freq[q_idx]


def naive_doc_weights(data) -> np.ndarray:
    clicks = getattr(data, "clicks_per_doc", None)
    displays = getattr(data, "displays_per_doc", None)
    if clicks is None:
        clicks = np.sum(np.asarray(data.clicks), axis=1)
    else:
        clicks = np.asarray(clicks)
    if displays is None:
        displays = np.sum(np.asarray(data.displays), axis=1)
    else:
        displays = np.asarray(displays)
    safe_displays = np.maximum(displays, 1)
    return _to_float64(clicks) / safe_displays


def clip_propensity(propensity: np.ndarray, propensity_clipping: float | None) -> np.ndarray:
    propensity = _to_float64(propensity)
    if propensity_clipping is None:
        return propensity
    return np.maximum(propensity, propensity_clipping)


def _safe_denom(values: np.ndarray) -> np.ndarray:
    values = _to_float64(values)
    return values + (values == 0).astype(values.dtype)


def compute_dr_weights(
    ctr: np.ndarray,
    propensity: np.ndarray,
    regression_pred: np.ndarray,
    propensity_clipping: float | None,
    propensity_for_denom: np.ndarray | None = None,
) -> np.ndarray:
    propensity = _to_float64(propensity)
    regression_pred = _to_float64(regression_pred)
    if propensity_for_denom is None:
        propensity_for_denom = propensity
    propensity_for_denom = _to_float64(propensity_for_denom)
    denom = clip_propensity(propensity_for_denom, propensity_clipping)
    safe_denom = _safe_denom(denom)
    return regression_pred + (ctr - propensity * regression_pred) / safe_denom


class NaivePolicyModel(PolicyModelBase):
    name = "naive"

    def compute_weights(self, data, **_) -> np.ndarray:
        return naive_doc_weights(data)


class NaiveHOModel(PolicyModelBase):
    name = "naive-ho"

    def compute_weights(self, data, **_) -> np.ndarray:
        # 2022 baseline: clicks per doc divided by sampled query frequency.
        return ctr_per_doc(data)


class MaxScorePolicyModel(PolicyModelBase):
    name = "max-score"

    def compute_weights(self, data, **_) -> np.ndarray:
        labels = getattr(data, "label_vector", None)
        if labels is None:
            raise ValueError("MaxScorePolicyModel requires data.label_vector for full-information training.")
        return _to_float64(labels)


class DMPolicyModel(PolicyModelBase):
    name = "dm"

    def compute_weights(
        self,
        data,
        *,
        regression_pred: np.ndarray | None = None,
        **_,
    ) -> np.ndarray:
        if regression_pred is None:
            raise ValueError("DMPolicyModel requires regression_pred to compute weights.")
        return regression_pred


class DRPolicyModel(PolicyModelBase):
    name = "dr"

    def compute_weights(
        self,
        data,
        *,
        propensity: np.ndarray | None = None,
        regression_pred: np.ndarray | None = None,
        propensity_clipping: float | None = None,
        propensity_for_denom: np.ndarray | None = None,
        **_,
    ) -> np.ndarray:
        if propensity is None or regression_pred is None:
            raise ValueError("DRPolicyModel requires propensity and regression_pred to compute weights.")
        ctr = ctr_per_doc(data)
        weights = compute_dr_weights(
            ctr,
            propensity,
            regression_pred,
            propensity_clipping,
            propensity_for_denom=propensity_for_denom,
        )
        return weights


class IPSModel(PolicyModelBase):
    name = "ips"

    def compute_weights(
        self,
        data,
        *,
        propensity: np.ndarray | None = None,
        propensity_clipping: float | None = None,
        **_,
    ) -> np.ndarray:
        if propensity is None:
            raise ValueError("IPSModel requires propensity to compute weights.")
        ctr = ctr_per_doc(data)
        denom = clip_propensity(propensity, propensity_clipping)
        safe_denom = _safe_denom(denom)
        return ctr / safe_denom


def build_policy_model(model_choice: str, **kwargs) -> PolicyModelBase:
    if model_choice == "naive":
        return NaivePolicyModel(**kwargs)
    if model_choice == "naive-ho":
        return NaiveHOModel(**kwargs)
    if model_choice == "dm":
        return DMPolicyModel(**kwargs)
    if model_choice == "dr":
        return DRPolicyModel(**kwargs)
    if model_choice == "ips":
        return IPSModel(**kwargs)
    if model_choice == "max-score":
        return MaxScorePolicyModel(**kwargs)
    raise ValueError(f"Unsupported policy model choice: {model_choice}")


__all__ = [
    "PolicyModelBase",
    "NaivePolicyModel",
    "NaiveHOModel",
    "MaxScorePolicyModel",
    "DMPolicyModel",
    "DRPolicyModel",
    "IPSModel",
    "build_policy_model",
    "clip_propensity",
    "compute_dr_weights",
    "ctr_per_doc",
    "naive_doc_weights",
]
