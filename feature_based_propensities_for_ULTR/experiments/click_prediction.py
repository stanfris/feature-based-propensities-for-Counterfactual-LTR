from __future__ import annotations

from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from feature_based_propensities_for_ULTR.logging_policy.samplers import plackett_luce_temperature_scale


def _sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-values))


def _softmax_with_mask(logits: np.ndarray, mask: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    safe_logits = np.where(mask, logits, -np.inf)
    row_max = np.max(np.where(mask, safe_logits, -np.inf), axis=1, keepdims=True)
    row_max = np.where(np.isfinite(row_max), row_max, 0.0)
    exp_logits = np.where(mask, np.exp(safe_logits - row_max), 0.0)
    denom = np.sum(exp_logits, axis=1, keepdims=True)
    return np.where(mask, exp_logits / np.maximum(denom, 1e-12), 0.0)


def _prepare_click_batch(click_dataset) -> dict[str, np.ndarray]:
    return {
        "query": np.asarray(click_dataset.query),
        "query_doc_features": np.asarray(click_dataset.query_doc_features),
        "lp_query_doc_features": np.asarray(click_dataset.lp_query_doc_features),
        "query_doc_ids": np.asarray(click_dataset.query_doc_ids),
        "labels": np.asarray(click_dataset.labels),
        "mask": np.asarray(click_dataset.mask, dtype=bool),
        "positions": np.broadcast_to(
            np.arange(np.asarray(click_dataset.positions).shape[1], dtype=np.int64)[None, :],
            np.asarray(click_dataset.positions).shape,
        ).copy(),
        "n": np.asarray(click_dataset.n),
    }


def _predict_policy_click_probabilities(
    model,
    click_dataset,
    *,
    alpha: np.ndarray,
    model_output_multiplier: float,
) -> np.ndarray:
    batch = _prepare_click_batch(click_dataset)
    feature_tensor = np.asarray(batch["query_doc_features"], dtype=np.float64)
    mask = np.asarray(batch["mask"], dtype=bool)
    n_sessions, max_len, n_features = feature_tensor.shape
    flat_features = feature_tensor.reshape(n_sessions * max_len, n_features)

    if hasattr(model, "policy_scores_for_split") and hasattr(click_dataset, "doc_id_map"):
        flat_scores = np.asarray(model.policy_scores_for_split(click_dataset), dtype=np.float64)
        doc_id_map = np.asarray(click_dataset.doc_id_map)
        query_scores = np.zeros((n_sessions, max_len), dtype=np.float64)
        valid = doc_id_map >= 0
        query_scores[valid] = flat_scores[doc_id_map[valid]]
    else:
        flat_scores = np.asarray(model.score_feature_matrix(flat_features), dtype=np.float64).reshape(-1)
        query_scores = flat_scores.reshape(n_sessions, max_len) * float(model_output_multiplier)

    if hasattr(model, "policy_family") and getattr(model, "policy_family", "") == "plackett_luce":
        query_scores = np.asarray(
            plackett_luce_temperature_scale(
                query_scores,
                policy_temperature=float(getattr(model, "temperature", 0.0)),
            ),
            dtype=np.float64,
        )

    policy_prob = _softmax_with_mask(query_scores, mask)
    alpha = np.asarray(alpha, dtype=np.float64).reshape(-1)
    alpha_by_rank = np.broadcast_to(alpha[:max_len][None, :], (n_sessions, max_len))
    return np.where(mask, policy_prob * alpha_by_rank, 0.0)


def predict_click_probabilities(
    *,
    model,
    click_dataset,
    alpha: np.ndarray,
    model_output_multiplier: float = 1.0,
) -> np.ndarray:
    return _predict_policy_click_probabilities(
        model,
        click_dataset,
        alpha=alpha,
        model_output_multiplier=model_output_multiplier,
    )


def _binary_log_loss(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    y_prob = np.clip(np.asarray(y_prob, dtype=np.float64), 1e-12, 1.0 - 1e-12)
    y_true = np.asarray(y_true, dtype=np.float64)
    return float(-np.mean(y_true * np.log(y_prob) + (1.0 - y_true) * np.log(1.0 - y_prob)))


def _brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    return float(np.mean(np.square(y_prob - y_true)))


def _roc_auc_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.int32)
    y_score = np.asarray(y_score, dtype=np.float64)
    pos = int(np.sum(y_true == 1))
    neg = int(np.sum(y_true == 0))
    if pos == 0 or neg == 0:
        return float("nan")
    order = np.argsort(y_score, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(y_score) + 1, dtype=np.float64)
    _, inverse, counts = np.unique(y_score, return_inverse=True, return_counts=True)
    if np.any(counts > 1):
        rank_sums = np.bincount(inverse, weights=ranks)
        avg_ranks = rank_sums / counts
        ranks = avg_ranks[inverse]
    pos_rank_sum = float(np.sum(ranks[y_true == 1]))
    return float((pos_rank_sum - pos * (pos + 1) / 2.0) / (pos * neg))


def evaluate_click_prediction(
    *,
    model,
    click_dataset,
    alpha: np.ndarray,
    model_output_multiplier: float = 1.0,
    prediction_output_path: str | Path | None = None,
) -> dict[str, Any]:
    probabilities = predict_click_probabilities(
        model=model,
        click_dataset=click_dataset,
        alpha=alpha,
        model_output_multiplier=model_output_multiplier,
    )
    mask = np.asarray(click_dataset.mask, dtype=bool)
    clicks = np.asarray(click_dataset.clicks, dtype=np.int8)

    y_true = clicks[mask].astype(np.float64)
    y_prob = probabilities[mask].astype(np.float64)
    metrics = {
        "n_observations": int(y_true.shape[0]),
        "click_rate": float(np.mean(y_true)) if y_true.size > 0 else float("nan"),
        "avg_predicted_click_probability": float(np.mean(y_prob)) if y_prob.size > 0 else float("nan"),
        "log_loss": _binary_log_loss(y_true, y_prob) if y_true.size > 0 else float("nan"),
        "brier_score": _brier_score(y_true, y_prob) if y_true.size > 0 else float("nan"),
        "roc_auc": _roc_auc_score(y_true, y_prob) if y_true.size > 0 else float("nan"),
    }

    output_path = None
    if prediction_output_path is not None:
        output_path = Path(prediction_output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output_path,
            probabilities=probabilities,
            clicks=clicks,
            mask=mask,
            query=np.asarray(click_dataset.query),
            query_doc_ids=np.asarray(click_dataset.query_doc_ids),
        )

    return {
        "metrics": metrics,
        "prediction_output_path": str(output_path.resolve()) if output_path is not None else None,
    }


def flatten_click_prediction_metrics(result: dict[str, Any]) -> dict[str, float]:
    metrics = dict(result.get("metrics", {}))
    return {
        f"click_prediction/{key}": float(value)
        for key, value in metrics.items()
        if isinstance(value, (int, float, np.floating)) and np.isfinite(value)
    }


__all__ = [
    "evaluate_click_prediction",
    "flatten_click_prediction_metrics",
    "predict_click_probabilities",
]
