from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import logging

import jax

from feature_based_propensities_for_ULTR.experiments.tracking import Tracker

logger = logging.getLogger(__name__)
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx
from hydra.utils import instantiate
from omegaconf import DictConfig

from feature_based_propensities_for_ULTR.models.policy_models import clip_propensity, ctr_per_doc
from feature_based_propensities_for_ULTR.models.regression_model import RegressionModel


@dataclass(frozen=True)
class RegressionBundle:
    model: nnx.Module
    train_pred: np.ndarray
    val_pred: np.ndarray


def _safe_corrcoef(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return float("nan")
    if np.std(a) == 0.0 or np.std(b) == 0.0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _log_regression_diagnostics(
    *,
    method: str,
    split: str,
    data,
    propensity: np.ndarray,
    propensity_clipping: float | None,
    pred: np.ndarray,
    tracker: Optional[Tracker] = None,
) -> None:
    ctr = np.asarray(ctr_per_doc(data), dtype=np.float64)
    alpha = np.asarray(propensity, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64).reshape(-1)

    n = min(ctr.shape[0], alpha.shape[0], pred.shape[0])
    if n == 0:
        print(f"[Regression diagnostics][{method}][{split}] empty arrays, skipping.")
        return

    ctr = ctr[:n]
    alpha = alpha[:n]
    pred = pred[:n]

    denom = clip_propensity(alpha, propensity_clipping)
    safe_denom = denom + (denom == 0).astype(denom.dtype)
    correction = (ctr - alpha * pred) / safe_denom
    dr_proxy = pred + correction

    print(
        f"[Regression diagnostics][{method}][{split}] "
        f"dr_proxy(mean={np.mean(dr_proxy):.6f}, std={np.std(dr_proxy):.6f})"
    )

    if tracker is not None:
        key_prefix = f"regression_diagnostics/{method}/{split}"
        tracker.log_metrics(
            {
                f"{key_prefix}/pred_mean": float(np.mean(pred)),
                f"{key_prefix}/pred_std": float(np.std(pred)),
                f"{key_prefix}/pred_min": float(np.min(pred)),
                f"{key_prefix}/pred_max": float(np.max(pred)),
                f"{key_prefix}/ctr_mean": float(np.mean(ctr)),
                f"{key_prefix}/corr_pred_ctr": _safe_corrcoef(pred, ctr),
                f"{key_prefix}/corr_pred_alpha": _safe_corrcoef(pred, alpha),
                f"{key_prefix}/residual_mean": float(np.mean(ctr - alpha * pred)),
                f"{key_prefix}/residual_std": float(np.std(ctr - alpha * pred)),
                f"{key_prefix}/dr_proxy_mean": float(np.mean(dr_proxy)),
                f"{key_prefix}/dr_proxy_std": float(np.std(dr_proxy)),
            },
            commit=False,
        )
        tracker.log_histogram(f"{key_prefix}/pred_dist", pred, commit=False)
        tracker.log_histogram(f"{key_prefix}/residual_dist", ctr - alpha * pred, commit=False)
        tracker.log_histogram(f"{key_prefix}/dr_proxy_dist", dr_proxy, commit=False)


def _print_regression_output_set(
    *,
    method: str,
    split: str,
    values: np.ndarray,
    head: int = 10,
) -> None:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        print(f"[Regression outputs][{method}][{split}] empty values")
        return
    preview = np.array2string(arr[:head], precision=6, separator=", ")
    q = np.quantile(arr, [0.05, 0.5, 0.95])
    print(
        f"[Regression outputs][{method}][{split}] "
        f"n={arr.size}, mean={arr.mean():.6f}, std={arr.std():.6f}, "
        f"min={arr.min():.6f}, p05={q[0]:.6f}, p50={q[1]:.6f}, p95={q[2]:.6f}, max={arr.max():.6f}, "
        f"head{head}={preview}"
    )


def _print_regression_debug_samples(
    *,
    method: str,
    split: str,
    features: np.ndarray,
    pred: np.ndarray,
    max_samples: int = 25,
    feature_head: int = 5,
) -> None:
    feat = np.asarray(features, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64).reshape(-1)
    n = min(int(max_samples), feat.shape[0], pred.shape[0])
    if n <= 0:
        print(f"[Regression debug][{method}][{split}] no samples available")
        return

    print(
        f"[Regression debug][{method}][{split}] "
        f"showing {n} samples (feature_head={feature_head})"
    )
    for idx in range(n):
        feature_row = feat[idx].reshape(-1)
        head = np.array2string(
            feature_row[:feature_head],
            precision=4,
            separator=", ",
        )
        print(
            f"  idx={idx} "
            f"pred={pred[idx]:.6f} "
            f"feat_norm={np.linalg.norm(feature_row):.6f} "
            f"feat_mean={feature_row.mean():.6f} "
            f"feat_head={head}"
        )


def _prepare_regression_arrays(
    data,
    propensity: np.ndarray,
    propensity_clipping: float | None,
):
    ctr = ctr_per_doc(data)
    alpha = np.asarray(propensity, dtype=np.float64)
    denom = clip_propensity(alpha, propensity_clipping)
    safe_denom = denom + (denom == 0).astype(denom.dtype)
    pos_weights = ctr / safe_denom
    neg_weights = (alpha - ctr) / safe_denom
    q_idx = np.asarray(data.query_index_per_document())
    query_freq = np.asarray(data.query_freq)
    valid = (
        np.isfinite(pos_weights)
        & np.isfinite(neg_weights)
        & (alpha > 0)
        & (query_freq[q_idx] > 0)
    )
    features = np.asarray(data.feature_matrix)[valid]
    return features, pos_weights[valid], neg_weights[valid]


def _count_valid_regression_rows(
    data,
    propensity: np.ndarray,
) -> tuple[int, int]:
    ctr = ctr_per_doc(data)
    alpha = np.asarray(propensity, dtype=np.float64)
    q_idx = np.asarray(data.query_index_per_document())
    query_freq = np.asarray(data.query_freq)
    valid = np.isfinite(alpha) & (alpha > 0) & (query_freq[q_idx] > 0) & np.isfinite(ctr)
    return int(np.sum(valid)), int(valid.shape[0])


def _run_regression_eval(
    model: nnx.Module,
    val_feat: np.ndarray,
    val_pos_w: np.ndarray,
    val_neg_w: np.ndarray,
    batch_size: int,
    eval_step_fn,
) -> float:
    model.eval()
    total = 0.0
    count = 0
    n_val = int(val_feat.shape[0])
    for start in range(0, n_val, batch_size):
        end = min(start + batch_size, n_val)
        batch_feat = jnp.asarray(val_feat[start:end])
        batch_pos = jnp.asarray(val_pos_w[start:end])
        batch_neg = jnp.asarray(val_neg_w[start:end])
        loss = eval_step_fn(model, batch_feat, batch_pos, batch_neg)
        total += float(loss) * (end - start)
        count += end - start
    return float(total) / max(count, 1)


def _run_regression_train_epoch(
    model: nnx.Module,
    optimizer: nnx.Optimizer,
    train_feat: np.ndarray,
    train_pos_w: np.ndarray,
    train_neg_w: np.ndarray,
    batch_size: int,
    train_step_fn,
    rng_perm: np.random.Generator,
) -> float:
    model.train()
    n_train = int(train_feat.shape[0])
    perm = rng_perm.permutation(n_train)
    total_loss = 0.0
    count = 0
    for start in range(0, n_train, batch_size):
        end = min(start + batch_size, n_train)
        idx = perm[start:end]
        batch_feat = jnp.asarray(train_feat[idx])
        batch_pos = jnp.asarray(train_pos_w[idx])
        batch_neg = jnp.asarray(train_neg_w[idx])
        loss = train_step_fn(model, optimizer, batch_feat, batch_pos, batch_neg)
        total_loss += float(loss) * (end - start)
        count += end - start
    return float(total_loss) / max(count, 1)


def train_regression_model(
    *,
    config: DictConfig,
    train_data,
    val_data,
    train_propensity: np.ndarray,
    val_propensity: np.ndarray,
    propensity_clipping: float | None,
    seed: int,
    tracker: Optional[Tracker] = None,
):
    ips_cfg = config.ips
    reg_cfg = getattr(ips_cfg, "regression", None)
    if reg_cfg is None:
        raise RuntimeError("Missing ips.regression configuration for DM/DR estimators.")

    model_cfg = reg_cfg.model
    trainer_cfg = reg_cfg.trainer

    # Use the universal random_state from top-level config for all RNG streams
    global_seed = int(getattr(config, "random_state", 42))

    n_features = int(np.asarray(train_data.feature_matrix).shape[1])
    rngs = nnx.Rngs(global_seed)
    model = RegressionModel(
        query_doc_features=n_features,
        layers=int(model_cfg.layers),
        hidden_units=int(model_cfg.hidden_units),
        dropout=float(model_cfg.dropout),
        rngs=rngs,
        final_activation=bool(getattr(model_cfg, "final_activation", True)),
    )

    train_feat, train_pos_w, train_neg_w = _prepare_regression_arrays(
        train_data, train_propensity, propensity_clipping
    )
    val_feat, val_pos_w, val_neg_w = _prepare_regression_arrays(
        val_data, val_propensity, propensity_clipping
    )

    batch_size = int(getattr(trainer_cfg, "batch_size", 1024))
    max_epochs = int(getattr(trainer_cfg, "max_epochs", 200))
    early_stop_diff = float(getattr(trainer_cfg, "early_stop_diff", 0.0))
    early_stop_per_epochs = int(getattr(trainer_cfg, "early_stop_per_epochs", 3))
    learning_rate = float(getattr(trainer_cfg, "learning_rate", 1e-3))
    print_updates = bool(getattr(trainer_cfg, "print_updates", True))
    eps = float(getattr(trainer_cfg, "eps", 1e-6))

    optimizer = nnx.Optimizer(model, optax.adam(learning_rate))
    best_state = nnx.state(model)
    best_val_loss = float("inf")

    def _batch_loss(pred, pos_w, neg_w):
        pred = jnp.clip(pred, eps, 1.0 - eps)
        return -jnp.mean(pos_w * jnp.log(pred) + neg_w * jnp.log(1.0 - pred))

    @nnx.jit
    def train_step(model, optimizer, batch_feat, batch_pos_w, batch_neg_w):
        def loss_fn(model):
            pred = model({"query_doc_features": batch_feat})
            return _batch_loss(pred, batch_pos_w, batch_neg_w)

        loss, grads = nnx.value_and_grad(loss_fn)(model)
        optimizer.update(grads)
        return loss

    @nnx.jit
    def eval_step(model, batch_feat, batch_pos_w, batch_neg_w):
        pred = model({"query_doc_features": batch_feat})
        return _batch_loss(pred, batch_pos_w, batch_neg_w)

    rng = np.random.default_rng(global_seed + 1)
    n_train = int(train_feat.shape[0])
    n_val = int(val_feat.shape[0])
    if n_train == 0 or n_val == 0:
        train_valid, train_total = _count_valid_regression_rows(train_data, train_propensity)
        val_valid, val_total = _count_valid_regression_rows(val_data, val_propensity)
        raise RuntimeError(
            "Regression training requires non-empty train/val data. "
            f"Valid rows train={n_train}/{train_total} (base_valid={train_valid}), "
            f"val={n_val}/{val_total} (base_valid={val_valid}). "
            "Check propensity scale/range (expected probabilities > 0) and session filtering."
        )

    if print_updates:
        logger.info("Training regression model for DM/DR...")

    for epoch in range(max_epochs):
        train_loss = _run_regression_train_epoch(
            model, optimizer, train_feat, train_pos_w, train_neg_w, batch_size, train_step, rng
        )

        if (epoch + 1) % early_stop_per_epochs != 0:
            continue

        val_loss = _run_regression_eval(
            model, val_feat, val_pos_w, val_neg_w, batch_size, eval_step
        )
        if print_updates:
            logger.info(f"Regression epoch {epoch + 1}: train_loss={train_loss:.6f} val_loss={val_loss:.6f}")

        improvement = best_val_loss - val_loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = nnx.state(model)
        
        if tracker is not None:
            tracker.log_metrics({
                "regression_trainer/epoch": epoch,
                "regression_trainer/train_loss": train_loss,
                "regression_trainer/val_loss": val_loss,
            }, commit=True)

        if improvement < early_stop_diff:
            break

    nnx.update(model, best_state)
    model.eval()
    return model, best_val_loss


def predict_regression_model(model: nnx.Module, data, batch_size: int = 2048) -> np.ndarray:
    model.eval()
    features = np.asarray(data.feature_matrix)
    if features.shape[0] == 0:
        return np.zeros((0,), dtype=np.float64)
    outputs = []
    for start in range(0, features.shape[0], batch_size):
        end = min(start + batch_size, features.shape[0])
        batch_feat = jnp.asarray(features[start:end])
        pred = model({"query_doc_features": batch_feat})
        outputs.append(np.asarray(jax.device_get(pred)))
    if not outputs:
        return np.zeros((features.shape[0],), dtype=np.float64)
    return np.concatenate(outputs, axis=0).astype(np.float64)


def train_regression_bundle(
    *,
    config: DictConfig,
    train_data,
    val_data,
    train_propensity: np.ndarray,
    val_propensity: np.ndarray,
    propensity_clipping: float | None,
    seed: int,
    tracker: Optional[Tracker] = None,
) -> RegressionBundle:
    ips_cfg = config.ips
    reg_cfg = getattr(ips_cfg, "regression", None)
    trainer_cfg = getattr(reg_cfg, "trainer", None)
    train_batch_size = int(getattr(trainer_cfg, "batch_size", 2048)) if trainer_cfg else 2048
    debug = bool(getattr(ips_cfg, "debug", False))
    debug_max_samples = int(getattr(ips_cfg, "debug_max_samples", 25))
    debug_feature_head = int(getattr(ips_cfg, "debug_feature_head", 5))

    method = "regression"

    model, _ = train_regression_model(
        config=config,
        train_data=train_data,
        val_data=val_data,
        train_propensity=train_propensity,
        val_propensity=val_propensity,
        propensity_clipping=propensity_clipping,
        seed=seed,
        tracker=tracker,
    )
    train_pred = predict_regression_model(model, train_data, batch_size=train_batch_size)
    val_pred = predict_regression_model(model, val_data, batch_size=train_batch_size)
    _print_regression_output_set(method=method, split="train", values=train_pred)
    _print_regression_output_set(method=method, split="val", values=val_pred)
    if debug:
        _print_regression_debug_samples(
            method=method,
            split="train",
            features=np.asarray(train_data.feature_matrix),
            pred=train_pred,
            max_samples=debug_max_samples,
            feature_head=debug_feature_head,
        )
        _print_regression_debug_samples(
            method=method,
            split="val",
            features=np.asarray(val_data.feature_matrix),
            pred=val_pred,
            max_samples=debug_max_samples,
            feature_head=debug_feature_head,
        )
    _log_regression_diagnostics(
        method=method,
        split="train",
        data=train_data,
        propensity=train_propensity,
        propensity_clipping=propensity_clipping,
        pred=train_pred,
        tracker=tracker,
    )
    _log_regression_diagnostics(
        method=method,
        split="val",
        data=val_data,
        propensity=val_propensity,
        propensity_clipping=propensity_clipping,
        pred=val_pred,
        tracker=tracker,
    )
    return RegressionBundle(model=model, train_pred=train_pred, val_pred=val_pred)


__all__ = [
    "RegressionBundle",
    "predict_regression_model",
    "train_regression_bundle",
    "train_regression_model",
]
