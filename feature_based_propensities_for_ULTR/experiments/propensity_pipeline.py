import time
from dataclasses import dataclass
import logging

import jax

logger = logging.getLogger(__name__)
import jax.numpy as jnp
import numpy as np
from omegaconf import DictConfig
from torch.utils.data import Subset

from feature_based_propensities_for_ULTR.experiments.logging_policy import (
    compute_rank_based_propensity_per_doc,
    load_logging_policy_ranker,
    logging_policy_sampler_family,
)
from feature_based_propensities_for_ULTR.models.propensity import (
    FrequencyPropensityModel,
    PropensityEstimator,
    PropensityModelSpec,
    PropensityPredictor,
    build_propensity_model_spec,
)


@dataclass(frozen=True)
class PropensityBundle:
    train: np.ndarray
    val: np.ndarray
    true_train: np.ndarray | None = None
    true_val: np.ndarray | None = None


def _stage_print(message: str) -> None:
    print(f"[PropensityPipeline] {message}", flush=True)


def _subset_click_dataset(dataset, max_samples: int | None, seed: int):
    if max_samples is None:
        return dataset
    max_samples = int(max_samples)
    if max_samples <= 0:
        return dataset
    n_total = len(dataset)
    if n_total <= max_samples:
        return dataset
    rng = np.random.default_rng(seed)
    subset_idx = rng.choice(n_total, size=max_samples, replace=False)
    logger.info(f"Subsampling propensity classifier training data: {max_samples}/{n_total} samples")
    _stage_print(f"Subsampling training data to {max_samples}/{n_total} samples")
    return Subset(dataset, subset_idx)


def build_propensity_estimator(
    *,
    config: DictConfig,
    method: str,
    train_click_dataset,
    val_click_dataset,
    alpha: np.ndarray,
    alpha_logits: np.ndarray,
    seed: int,
    checkpoint_dir: str | None,
    max_train_samples: int | None = None,
    apply_sigmoid: bool = True,
    dataloader_cfg: DictConfig | None = None,
) -> PropensityEstimator:
    _stage_print(f"Building propensity estimator for method='{method}'")
    relative_bias_logits = jnp.asarray(alpha_logits) - float(np.asarray(alpha_logits)[0])
    if method == "frequency_based":
        model_spec = PropensityModelSpec(
            name="frequency_based",
            model=FrequencyPropensityModel(),
            trainable=False,
        )
    else:
        model_spec = build_propensity_model_spec(
            config=config,
            train_dataset=train_click_dataset,
            bias_value_array=relative_bias_logits,
            seed=seed,
            method=method,
        )

    if dataloader_cfg is None:
        dataloader_cfg = {}

    # Read training hyperparameters from propensity_model.training config
    training_cfg = getattr(config.propensity_model, "training", None) or {}
    propensity_lr = float(getattr(training_cfg, "learning_rate", 5e-3))
    propensity_min_delta = float(getattr(training_cfg, "min_delta", 1e-5))

    # Read inference batch size from ips.propensity config
    propensity_cfg = getattr(getattr(config, "ips", {}), "propensity", None) or {}
    inference_batch_size = int(getattr(propensity_cfg, "inference_batch_size", 512))

    estimator = PropensityEstimator(
        model_spec=model_spec,
        alpha=alpha,
        checkpoint_dir=checkpoint_dir,
        apply_sigmoid=apply_sigmoid,
        batch_size=inference_batch_size,
        learning_rate=propensity_lr,
        min_delta=propensity_min_delta,
        dataloader_num_workers=int(getattr(dataloader_cfg, "num_workers", 0)),
        dataloader_prefetch_factor=getattr(dataloader_cfg, "prefetch_factor", None),
        dataloader_persistent_workers=bool(getattr(dataloader_cfg, "persistent_workers", False)),
        dataloader_pin_memory=bool(getattr(dataloader_cfg, "pin_memory", False)),
    )
    train_click_dataset_for_fit = train_click_dataset
    val_click_dataset_for_fit = val_click_dataset
    if model_spec.trainable:
        _stage_print(
            f"Estimator is trainable ({model_spec.name}); preparing fit datasets"
        )
        train_click_dataset_for_fit = _subset_click_dataset(
            train_click_dataset,
            max_train_samples,
            seed,
        )
        # Match validation size to training subsample for faster epochs.
        val_click_dataset_for_fit = _subset_click_dataset(
            val_click_dataset,
            max_train_samples,
            seed + 1,
        )
    _stage_print(f"Starting estimator.fit() for '{model_spec.name}'")
    estimator.fit(train_click_dataset_for_fit, val_click_dataset_for_fit)
    _stage_print(f"Finished estimator.fit() for '{model_spec.name}'")
    return estimator


def _apply_policy_temperature_to_logits(
    logits: np.ndarray,
    policy_temperature: float,
    mean_alpha_logit: float,
) -> np.ndarray:
    return (1.0 - policy_temperature) * logits + policy_temperature * mean_alpha_logit


def compute_true_propensity_per_doc(
    *,
    data,
    click_dataset,
    config: DictConfig,
    alpha_logits: np.ndarray,
    policy_temperature: float,
    cutoff: int,
) -> np.ndarray:
    """
    Compute oracle propensities by aggregating a distribution over ranks within groups
    (as in compare_propensity_param.py), then taking the expected alpha values.
    """
    alpha_logits = np.asarray(alpha_logits, dtype=np.float64)
    n_positions = int(alpha_logits.shape[0])
    cutoff = int(min(max(cutoff, 0), n_positions))

    if n_positions == 0:
        return np.zeros((data.num_docs(),), dtype=np.float64)

    sampler_family = logging_policy_sampler_family(config)
    if sampler_family == "plackett_luce":
        ranker = load_logging_policy_ranker(config, data)
        return compute_rank_based_propensity_per_doc(
            ranker=ranker,
            data=data,
            alpha=jax.nn.sigmoid(jnp.asarray(alpha_logits)).astype(np.float64),
            policy_temperature=policy_temperature,
        )

    mean_alpha_logit = float(np.mean(alpha_logits[:cutoff])) if cutoff > 0 else float(np.mean(alpha_logits))
    effective_alpha_logits = _apply_policy_temperature_to_logits(
        alpha_logits,
        policy_temperature,
        mean_alpha_logit,
    )

    alpha_probs = jax.nn.sigmoid(jnp.asarray(effective_alpha_logits)).astype(np.float64)

    predictor = PropensityPredictor(
        model=FrequencyPropensityModel(),
        alpha=np.asarray(alpha_probs),
        apply_sigmoid=False,
    )
    return predictor.predict_per_doc(data, show_progress=False)


def predict_propensity(
    *,
    config: DictConfig,
    method: str,
    train_data,
    val_data,
    train_click_dataset,
    val_click_dataset,
    alpha: np.ndarray,
    alpha_logits: np.ndarray,
    mean_alpha_logit: float,
    seed: int,
    use_true_propensity: bool,
    use_mlp_propensity: bool,
    compute_true: bool,
    checkpoint_dir: str | None = None,
    model_alpha: np.ndarray | None = None,
    model_alpha_logits: np.ndarray | None = None,
) -> PropensityBundle:
    propensity_cfg = config.ips.propensity
    propensity_max_train_samples = getattr(propensity_cfg, "max_train_samples", None)
    policy_temperature = float(getattr(config, "policy_temperature", 0.0))
    estimator_alpha = np.asarray(alpha if model_alpha is None else model_alpha)
    estimator_alpha_logits = np.asarray(alpha_logits if model_alpha_logits is None else model_alpha_logits)

    start = time.time()
    logger.info(f"Computing propensity per doc (method={method})...")
    _stage_print(f"Starting propensity computation with method='{method}'")

    if use_true_propensity:
        _stage_print("Using oracle propensity path")
        train_propensity = compute_true_propensity_per_doc(
            data=train_data,
            click_dataset=train_click_dataset,
            config=config,
            alpha_logits=np.asarray(alpha_logits),
            policy_temperature=policy_temperature,
            cutoff=int(train_data.cutoff),
        )
        val_propensity = compute_true_propensity_per_doc(
            data=val_data,
            click_dataset=val_click_dataset,
            config=config,
            alpha_logits=np.asarray(alpha_logits),
            policy_temperature=policy_temperature,
            cutoff=int(val_data.cutoff),
        )
        logger.info("Propensity computed in %.2f seconds" % (time.time() - start))
        logger.info("Alpha logits: %s", np.asarray(alpha_logits))
        logger.info("Alpha: %s", np.asarray(alpha))
        _stage_print("Oracle propensity computation finished")
        return PropensityBundle(
            train=train_propensity,
            val=val_propensity,
            true_train=train_propensity,
            true_val=val_propensity,
        )

    train_propensity_dataset = train_click_dataset if use_mlp_propensity else train_data
    val_propensity_dataset = val_click_dataset if use_mlp_propensity else val_data

    estimator = build_propensity_estimator(
        config=config,
        method=method,
        train_click_dataset=train_propensity_dataset,
        val_click_dataset=val_propensity_dataset,
        alpha=estimator_alpha,
        alpha_logits=estimator_alpha_logits,
        seed=seed,
        checkpoint_dir=checkpoint_dir,
        max_train_samples=propensity_max_train_samples,
        apply_sigmoid=False,
        dataloader_cfg=getattr(propensity_cfg, "dataloader", None),
    )

    _stage_print("Running per-document propensity inference on train split")
    train_propensity_pred = estimator.predict_per_doc(train_data)
    _stage_print("Train propensity inference finished")
    _stage_print("Running per-document propensity inference on validation split")
    val_propensity_pred = estimator.predict_per_doc(val_data)
    _stage_print("Validation propensity inference finished")

    if estimator.model_spec.name == "propensity_mlp_classifier":
        train_logits = np.asarray(train_propensity_pred, dtype=np.float64)
        val_logits = np.asarray(val_propensity_pred, dtype=np.float64)
        if policy_temperature > 0.0:
            train_logits = _apply_policy_temperature_to_logits(
                train_logits, policy_temperature, mean_alpha_logit
            )
            val_logits = _apply_policy_temperature_to_logits(
                val_logits, policy_temperature, mean_alpha_logit
            )
        train_propensity = jax.nn.sigmoid(jnp.asarray(train_logits)).astype(np.float64)
        val_propensity = jax.nn.sigmoid(jnp.asarray(val_logits)).astype(np.float64)
    elif estimator.model_spec.name == "propensity_mlp_regression":
        train_propensity = jax.nn.sigmoid(jnp.asarray(train_propensity_pred)).astype(np.float64)
        val_propensity = jax.nn.sigmoid(jnp.asarray(val_propensity_pred)).astype(np.float64)
    else:
        train_propensity = train_propensity_pred
        val_propensity = val_propensity_pred

    logger.info("Propensity computed in %.2f seconds" % (time.time() - start))
    logger.info("Alpha logits: %s", np.asarray(alpha_logits))
    logger.info("Alpha: %s", np.asarray(alpha))
    _stage_print(f"Finished propensity computation in {time.time() - start:.2f}s")

    true_train = None
    true_val = None
    if compute_true:
        true_train = compute_true_propensity_per_doc(
            data=train_data,
            click_dataset=train_click_dataset,
            config=config,
            alpha_logits=np.asarray(alpha_logits),
            policy_temperature=policy_temperature,
            cutoff=int(train_data.cutoff),
        )
        true_val = compute_true_propensity_per_doc(
            data=val_data,
            click_dataset=val_click_dataset,
            config=config,
            alpha_logits=np.asarray(alpha_logits),
            policy_temperature=policy_temperature,
            cutoff=int(val_data.cutoff),
        )

    return PropensityBundle(
        train=train_propensity,
        val=val_propensity,
        true_train=true_train,
        true_val=true_val,
    )


def compute_propensity_clipping(trainer_cfg, n_sessions_used: int) -> float | None:
    clip_propensity = bool(getattr(trainer_cfg, "clip_propensity", True))
    if clip_propensity:
        clip_base = float(getattr(trainer_cfg, "clip_base", 10.0))
        return min(clip_base / np.sqrt(max(n_sessions_used, 1)) * trainer_cfg.clip_multiplier, 1.0)
    logger.info("Propensity clipping disabled.")
    return None


__all__ = [
    "PropensityBundle",
    "build_propensity_estimator",
    "compute_propensity_clipping",
    "compute_true_propensity_per_doc",
    "predict_propensity",
]
