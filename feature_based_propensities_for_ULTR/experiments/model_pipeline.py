import logging
import time
from pathlib import Path

import numpy as np
import optax
from flax import nnx
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from feature_based_propensities_for_ULTR.experiments.config_resolver import resolve_data_cutoff
from feature_based_propensities_for_ULTR.experiments.logging_policy import (
    load_logging_policy_ranker,
    logging_policy_sampler_family,
)
from feature_based_propensities_for_ULTR.metrics import MRR, NDCG, NegativeLogLikelihood, PointwiseBinaryCrossEntropy
from feature_based_propensities_for_ULTR.logging_policy.samplers import plackett_luce_temperature_scale
from feature_based_propensities_for_ULTR.models.policy_models import build_policy_model
from feature_based_propensities_for_ULTR.models.towers import DeepRelevanceTower
from feature_based_propensities_for_ULTR.ranking import (
    evaluate_policy,
    evaluate_policy_direct,
    evaluate_policy_with_score_randomization,
)
from feature_based_propensities_for_ULTR.trainer import ExposurePolicyTrainer, PolicyTrainer, Trainer
logger = logging.getLogger(__name__)


def build_relevance_metrics(config: DictConfig) -> dict[str, object]:
    metrics: dict[str, object] = {"ndcg": NDCG()}
    cutoff = resolve_data_cutoff(config)
    if cutoff is not None:
        metrics[f"ndcg@{cutoff}"] = NDCG(top_k=cutoff)
    metrics["mrr@10"] = MRR(top_k=10)
    return metrics


class LoggingPolicyModelAdapter:
    def __init__(
        self,
        *,
        scores: np.ndarray,
        mask: np.ndarray,
        temperature: float,
        policy_family: str,
    ):
        self.scores = np.asarray(scores)
        self.mask = np.asarray(mask, dtype=bool)
        self.temperature = float(temperature)
        self.policy_family = str(policy_family)

    def policy_scores_for_split(self, data_split) -> np.ndarray:
        doc_id_map = np.asarray(data_split.doc_id_map)
        n_docs = int(data_split.num_docs())
        out = np.zeros((n_docs,), dtype=np.float64)

        n_queries = min(doc_id_map.shape[0], self.scores.shape[0], self.mask.shape[0])
        n_positions = min(doc_id_map.shape[1], self.scores.shape[1], self.mask.shape[1])
        ids = doc_id_map[:n_queries, :n_positions]
        score_block = np.asarray(self.scores[:n_queries, :n_positions], dtype=np.float64)
        mask_block = np.asarray(self.mask[:n_queries, :n_positions], dtype=bool)
        valid = (ids >= 0) & mask_block
        out[ids[valid]] = score_block[valid]
        return out


def build_logging_policy_model(
    *,
    config: DictConfig,
    data_split,
):
    ranker = load_logging_policy_ranker(config, data_split)
    scores = ranker(
        lp_query_doc_features=data_split.lp_query_doc_features,
        labels=data_split.labels,
        where=data_split.mask,
    )
    temperature = float(getattr(config, "policy_temperature", 0.0))
    return LoggingPolicyModelAdapter(
        scores=np.asarray(scores),
        mask=np.asarray(data_split.mask),
        temperature=temperature,
        policy_family=logging_policy_sampler_family(config),
    )


class _ScoreAdapter:
    def __init__(self, wrapped):
        self._wrapped = wrapped

    def score_feature_matrix(self, feature_matrix):
        batch = {"query_doc_features": feature_matrix}
        if hasattr(self._wrapped, "predict_relevance"):
            output = self._wrapped.predict_relevance(batch)
        else:
            output = self._wrapped(batch)
        return getattr(output, "click", output)


class _FixedScoreModel:
    def __init__(self, scores: np.ndarray):
        self._scores = np.asarray(scores, dtype=np.float64)

    def score_feature_matrix(self, _feature_matrix):
        return self._scores


def train_policy_model(
    *,
    config: DictConfig,
    model_choice: str,
    model_cfg,
    train_data,
    val_data,
    train_doc_weights,
    val_doc_weights,
    alpha,
    seed: int,
):
    trainer_cfg = config.ips.trainer

    rngs = nnx.Rngs(seed)
    relevance_tower = DeepRelevanceTower(
        query_doc_features=train_data.feature_matrix.shape[1],
        layers=model_cfg.relevance_tower.layers,
        hidden_units=model_cfg.relevance_tower.hidden_units,
        dropout=model_cfg.relevance_tower.dropout,
        rngs=rngs,
    )
    eps = float(getattr(model_cfg, "eps", 1e-8))
    model = build_policy_model(
        model_choice,
        scorer=relevance_tower,
        eps=eps,
    )

    n_samples = int(getattr(trainer_cfg, "n_samples", 100))
    trainer_type = str(getattr(trainer_cfg, "trainer_type", "pl-rank")).strip().lower()

    start = time.time()
    logger.info(f"Loading trainer (trainer_type='{trainer_type}', n_samples={n_samples})...")

    if trainer_type == "exposure":
        trainer = ExposurePolicyTrainer(
            optimizer=optax.adamw(learning_rate=trainer_cfg.learning_rate),
            n_samples=n_samples,
            n_eval_samples=trainer_cfg.n_eval_samples,
            eval_max_queries=getattr(trainer_cfg, "eval_max_queries", None),
            max_epochs=trainer_cfg.max_epochs,
            early_stop_diff=trainer_cfg.early_stop_diff,
            early_stop_per_epochs=trainer_cfg.early_stop_per_epochs,
            early_stop_min_epochs=int(getattr(trainer_cfg, "early_stop_min_epochs", 15)),
            patience=int(getattr(trainer_cfg, "patience", 3)),
            print_updates=trainer_cfg.print_updates,
            use_wandb=trainer_cfg.get("use_wandb", False),
            log_freq=trainer_cfg.get("log_freq", 100),
            rng_seed=seed,
            eval_first_epoch=bool(getattr(trainer_cfg, "eval_first_epoch", False)),
            debug_eval_train_split=bool(getattr(trainer_cfg, "eval_train_split", False)),
        )
    elif trainer_type == "pl-rank":
        trainer = PolicyTrainer(
            optimizer=optax.adamw(learning_rate=trainer_cfg.learning_rate),
            n_grad_samples=n_samples,
            n_eval_samples=trainer_cfg.n_eval_samples,
            eval_max_queries=getattr(trainer_cfg, "eval_max_queries", None),
            max_epochs=trainer_cfg.max_epochs,
            early_stop_diff=trainer_cfg.early_stop_diff,
            early_stop_per_epochs=trainer_cfg.early_stop_per_epochs,
            early_stop_min_epochs=int(getattr(trainer_cfg, "early_stop_min_epochs", 15)),
            patience=int(getattr(trainer_cfg, "patience", 3)),
            n_grad_samples_dynamic_fallback=int(getattr(trainer_cfg, "n_grad_samples_dynamic_fallback", 100)),
            print_updates=trainer_cfg.print_updates,
            use_wandb=trainer_cfg.get("use_wandb", False),
            log_freq=trainer_cfg.get("log_freq", 100),
            eval_first_epoch=bool(getattr(trainer_cfg, "eval_first_epoch", False)),
            debug_eval_train_split=bool(getattr(trainer_cfg, "eval_train_split", False)),
        )
    else:
        raise ValueError(
            f"Unknown trainer_type '{trainer_type}'. "
            "Valid options: 'pl-rank', 'exposure'."
        )

    logger.info("Trainer loaded in %.2f seconds" % (time.time() - start))

    logger.info("Training model...")
    model, _ = trainer.train(
        model=model,
        data_train=train_data,
        train_doc_weights=train_doc_weights,
        train_alpha=alpha,
        data_vali=val_data,
        vali_doc_weights=val_doc_weights,
        vali_alpha=alpha,
    )
    return model


def evaluate_model(
    *,
    model,
    test_data,
    alpha,
    n_samples: int,
    model_output_multiplier: float,
    eval_label_scale: float = 0.25,
):
    click_alpha = np.asarray(alpha, dtype=np.float64)
    true_test_doc_weights = np.asarray(test_data.label_vector, dtype=np.float64) * eval_label_scale

    if hasattr(model, "policy_scores_for_split") and hasattr(model, "temperature"):
        policy_scores = model.policy_scores_for_split(test_data) * float(model_output_multiplier)
        if getattr(model, "policy_family", "") == "plackett_luce":
            effective_scores = plackett_luce_temperature_scale(
                policy_scores,
                policy_temperature=float(model.temperature),
            )
            if float(model.temperature) <= 0.0:
                effective_scores = np.asarray(effective_scores, dtype=np.float64) * 1e6
            return evaluate_policy(
                _FixedScoreModel(effective_scores),
                test_data,
                true_test_doc_weights,
                click_alpha,
                n_samples=n_samples,
                model_output_multiplier=1.0,
            )
        return evaluate_policy_with_score_randomization(
            test_data,
            true_test_doc_weights,
            click_alpha,
            policy_scores=policy_scores,
            random_prob=float(model.temperature),
            n_samples=n_samples,
        )

    eval_model = model
    if not hasattr(model, "score_feature_matrix"):
        eval_model = _ScoreAdapter(model)

    return evaluate_policy(
        eval_model,
        test_data,
        true_test_doc_weights,
        click_alpha,
        n_samples=n_samples,
        model_output_multiplier=model_output_multiplier,
    )


__all__ = [
    "build_logging_policy_model",
    "evaluate_model",
    "train_policy_model",
]
