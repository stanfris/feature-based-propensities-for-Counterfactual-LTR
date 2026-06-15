from dataclasses import dataclass
from pathlib import Path
import logging
import shutil

import numpy as np
from hydra.utils import instantiate
from omegaconf import DictConfig

from feature_based_propensities_for_ULTR.data.feature_validation import validate_expected_feature_dim

from .config_resolver import (
    click_signature,
    prefixed_artifact_name,
)

logger = logging.getLogger(__name__)


@dataclass
class DominantPositionLookup:
    pos_lookup: list
    query_to_index: dict
    rank_pos: np.ndarray


def logging_policy_sampler_family(config: DictConfig) -> str:
    sampler_cfg = getattr(config, "logging_policy_sampler", None)
    if sampler_cfg is None:
        return "unknown"
    target = str(getattr(sampler_cfg, "_target_", "")).lower()
    if "plackettlucesampler" in target:
        return "plackett_luce"
    if "egreedysampler" in target:
        return "e_greedy"
    return target.rsplit(".", maxsplit=1)[-1] or "unknown"


def _logging_policy_feature_dim(dataset) -> int:
    if hasattr(dataset, "n_logging_policy_features"):
        return int(dataset.n_logging_policy_features)
    if hasattr(dataset, "lp_query_doc_features"):
        return int(np.asarray(dataset.lp_query_doc_features).shape[-1])
    raise ValueError("Dataset does not expose lp_query_doc_features for logging policy.")
def resolve_logging_policy_ckpt_dir(config: DictConfig) -> Path:
    try:
        save_subdir = config.ips.save_subdir
        dataset_dir = Path(config.dataset_dir).expanduser()
        save_dir = dataset_dir / save_subdir
        return save_dir / prefixed_artifact_name(
            config,
            f"logging_policy_{click_signature(config, 'train')}",
        )
    except Exception:
        return Path("checkpoint_lp")


def _delete_corrupt_checkpoint(ckpt_dir: Path) -> None:
    """Best-effort deletion of a corrupt logging-policy checkpoint."""
    try:
        if not ckpt_dir.exists():
            return
        if ckpt_dir.is_dir():
            shutil.rmtree(ckpt_dir)
        else:
            ckpt_dir.unlink()
        logger.info("Deleted corrupt logging-policy checkpoint: %s", ckpt_dir)
    except OSError as exc:
        logger.warning(
            "Failed to delete corrupt logging-policy checkpoint %s (%s: %s).",
            ckpt_dir,
            type(exc).__name__,
            exc,
        )


def load_logging_policy_ranker(config: DictConfig, dataset) -> object:
    """Load or train the deterministic logging policy ranker used in click simulation."""
    ranker = instantiate(config.logging_policy_ranker)
    ckpt_dir = resolve_logging_policy_ckpt_dir(config)

    if ckpt_dir.exists():
        try:
            ranker.load_logging_policy(
                ckpt_dir=str(ckpt_dir),
                n_logging_policy_features=_logging_policy_feature_dim(dataset),
            )
            return ranker
        except Exception as exc:
            logger.warning(
                "Failed to load cached logging-policy checkpoint (%s: %s). "
                "Deleting corrupt checkpoint and re-training...",
                type(exc).__name__,
                exc,
            )
            _delete_corrupt_checkpoint(ckpt_dir)

    data = instantiate(config.data.dataset)
    preprocessor = instantiate(config.data.preprocessor)
    train_dataset = preprocessor(data.load("train"), split="train")
    validate_expected_feature_dim(
        config,
        _logging_policy_feature_dim(train_dataset),
        source="train_dataset.lp_query_doc_features",
    )
    ranker.fit(train_dataset)
    ranker.save_logging_policy(ckpt_dir=str(ckpt_dir))
    return ranker


def build_dominant_position_lookup(ranker, dataset) -> DominantPositionLookup:
    """Map (query_id, doc_id) -> dominant position under the logging policy."""
    scores = ranker(
        lp_query_doc_features=dataset.lp_query_doc_features,
        labels=dataset.labels,
        where=dataset.mask,
    )
    scores = np.where(dataset.mask, scores, -np.inf)
    ranked_indices = np.argsort(-scores, axis=1)

    query_doc_ids = np.asarray(dataset.query_doc_ids)
    query_mask = np.asarray(dataset.mask)
    query_ids = np.asarray(dataset.query)
    query_to_index = {int(qid): idx for idx, qid in enumerate(query_ids)}

    rank_pos = np.full(ranked_indices.shape, -1, dtype=np.int64)
    row_idx = np.arange(ranked_indices.shape[0])[:, None]
    rank_pos[row_idx, ranked_indices] = np.arange(ranked_indices.shape[1])[None, :]

    pos_lookup = []
    for q in range(query_doc_ids.shape[0]):
        ids = query_doc_ids[q]
        mask = query_mask[q].astype(bool)
        pos_lookup.append(
            {int(doc_id): int(rank_pos[q, pos]) for pos, doc_id in enumerate(ids) if mask[pos]}
        )

    return DominantPositionLookup(
        pos_lookup=pos_lookup,
        query_to_index=query_to_index,
        rank_pos=rank_pos,
    )


def compute_dominant_pos_per_doc(
    data,
    lookup: DominantPositionLookup,
) -> np.ndarray:
    """Dominant position per doc using logging policy ranking."""
    doc_id_map = np.asarray(data.doc_id_map)
    rank_pos = np.asarray(lookup.rank_pos)

    n_queries = min(doc_id_map.shape[0], rank_pos.shape[0])
    n_positions = min(doc_id_map.shape[1], rank_pos.shape[1])
    doc_id_map = doc_id_map[:n_queries, :n_positions]
    rank_pos = rank_pos[:n_queries, :n_positions]

    dominant_pos = np.full(data.num_docs(), -1, dtype=np.int64)
    valid = doc_id_map >= 0
    dominant_pos[doc_id_map[valid]] = rank_pos[valid]
    return dominant_pos


def compute_rank_based_propensity_per_doc(
    *,
    ranker,
    data,
    alpha: np.ndarray,
    policy_temperature: float,
) -> np.ndarray:
    """
    Compute query-document propensities from the deterministic rank implied by
    the logging-policy scores, then smooth them toward the mean exposure as
    temperature increases.
    """
    scores = ranker(
        lp_query_doc_features=data.lp_query_doc_features,
        labels=data.labels,
        where=data.mask,
    )
    scores = np.where(np.asarray(data.mask, dtype=bool), np.asarray(scores, dtype=np.float64), -np.inf)
    ranked_indices = np.argsort(-scores, axis=1, kind="stable")

    row_idx = np.arange(ranked_indices.shape[0])[:, None]
    rank_pos = np.full(ranked_indices.shape, -1, dtype=np.int64)
    rank_pos[row_idx, ranked_indices] = np.arange(ranked_indices.shape[1])[None, :]

    doc_id_map = np.asarray(data.doc_id_map)
    query_mask = np.asarray(data.mask, dtype=bool)
    n_queries = min(doc_id_map.shape[0], rank_pos.shape[0], query_mask.shape[0])
    n_positions = min(doc_id_map.shape[1], rank_pos.shape[1], query_mask.shape[1])

    doc_id_map = doc_id_map[:n_queries, :n_positions]
    rank_pos = rank_pos[:n_queries, :n_positions]
    query_mask = query_mask[:n_queries, :n_positions]
    alpha = np.asarray(alpha, dtype=np.float64)

    mean_alpha = float(np.mean(alpha)) if alpha.size > 0 else 0.0
    smooth = float(np.clip(policy_temperature, 0.0, 1.0))

    propensity_per_doc = np.zeros((data.num_docs(),), dtype=np.float64)
    valid = (doc_id_map >= 0) & query_mask
    valid_doc_ids = doc_id_map[valid]
    valid_ranks = rank_pos[valid]
    deterministic_prop = np.zeros(valid_ranks.shape, dtype=np.float64)
    in_alpha_support = (valid_ranks >= 0) & (valid_ranks < alpha.shape[0])
    deterministic_prop[in_alpha_support] = alpha[valid_ranks[in_alpha_support]]

    propensity = (1.0 - smooth) * deterministic_prop + smooth * mean_alpha
    if smooth > 0.0:
        # If a document has zero deterministic exposure and is only shown via
        # randomization, it is equally likely to land at any rank, so its
        # propensity should be the mean exposure across ranks.
        propensity = np.where(deterministic_prop == 0.0, mean_alpha, propensity)

    propensity_per_doc[valid_doc_ids] = propensity
    return propensity_per_doc


__all__ = [
    "DominantPositionLookup",
    "build_dominant_position_lookup",
    "compute_dominant_pos_per_doc",
    "compute_rank_based_propensity_per_doc",
    "logging_policy_sampler_family",
    "load_logging_policy_ranker",
    "resolve_logging_policy_ckpt_dir",
]
