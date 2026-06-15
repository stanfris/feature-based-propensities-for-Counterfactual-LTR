# Copyright (C) H.R. Oosterhuis 2022.
# Distributed under the MIT License (see the accompanying README.md and LICENSE files).

import jax
import jax.numpy as jnp
import numpy as np

import feature_based_propensities_for_ULTR.ranking.plackettluce as pl


def _metric_cutoff(data_split, click_alpha) -> int:
    split_cutoff = getattr(data_split, "cutoff", None)
    if split_cutoff is not None:
        return int(split_cutoff)
    return int(jnp.asarray(click_alpha).shape[0])


def _policy_scores(model, feature_matrix, multiplier: float) -> jnp.ndarray:
    if hasattr(model, "score_feature_matrix"):
        policy_scores = model.score_feature_matrix(jnp.asarray(feature_matrix))
    else:
        policy_scores = model(feature_matrix)
    if hasattr(policy_scores, "numpy"):
        policy_scores = policy_scores.numpy()
    return jnp.asarray(policy_scores).reshape(-1) * multiplier


@jax.jit
def _prepare_metric_weights(click_alpha: jnp.ndarray):
    cutoff = click_alpha.shape[0]
    dcg_weights = 1.0 / jnp.log2(jnp.arange(cutoff) + 2.0)
    stacked_alphas = jnp.stack([click_alpha, click_alpha, dcg_weights], axis=-1)
    stacked_additions = jnp.zeros_like(stacked_alphas)
    return stacked_alphas, stacked_additions


@jax.jit
def _expand_with_normalization(
    stacked_alphas: jnp.ndarray,
    stacked_additions: jnp.ndarray,
    norm_factors: jnp.ndarray,
):
    stacked_alphas = stacked_alphas[:, [0, 0, 1, 1, 2, 2]]
    stacked_additions = stacked_additions[:, [0, 0, 1, 1, 2, 2]]
    ones = jnp.ones_like(norm_factors[:, 0])
    norm_factors = jnp.stack(
        [
            norm_factors[:, 0],
            ones,
            norm_factors[:, 1],
            ones,
            norm_factors[:, 2],
            ones,
        ],
        axis=-1,
    )
    return stacked_alphas, stacked_additions, norm_factors


def evaluate_policy(
    model,
    data_split,
    doc_weights,
    click_alpha,
    n_samples,
    model_output_multiplier,
):
    cutoff = _metric_cutoff(data_split, click_alpha)
    stacked_alphas, stacked_additions = _prepare_metric_weights(jnp.asarray(click_alpha))
    norm_factors = max_score_per_query(
        data_split,
        doc_weights,
        stacked_alphas,
        stacked_additions,
    )
    stacked_alphas, stacked_additions, norm_factors = _expand_with_normalization(
        stacked_alphas,
        stacked_additions,
        norm_factors,
    )

    policy_scores = _policy_scores(
        model,
        data_split.feature_matrix,
        model_output_multiplier,
    )
    metrics = pl.datasplit_metrics(
        data_split,
        policy_scores,
        stacked_alphas,
        stacked_additions,
        doc_weights,
        query_norm_factors=norm_factors,
        n_samples=n_samples,
    )
    result = {
        "NCTR": metrics[0],
        "CTR": metrics[1],
        "NRCTR": metrics[2],
        "RCTR": metrics[3],
        "NDCG": metrics[4],
        f"NDCG@{cutoff}": metrics[4],
        "DCG": metrics[5],
    }
    for k, v in result.items():
        result[k] = float(v)
    return result


def _deterministic_query_metrics(
    q_scores: np.ndarray,
    q_doc_weights: np.ndarray,
    q_mask: np.ndarray,
    weight_per_rank: np.ndarray,
    addition_per_rank: np.ndarray,
) -> np.ndarray:
    q_mask = np.asarray(q_mask, dtype=bool)
    q_scores = np.asarray(q_scores, dtype=np.float64)
    q_doc_weights = np.asarray(q_doc_weights, dtype=np.float64)
    n_metrics = int(weight_per_rank.shape[1])
    if not np.any(q_mask) or not np.any(q_doc_weights != 0.0):
        return np.zeros((n_metrics,), dtype=np.float64)

    n_valid = int(np.sum(q_mask))
    rank_mask = (np.arange(weight_per_rank.shape[0]) < n_valid)[:, None]
    effective_rank_weights = weight_per_rank * rank_mask
    effective_additions = addition_per_rank * rank_mask

    valid_idx = np.flatnonzero(q_mask)
    ranked_valid = valid_idx[np.argsort(-q_scores[valid_idx], kind="stable")]
    ranking = jnp.asarray(ranked_valid[None, : min(weight_per_rank.shape[0], ranked_valid.shape[0])])
    metrics = pl.metrics_based_on_samples(
        ranking,
        jnp.asarray(effective_rank_weights),
        jnp.asarray(effective_additions),
        jnp.asarray(q_doc_weights[:, None]),
    )
    return np.asarray(metrics, dtype=np.float64)


def evaluate_policy_direct(
    model,
    data_split,
    doc_weights,
    click_alpha,
    model_output_multiplier,
):
    cutoff = _metric_cutoff(data_split, click_alpha)
    stacked_alphas, stacked_additions = _prepare_metric_weights(jnp.asarray(click_alpha))
    norm_factors = max_score_per_query(
        data_split,
        doc_weights,
        stacked_alphas,
        stacked_additions,
    )
    stacked_alphas, stacked_additions, norm_factors = _expand_with_normalization(
        stacked_alphas,
        stacked_additions,
        norm_factors,
    )

    policy_scores = np.asarray(
        _policy_scores(
            model,
            data_split.feature_matrix,
            model_output_multiplier,
        )
    ).reshape(-1)
    weight_per_doc = np.asarray(doc_weights, dtype=np.float64).reshape(-1)
    weight_per_rank = np.asarray(stacked_alphas, dtype=np.float64)
    addition_per_rank = np.asarray(stacked_additions, dtype=np.float64)
    norm_factors = np.asarray(norm_factors, dtype=np.float64)

    if hasattr(data_split, "doc_id_map") and hasattr(data_split, "mask"):
        doc_id_map = np.asarray(data_split.doc_id_map)
        mask = np.asarray(data_split.mask, dtype=bool)
        q_scores = np.asarray(pl.pad_by_doc_id(policy_scores, doc_id_map, fill_value=-1e9))
        q_weights = np.asarray(pl.pad_by_doc_id(weight_per_doc, doc_id_map, fill_value=0.0))
        q_scores = np.where(mask, q_scores, -1e9)
        q_weights = np.where(mask, q_weights, 0.0)
        results = np.stack(
            [
                _deterministic_query_metrics(
                    q_scores[qid],
                    q_weights[qid],
                    mask[qid],
                    weight_per_rank,
                    addition_per_rank,
                )
                for qid in range(doc_id_map.shape[0])
            ],
            axis=0,
        )
    else:
        n_queries = data_split.num_queries()
        results = np.zeros((n_queries, weight_per_rank.shape[1]), dtype=np.float64)
        for qid in range(n_queries):
            q_doc_weights = np.asarray(data_split.query_values_from_vector(qid, weight_per_doc))
            if np.all(q_doc_weights == 0.0):
                continue
            q_scores = np.asarray(data_split.query_values_from_vector(qid, policy_scores))
            q_mask = np.ones_like(q_scores, dtype=bool)
            results[qid] = _deterministic_query_metrics(
                q_scores,
                q_doc_weights,
                q_mask,
                weight_per_rank,
                addition_per_rank,
            )

    safe_norm = np.where(
        np.isfinite(norm_factors) & (np.abs(norm_factors) > 1e-12),
        norm_factors,
        1.0,
    )
    results = results / safe_norm
    metrics = np.mean(results, axis=0)

    result = {
        "NCTR": metrics[0],
        "CTR": metrics[1],
        "NRCTR": metrics[2],
        "RCTR": metrics[3],
        "NDCG": metrics[4],
        f"NDCG@{cutoff}": metrics[4],
        "DCG": metrics[5],
    }
    for k, v in result.items():
        result[k] = float(v)
    return result


def evaluate_policy_with_score_randomization(
    data_split,
    doc_weights,
    click_alpha,
    *,
    policy_scores,
    random_prob: float,
    n_samples: int,
):
    cutoff = _metric_cutoff(data_split, click_alpha)
    stacked_alphas, stacked_additions = _prepare_metric_weights(jnp.asarray(click_alpha))
    norm_factors = max_score_per_query(
        data_split,
        doc_weights,
        stacked_alphas,
        stacked_additions,
    )
    stacked_alphas, stacked_additions, norm_factors = _expand_with_normalization(
        stacked_alphas,
        stacked_additions,
        norm_factors,
    )

    policy_scores = jnp.asarray(policy_scores).reshape(-1)
    metrics_deterministic = pl.datasplit_metrics(
        data_split,
        policy_scores,
        stacked_alphas,
        stacked_additions,
        doc_weights,
        query_norm_factors=norm_factors,
        n_samples=n_samples,
    )

    if random_prob <= 0.0:
        metrics = metrics_deterministic
    else:
        random_scores = np.random.uniform(0, 1, size=policy_scores.shape)
        metrics_random = pl.datasplit_metrics(
            data_split,
            random_scores,
            stacked_alphas,
            stacked_additions,
            doc_weights,
            query_norm_factors=norm_factors,
            n_samples=n_samples,
        )
        metrics = (1.0 - random_prob) * metrics_deterministic + random_prob * metrics_random

    result = {
        "NCTR": metrics[0],
        "CTR": metrics[1],
        "NRCTR": metrics[2],
        "RCTR": metrics[3],
        "NDCG": metrics[4],
        f"NDCG@{cutoff}": metrics[4],
        "DCG": metrics[5],
    }
    for k, v in result.items():
        result[k] = float(v)
    return result


@jax.jit
def _max_score_per_query_jax(
    weight_per_doc: jnp.ndarray,
    weight_per_rank: jnp.ndarray,
    addition_per_rank: jnp.ndarray,
    doc_id_map: jnp.ndarray,
    mask: jnp.ndarray,
) -> jnp.ndarray:
    q_doc_weights = pl.pad_by_doc_id(weight_per_doc, doc_id_map, fill_value=0.0)
    q_mask = mask.astype(bool)
    q_doc_weights = jnp.where(q_mask, q_doc_weights, 0.0)
    n_valid = jnp.sum(q_mask, axis=1, dtype=jnp.int32)
    cutoff = weight_per_rank.shape[0]
    rank_mask = jnp.arange(cutoff)[None, :] < n_valid[:, None]

    sort_i = jnp.argsort(-weight_per_rank, axis=0)
    metric_idx = jnp.arange(weight_per_rank.shape[1])[None, :]
    sorted_weights = weight_per_rank[sort_i, metric_idx]
    sorted_additions = addition_per_rank[sort_i, metric_idx]

    best_rankings = jnp.argsort(-q_doc_weights, axis=1)
    best_rankings = best_rankings[:, :cutoff]

    effective_weights = sorted_weights[None, :, :] * rank_mask[:, :, None]
    effective_additions = sorted_additions[None, :, :] * rank_mask[:, :, None]

    def per_query(weights, ranking, eff_weights, eff_additions):
        return pl.metrics_based_on_samples(
            ranking[None, :],
            eff_weights,
            eff_additions,
            weights[:, None],
        )

    return jax.vmap(per_query)(q_doc_weights, best_rankings, effective_weights, effective_additions)


def max_score_per_query(
    data_split,
    weight_per_doc,
    weight_per_rank,
    addition_per_rank,
):
    if not hasattr(data_split, "doc_id_map") or not hasattr(data_split, "mask"):
        weight_per_rank = jnp.asarray(weight_per_rank)
        addition_per_rank = jnp.asarray(addition_per_rank)
        cutoff = weight_per_rank.shape[0]
        n_queries = data_split.num_queries()
        results = jnp.zeros((n_queries, weight_per_rank.shape[1]))

        sort_i = jnp.argsort(-weight_per_rank, axis=0)
        metric_idx = jnp.arange(weight_per_rank.shape[1])[None, :]
        sorted_weights = weight_per_rank[sort_i, metric_idx]
        sorted_additions = addition_per_rank[sort_i, metric_idx]

        for qid in range(n_queries):
            q_doc_weights = jnp.asarray(
                data_split.query_values_from_vector(qid, weight_per_doc)
            )
            best_ranking = jnp.argsort(-q_doc_weights)[:cutoff]
            results = results.at[qid].set(
                pl.metrics_based_on_samples(
                    best_ranking[None, :],
                    sorted_weights,
                    sorted_additions,
                    q_doc_weights[:, None],
                )
            )
        return results

    return _max_score_per_query_jax(
        jnp.asarray(weight_per_doc).reshape(-1),
        jnp.asarray(weight_per_rank),
        jnp.asarray(addition_per_rank),
        jnp.asarray(data_split.doc_id_map),
        jnp.asarray(data_split.mask),
    )
