# Copyright (C) H.R. Oosterhuis 2022.
# Distributed under the MIT License (see the accompanying README.md and LICENSE files).

from __future__ import annotations

from functools import partial
from typing import Optional

import jax
import jax.numpy as jnp

_NEG_INF = -1e9
_NORM_EPS = 1e-12


def _ensure_rng_key(rng_key: Optional[jax.Array]) -> jax.Array:
    return jax.random.PRNGKey(0) if rng_key is None else rng_key


def pad_by_doc_id(
    values: jax.Array,
    doc_id_map: jax.Array,
    *,
    fill_value: float = 0.0,
) -> jax.Array:
    values = jnp.asarray(values)
    doc_id_map = jnp.asarray(doc_id_map)
    n_docs = values.shape[0]
    pad_shape = (1,) + values.shape[1:]
    pad = jnp.full(pad_shape, fill_value, dtype=values.dtype)
    safe_values = jnp.concatenate([values, pad], axis=0)
    safe_ids = jnp.where(doc_id_map >= 0, doc_id_map, n_docs)
    return safe_values[safe_ids]


@partial(jax.jit, static_argnames=("n_samples", "cutoff", "return_full_rankings"))
def gumbel_sample_rankings(
    log_scores,
    n_samples: int,
    cutoff: int,
    return_full_rankings: bool = False,
    rng_key: Optional[jax.Array] = None,
):
    log_scores = jnp.asarray(log_scores)
    n_docs = log_scores.shape[0]
    ranking_len = min(n_docs, cutoff)
    rng_key = _ensure_rng_key(rng_key)

    gumbel_samples = jax.random.gumbel(
        rng_key, shape=(n_samples, n_docs), dtype=log_scores.dtype
    )
    gumbel_scores = log_scores[None, :] + gumbel_samples

    rankings, _ = multiple_cutoff_rankings(
        -gumbel_scores,
        ranking_len,
        return_full_rankings=return_full_rankings,
    )
    return rankings


@jax.jit
def metrics_based_on_samples(
    sampled_rankings,
    weight_per_rank,
    addition_per_rank,
    weight_per_doc,
):
    cutoff = sampled_rankings.shape[1]
    weighted_docs = weight_per_doc[sampled_rankings] * weight_per_rank[None, :cutoff]
    return jnp.sum(jnp.mean(weighted_docs, axis=0) + addition_per_rank[:cutoff], axis=0)


def _per_query_metrics(
    q_scores: jax.Array,
    q_doc_weights: jax.Array,
    q_mask: jax.Array,
    weight_per_rank: jax.Array,
    addition_per_rank: jax.Array,
    n_samples: int,
) -> jax.Array:
    has_weight = jnp.any(jnp.not_equal(q_doc_weights, 0.0))
    n_valid = jnp.sum(q_mask, dtype=jnp.int32)
    rank_mask = jnp.arange(weight_per_rank.shape[0]) < n_valid
    weight_per_rank = weight_per_rank * rank_mask[:, None]
    addition_per_rank = addition_per_rank * rank_mask[:, None]

    def compute(_):
        sampled_rankings = gumbel_sample_rankings(
            q_scores,
            n_samples,
            cutoff=weight_per_rank.shape[0],
        )
        return metrics_based_on_samples(
            sampled_rankings,
            weight_per_rank,
            addition_per_rank,
            q_doc_weights[:, None],
        )

    return jax.lax.cond(
        has_weight,
        compute,
        lambda _: jnp.zeros((weight_per_rank.shape[1],), dtype=weight_per_rank.dtype),
        operand=None,
    )


def _datasplit_metrics_core(
    policy_scores: jax.Array,
    weight_per_rank: jax.Array,
    addition_per_rank: jax.Array,
    weight_per_doc: jax.Array,
    doc_id_map: jax.Array,
    mask: jax.Array,
    n_samples: int,
) -> jax.Array:
    q_scores = pad_by_doc_id(policy_scores, doc_id_map, fill_value=_NEG_INF)
    q_weights = pad_by_doc_id(weight_per_doc, doc_id_map, fill_value=0.0)
    q_mask = mask.astype(bool)
    q_scores = jnp.where(q_mask, q_scores, _NEG_INF)
    q_weights = jnp.where(q_mask, q_weights, 0.0)

    def scan_body(carry, inputs):
        q_scores_i, q_weights_i, q_mask_i = inputs
        metrics = _per_query_metrics(
            q_scores_i,
            q_weights_i,
            q_mask_i,
            weight_per_rank,
            addition_per_rank,
            n_samples,
        )
        return carry, metrics

    _, results = jax.lax.scan(scan_body, None, (q_scores, q_weights, q_mask))
    return results


@partial(jax.jit, static_argnames=("n_samples", "use_norm"))
def _datasplit_metrics_jax(
    policy_scores: jax.Array,
    weight_per_rank: jax.Array,
    addition_per_rank: jax.Array,
    weight_per_doc: jax.Array,
    doc_id_map: jax.Array,
    mask: jax.Array,
    query_norm_factors: jax.Array,
    n_samples: int,
    use_norm: bool,
) -> jax.Array:
    results = _datasplit_metrics_core(
        policy_scores,
        weight_per_rank,
        addition_per_rank,
        weight_per_doc,
        doc_id_map,
        mask,
        n_samples,
    )
    if use_norm:
        norm = jnp.asarray(query_norm_factors)
        safe_norm = jnp.where(
            jnp.isfinite(norm) & (jnp.abs(norm) > _NORM_EPS),
            norm,
            jnp.ones_like(norm),
        )
        results = results / safe_norm
    return jnp.mean(results, axis=0)


def datasplit_metrics(
    data_split,
    policy_scores,
    weight_per_rank,
    addition_per_rank,
    weight_per_doc,
    query_norm_factors,
    n_samples,
):
    policy_scores = jnp.asarray(policy_scores).reshape(-1)
    weight_per_doc = jnp.asarray(weight_per_doc).reshape(-1)
    weight_per_rank = jnp.asarray(weight_per_rank)
    addition_per_rank = jnp.asarray(addition_per_rank)

    if not hasattr(data_split, "doc_id_map") or not hasattr(data_split, "mask"):
        cutoff = weight_per_rank.shape[0]
        n_queries = data_split.num_queries()
        results = jnp.zeros((n_queries, weight_per_rank.shape[1]))
        for qid in range(n_queries):
            q_doc_weights = jnp.asarray(
                data_split.query_values_from_vector(qid, weight_per_doc)
            )
            if jnp.all(jnp.equal(q_doc_weights, 0.0)):
                continue
            q_policy_scores = jnp.asarray(
                data_split.query_values_from_vector(qid, policy_scores)
            )
            sampled_rankings = gumbel_sample_rankings(
                q_policy_scores,
                n_samples,
                cutoff=cutoff,
            )
            results = results.at[qid].set(
                metrics_based_on_samples(
                    sampled_rankings,
                    weight_per_rank,
                    addition_per_rank,
                    q_doc_weights[:, None],
                )
            )
        if query_norm_factors is not None:
            norm = jnp.asarray(query_norm_factors)
            safe_norm = jnp.where(
                jnp.isfinite(norm) & (jnp.abs(norm) > _NORM_EPS),
                norm,
                jnp.ones_like(norm),
            )
            results = results / safe_norm
        return jnp.mean(results, axis=0)

    doc_id_map = jnp.asarray(data_split.doc_id_map)
    mask = jnp.asarray(data_split.mask)

    n_queries = doc_id_map.shape[0]
    n_metrics = weight_per_rank.shape[1]
    if query_norm_factors is None:
        dummy_norm = jnp.ones((n_queries, n_metrics), dtype=weight_per_rank.dtype)
        return _datasplit_metrics_jax(
            policy_scores,
            weight_per_rank,
            addition_per_rank,
            weight_per_doc,
            doc_id_map,
            mask,
            dummy_norm,
            n_samples=n_samples,
            use_norm=False,
        )

    return _datasplit_metrics_jax(
        policy_scores,
        weight_per_rank,
        addition_per_rank,
        weight_per_doc,
        doc_id_map,
        mask,
        jnp.asarray(query_norm_factors),
        n_samples=n_samples,
        use_norm=True,
    )


@partial(jax.jit, static_argnames=("n_samples",))
def gradient_based_on_samples(rank_weights, labels, scores, n_samples):
    rank_weights = jnp.asarray(rank_weights)
    labels = jnp.asarray(labels)
    scores = jnp.asarray(scores)
    n_docs = labels.shape[0]
    cutoff = min(rank_weights.shape[0], n_docs)
    if n_docs == 1:
        return jnp.zeros_like(scores)

    scores = scores - jnp.amax(scores) + 10.0
    sampled_rankings = gumbel_sample_rankings(
        scores,
        n_samples,
        cutoff=cutoff,
        return_full_rankings=True,
    )
    cutoff_sampled_rankings = sampled_rankings[:, :cutoff]

    weighted_labels = labels[cutoff_sampled_rankings] * rank_weights[None, :cutoff]
    cumsum_labels = jnp.cumsum(weighted_labels[:, ::-1], axis=1)[:, ::-1]

    result = jnp.zeros(n_docs, dtype=jnp.float32)
    result = jnp.add.at(
        result,
        cutoff_sampled_rankings[:, :-1],
        cumsum_labels[:, 1:],
        inplace=False,
    )
    result = result / n_samples

    exp_scores = jnp.exp(scores)
    denom_per_rank = jnp.cumsum(exp_scores[sampled_rankings[:, ::-1]], axis=1)[:, :-cutoff - 1 : -1]

    # Queries with padded docs can have trailing ranks where all remaining
    # exp-scores are zero. Use a safe denominator and explicitly zero those
    # terms to avoid 0/0 -> NaN in the policy-gradient estimator.
    denom_is_positive = denom_per_rank > 0.0
    safe_denom = jnp.where(denom_is_positive, denom_per_rank, 1.0)

    rank_weights_over_denom = jnp.where(
        denom_is_positive,
        rank_weights[None, :cutoff] / safe_denom,
        0.0,
    )
    cumsum_weight_denom = jnp.cumsum(rank_weights_over_denom, axis=1)

    reward_over_denom = jnp.where(
        denom_is_positive,
        cumsum_labels / safe_denom,
        0.0,
    )
    cumsum_reward_denom = jnp.cumsum(reward_over_denom, axis=1)

    if cutoff < n_docs:
        second_part = -exp_scores[None, :] * cumsum_reward_denom[:, -1, None]
        nonzero_label = jnp.not_equal(labels, 0)
        positive_term = jnp.where(nonzero_label, labels * exp_scores, 0.0)
        second_part = second_part + positive_term[None, :] * cumsum_weight_denom[:, -1, None]
    else:
        second_part = jnp.zeros((n_samples, n_docs), dtype=jnp.float32)

    srange = jnp.arange(n_samples)
    sampled_direct_reward = (
        labels[cutoff_sampled_rankings]
        * exp_scores[cutoff_sampled_rankings]
        * cumsum_weight_denom
    )
    sampled_following_reward = exp_scores[cutoff_sampled_rankings] * cumsum_reward_denom
    second_part = second_part.at[srange[:, None], cutoff_sampled_rankings].set(
        sampled_direct_reward - sampled_following_reward
    )

    return result + jnp.mean(second_part, axis=0)


@partial(jax.jit, static_argnames=("cutoff", "invert", "return_full_rankings"))
def multiple_cutoff_rankings(scores, cutoff, invert=True, return_full_rankings=False):
    n_docs = scores.shape[1]
    cutoff = min(n_docs, cutoff)

    full_rankings = jnp.argsort(scores, axis=1)
    rankings = full_rankings[:, :cutoff]

    if not invert:
        inverted = None
    else:
        inverted_full = jnp.argsort(full_rankings, axis=1)
        inverted = jnp.where(inverted_full < cutoff, inverted_full, cutoff)

    if return_full_rankings:
        rankings = full_rankings

    return rankings, inverted
