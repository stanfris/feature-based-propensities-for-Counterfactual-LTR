import jax
import jax.numpy as jnp
import jax.lax as jlax
import jax.random as jrng
import numpy as np
from functools import partial


@partial(jax.jit, static_argnames=['num_samples'])
def gumbel_rankings(
    rng_key: jnp.ndarray,
    scores: jnp.ndarray,
    num_samples: int,
    where: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """
    Sample rankings using the Gumbel-max trick for the Plackett-Luce (PL) model.

    Args:
        rng_key: JAX PRNGKey.
        scores: [D] float vector of scores for ranking policy.
        num_samples: Integer number of rankings to sample.
        where: (Optional) [D] boolean vector masking padded items.

    Returns:
        [num_samples, D] integer matrix of sampled rankings.
    """
    n_docs = scores.shape[0]
    uniform_noise = jrng.uniform(rng_key, shape=(num_samples, n_docs))
    gumbel_noise = -jnp.log(-jnp.log(uniform_noise))
    scores = scores[None, :] + gumbel_noise
    if where is not None:
        scores = jnp.where(where[None, :], scores, -jnp.inf)
    sampled_rankings = jnp.argsort(scores, axis=-1, descending=True)
    return sampled_rankings


@partial(jax.jit, static_argnames=['num_samples'])
def exposure(
    rng_key: jnp.ndarray,
    scores: jnp.ndarray,
    exposure_per_rank: jnp.ndarray,
    where: jnp.ndarray | None = None,
    num_samples: int = 100,
) -> jnp.ndarray:
    """
    Computes the pseudo-gradient expected exposure under a sequence of Gumbel samples.

    Args:
        rng_key: JAX PRNGKey for Gumbel sampling.
        scores: [D] float vector of scores for the ranking policy.
        exposure_per_rank: [K] float vector of exposure weights per rank.
        where: (Optional) [D] boolean vector indicating which scores are not padding.
        num_samples: (Optional) integer indicating number of rankings to sample.

    Returns:
        [D] float vector indicating the expected exposure per document.

    Notes:
        Gradient w.r.t. `scores` and `exposure_per_rank` are supported.
    """
    K = exposure_per_rank.shape[0]
    if where is not None:
        n_docs = jnp.sum(where)
        scores = jnp.where(where, scores, -jnp.inf)
    else:
        n_docs = scores.shape[0]
    # rankings [N, D] matrix of sampled rankings from PL ranking model
    rankings = gumbel_rankings(rng_key, scores, num_samples, where=where)
    ranked_scores = scores[rankings]
    if where is not None:
        ranked_where = where[rankings]
    # [N, K] log denominator of placement prob. per position per sample
    log_denom = jlax.cumlogsumexp(ranked_scores, axis=1, reverse=True)[:, :K]
    # the following masking avoids division by zero
    if where is not None:
        log_denom = jnp.where(ranked_where[:, :K], log_denom, 0)
    # [N, K-1] log prob. of sampled ranking up to pos. k (not including)
    log_prefix = ranked_scores[:, :K-1] - log_denom[:, :K-1]
    log_prefix = jnp.cumsum(jnp.pad(log_prefix, ((0, 0), (1, 0))), axis=1)
    if where is not None:
        log_prefix = jnp.where(ranked_where[:, :K], log_prefix, 0)
    # [N, K, D] placement log prob. of every document per position & sample
    logprob = scores[None, None, :] - log_denom[:, :, None]
    # placement mask indicates which documents are placed at previous ranks
    placement_mask = jnp.zeros(
        (logprob.shape[0], logprob.shape[1] + 1, logprob.shape[2]), dtype=bool
    )
    placement_mask = placement_mask.at[
        jnp.arange(num_samples)[:, None],
        jnp.arange(1, K)[None, :],
        rankings[:, :(K-1)],
    ].set(True)
    placement_mask = placement_mask.at[:, jnp.minimum(n_docs, K), :].set(True)
    placement_mask = jnp.cumsum(placement_mask[:, :-1, :], axis=1).astype(bool)
    masked_logprob = jnp.where(placement_mask, -jnp.inf, logprob)
    # [N, K, D] placement prob. of every document per position & sample
    prob = jnp.exp(masked_logprob)
    # [N, K, D] loss value used for computing policy gradient
    loss_K_D = prob + jlax.stop_gradient(prob) * log_prefix[:, :, None]
    # stop_gradient used because prob gradient comes from loss_K_D
    # but gradient through exposure_per_rank is maintained
    exposure_prob_prod = jlax.stop_gradient(prob) * exposure_per_rank[None, :, None]
    # [N, D] exposure of document per sample
    sample_exposure = jnp.sum(exposure_prob_prod, axis=1)
    # [D] expected exposure of document estimated from samples
    mean_exposure = jnp.mean(sample_exposure, axis=0)
    # [N, K, D] leave-one-out baseline corrections
    baseline = mean_exposure[None, :] - sample_exposure / num_samples
    baseline *= num_samples / (num_samples - 1)
    baseline = jlax.stop_gradient(baseline)
    # [N, K, D] baseline-corrected rewards for every doc., pos., sample
    # gradient stopped for exposure_per_rank as this loss is for scores only
    rewards = (
        jlax.stop_gradient(exposure_per_rank[None, :, None]) - baseline[:, None, :]
    )
    # loss covering placement actions in the top-K ranking
    loss_inside_K = jnp.sum(rewards * loss_K_D, axis=1)
    # probability of not being placed in a ranking per document, given the K-1
    # positions of the sampled rankings, this means 0 exposure is given
    no_prob = 1 - prob[:, K-1, :]
    # loss for the action of not placing a document in the top-K ranking
    loss_outside_K = no_prob + jlax.stop_gradient(no_prob) * log_prefix[:, K-1, None]
    loss_outside_K = jnp.where(placement_mask[:, K-1, :], 0, loss_outside_K)
    if where is not None:
        loss_outside_K = jnp.where(where[None, :], loss_outside_K, 0)
    loss_outside_K *= -baseline
    loss = jnp.mean(loss_inside_K + loss_outside_K, axis=0)
    # trick to make value the expected exposure but gradient come from loss
    # note that mean_exposure has its gradient to scores blocked already
    grad_loss = mean_exposure + loss - jlax.stop_gradient(loss)
    return grad_loss


@partial(jax.jit, static_argnames=['num_samples'])
def exposure_objective(
    rng_key: jnp.ndarray,
    scores: jnp.ndarray,
    exposure_per_rank: jnp.ndarray,
    doc_rewards: jnp.ndarray,
    where: jnp.ndarray | None = None,
    num_samples: int = 100,
) -> jnp.ndarray:
    """
    Scalar ranking objective based on expected exposure.

    Computes the dot product of the expected exposure vector (under the
    Plackett-Luce policy defined by `scores`) with `doc_rewards`, giving a
    differentiable scalar suitable for use as a training loss.

    INPUT
      rng_key          : JAX PRNGKey for Gumbel sampling
      scores           : [D] float – policy scores (will be differentiated)
      exposure_per_rank: [K] float – exposure weight per rank (e.g. 1/log2(k+1))
      doc_rewards      : [D] float – per-document reward (IPS/DR/DM weights);
                         gradient is stopped internally
      where            : (optional) [D] bool – padding mask
      num_samples      : (static) int – MC samples for the exposure estimator
    OUTPUT
      scalar – weighted expected exposure; maximise this objective
    GRADIENT
      Differentiable w.r.t. `scores` via the policy-gradient trick in exposure()
    """
    exp = exposure(rng_key, scores, exposure_per_rank, where=where, num_samples=num_samples)
    return jnp.sum(exp * jlax.stop_gradient(doc_rewards))


__all__ = [
    "gumbel_rankings",
    "exposure",
    "exposure_objective",
]