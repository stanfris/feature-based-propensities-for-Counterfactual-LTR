from __future__ import annotations

from typing import Optional

import jax.numpy as jnp
from jax import Array
from rax._src import utils as rax_utils
from rax._src.types import ReduceFn


def pointwise_binary_cross_entropy_loss(
    scores: Array,
    labels: Array,
    *,
    where: Optional[Array] = None,
    segments: Optional[Array] = None,
    weights: Optional[Array] = None,
    reduce_fn: Optional[ReduceFn] = jnp.mean,
    eps: float = 1e-7,
) -> Array:
    """Binary cross-entropy for probability-valued scores."""
    del segments

    labels = jnp.clip(labels, 0.0, 1.0)
    scores = jnp.clip(scores, eps, 1.0 - eps)
    loss = -(labels * jnp.log(scores) + (1.0 - labels) * jnp.log1p(-scores))

    if weights is not None:
        loss *= weights

    return rax_utils.safe_reduce(loss, where=where, reduce_fn=reduce_fn)
