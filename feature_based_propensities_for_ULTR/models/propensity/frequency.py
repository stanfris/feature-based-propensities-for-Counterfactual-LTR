from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from .base import BasePropensityModel


class FrequencyPropensityModel(BasePropensityModel):
    @property
    def requires_alpha(self) -> bool:
        return True

    def compute_output(self, batch: dict) -> Array:
        mask = batch["mask"]
        n_queries, n_positions = int(mask.shape[0]), int(mask.shape[1])
        return jnp.tile(jnp.arange(n_positions, dtype=jnp.int32), (n_queries, 1))

    def compute_loss(self, output) -> Array:
        return jnp.asarray(0.0, dtype=getattr(output, "dtype", jnp.float32))
