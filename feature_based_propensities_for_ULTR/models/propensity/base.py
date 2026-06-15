from __future__ import annotations

from dataclasses import dataclass

from flax import nnx
from jax import Array


class BasePropensityModel(nnx.Module):
    def compute_output(self, batch: dict) -> Array:
        raise NotImplementedError

    def compute_loss(self, output, batch: dict | None = None) -> Array:
        raise NotImplementedError

    @property
    def requires_alpha(self) -> bool:
        return False

    @property
    def uses_global_groups(self) -> bool:
        return False


@dataclass
class PropensityModelSpec:
    name: str
    model: BasePropensityModel
    trainable: bool
    ckpt_dir: str | None = None
