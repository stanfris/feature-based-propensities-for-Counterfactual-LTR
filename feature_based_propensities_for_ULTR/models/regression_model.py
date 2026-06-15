from __future__ import annotations

from typing import Dict

import jax.numpy as jnp
from flax import nnx
from jax import Array

from feature_based_propensities_for_ULTR.models.utils import get_sequential


class RegressionModel(nnx.Module):
    """
    Regression model for DM/DR estimators.
    Produces a relevance probability in (0, 1) via a sigmoid output layer.
    """

    def __init__(
        self,
        query_doc_features: int,
        layers: int,
        hidden_units: int,
        dropout: float,
        *,
        rngs: nnx.Rngs,
        final_activation: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.modules = get_sequential(
            features=query_doc_features,
            hidden_units=hidden_units,
            layers=layers,
            dropout=dropout,
            rngs=rngs,
        )
        self.output = nnx.Linear(
            in_features=hidden_units if layers > 0 else query_doc_features,
            out_features=1,
            rngs=rngs,
        )
        self.final_activation = bool(final_activation)

    def __call__(self, batch: Dict) -> Array:
        x = batch["query_doc_features"]
        for module in self.modules:
            x = module(x)
        x = self.output(x).squeeze()
        if self.final_activation:
            x = nnx.sigmoid(x)
        return x
