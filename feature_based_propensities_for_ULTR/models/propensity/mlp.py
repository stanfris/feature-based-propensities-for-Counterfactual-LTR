from __future__ import annotations

import jax
import jax.numpy as jnp
import optax
from flax import nnx
from jax import Array

from feature_based_propensities_for_ULTR.models.towers import EmbeddingBiasTower
from feature_based_propensities_for_ULTR.models.utils import get_sequential
from .base import BasePropensityModel


class ClassifierPropensityMLP(BasePropensityModel):
    """
    Predicts a position for each item in a ranked list.
    Input shape: (batch_size, positions, feature_dim)
    Output shape: (batch_size, positions, positions)
    """

    def __init__(
        self,
        query_doc_features: int,
        layers: int,
        hidden_units: int,
        dropout: float,
        *,
        rngs: nnx.Rngs,
        bias_tower: EmbeddingBiasTower = None,
        bias_value_array: Array = None,
        positions: int,
        **kwargs,
    ):
        super().__init__()

        self.positions = positions

        self.mlp = nnx.Sequential(
            *get_sequential(
                features=query_doc_features,
                hidden_units=hidden_units,
                layers=layers,
                dropout=dropout,
                rngs=rngs,
            )
        )

        self.output = nnx.Linear(
            in_features=hidden_units if layers > 0 else query_doc_features,
            out_features=positions,
            rngs=rngs,
        )

        self.target_positions = nnx.Variable(
            jnp.eye(positions),
            collection="constants",
        )

        if bias_tower is not None:
            bias_tower_output = bias_tower({"positions": jnp.arange(positions)}).squeeze()
            self.bias_values = nnx.Variable(
                jax.lax.stop_gradient(bias_tower_output - bias_tower_output[0]),
                collection="constants",
            )
        else:
            if bias_value_array is None:
                raise ValueError("bias_value_array or bias_tower must be provided")

            self.bias_values = nnx.Variable(
                bias_value_array,
                collection="constants",
            )

    def __call__(self, batch: dict) -> Array:
        """
        batch["query_doc_features"]: (B, P, F)
        returns logits: (B, P, P)
        """
        x = batch["query_doc_features"]  # (B, P, F)
        x = self.mlp(x)  # (B, P, H)
        logits = self.output(x)  # (B, P, P)
        return logits

    def compute_loss(self, output, batch: dict | None = None) -> Array:
        # target_positions is (P, P); optax broadcasts it against (B, P, P) internally
        loss = optax.softmax_cross_entropy(output, self.target_positions.value)
        return loss.mean()

    def compute_output(self, batch: dict) -> Array:
        """
        Map predicted positions to bias values.
        shape: (B, P)
        """
        logits = self(batch)  # (B, P, P)
        pred_positions = logits.argmax(axis=-1)
        return self.bias_values.value[pred_positions]


class RegressionPropensityMLP(BasePropensityModel):
    """
    Regression propensity model that directly predicts the alpha (position bias)
    value for each (query, position) pair by minimising MSE loss.

    Input shape:  (batch_size, positions, feature_dim)
    Output shape: (batch_size, positions)  — one scalar per position
    """

    def __init__(
        self,
        query_doc_features: int,
        layers: int,
        hidden_units: int,
        dropout: float,
        alpha: Array,
        *,
        rngs: nnx.Rngs,
        **kwargs,
    ):
        super().__init__()

        self.mlp = nnx.Sequential(
            *get_sequential(
                features=query_doc_features,
                hidden_units=hidden_units,
                layers=layers,
                dropout=dropout,
                rngs=rngs,
            )
        )

        # Scalar output head: one value per position
        self.output = nnx.Linear(
            in_features=hidden_units if layers > 0 else query_doc_features,
            out_features=1,
            rngs=rngs,
        )

        # Target alpha values; shape (P,) — stored as a non-trainable constant
        self.alpha = nnx.Variable(
            jnp.asarray(alpha, dtype=jnp.float32),
            collection="constants",
        )

    def __call__(self, batch: dict) -> Array:
        """
        batch["query_doc_features"]: (B, P, F)  [training]
        returns predictions: (B, P)
        """
        x = batch["query_doc_features"]  # (B, P, F)
        x = self.mlp(x)                  # (B, P, H)
        x = self.output(x)               # (B, P, 1)
        return x.squeeze(-1)             # (B, P)

    def predict_per_document(self, features: Array) -> Array:
        """
        Flat per-document inference — avoids rank-ordering ambiguity.
        features: (N, F)
        returns:  (N,)
        """
        x = self.mlp(features)   # (N, H)
        x = self.output(x)       # (N, 1)
        return x.squeeze(-1)     # (N,)

    def compute_loss(self, output: Array, batch: dict | None = None) -> Array:
        """
        MSE between predicted alpha and the true position bias values.
        output: (B, P)  — model predictions from __call__
        alpha:  (P,)    — broadcast across the batch dimension
        """
        target = self.alpha.value  # (P,)
        sq_error = (output - target) ** 2
        if batch is None or "mask" not in batch:
            return jnp.mean(sq_error)

        mask = jnp.asarray(batch["mask"]).astype(bool)
        valid_count = jnp.sum(mask)
        safe_denom = jnp.maximum(valid_count, 1)
        masked_error = jnp.where(mask, sq_error, 0.0)
        return jnp.sum(masked_error) / safe_denom

    def compute_output(self, batch: dict) -> Array:
        """
        Return the predicted alpha values per (query, position).
        shape: (B, P)
        """
        return self(batch)
