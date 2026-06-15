from typing import Dict

from feature_based_propensities_for_ULTR.models.utils import get_sequential
import jax.numpy as jnp
from flax import nnx
from jax import Array

class DeepRelevanceTower(nnx.Module):
    """
    Relevance tower using a feed forward network with elu activations.
    """

    def __init__(
        self,
        query_doc_features: int,
        layers: int,
        hidden_units: int,
        dropout: float,
        *,
        rngs: nnx.Rngs,
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

    def __call__(self, batch: Dict) -> Array:
        x = batch["query_doc_features"]

        for module in self.modules:
            x = module(x)

        return self.output(x).squeeze()


class EmbeddingBiasTower(nnx.Module):
    """
    Bias tower allocating a separate parameter per position \theta_{k}.
    Uses a single embedding dimension by default.
    """

    def __init__(
        self,
        positions: int,
        embedding_dims: int = 1,
        *,
        rngs: nnx.Rngs,
        **kwargs,
    ):
        super().__init__()
        self.positions = positions
        self.embedding = nnx.Embed(
            num_embeddings=positions,
            features=embedding_dims,
            rngs=rngs,
        )


    def __call__(self, batch: Dict) -> Array:
        x = batch["positions"]
        embedding = self.embedding(x).squeeze()
        embedding = jnp.atleast_2d(embedding)
        return embedding

    def get_position_bias(self) -> Array:
        positions = jnp.arange(self.positions)
        return self({"positions": positions}).squeeze()
