"""Logging policy ranker models and their training routines."""

from __future__ import annotations

from functools import partial
from typing import Dict

import jax
import numpy as np
import optax
import rax
from flax import nnx
from flax.training.early_stopping import EarlyStopping
from optax._src.base import GradientTransformation
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from tqdm import tqdm

from feature_based_propensities_for_ULTR.data.base import RatingDataset
from feature_based_propensities_for_ULTR.models.towers import DeepRelevanceTower

from .serialization import (
    load_logging_policy_metadata,
    load_nnx_model,
    save_logging_policy_metadata,
    save_nnx_model,
)


class NeuralRanker:
    def __init__(
        self,
        max_epochs: int = 250,
        max_label: int = 4,
        batch_size: int = 512,
        patience: int = 1,
        optimizer: GradientTransformation = optax.adamw(learning_rate=0.001),
        *,
        model_type: str,
        policy_strength: float,
        random_state: int,
        training_data_percentage: float = 100.0,
    ):
        self.model_type = model_type
        self.max_epochs = max_epochs
        self.max_label = max_label
        self.batch_size = batch_size
        self.patience = patience
        self.optimizer = optimizer
        self.policy_strength = policy_strength
        self.training_data_percentage = float(training_data_percentage)
        if self.training_data_percentage <= 0.0 or self.training_data_percentage > 100.0:
            raise ValueError(
                "training_data_percentage must be in (0, 100], "
                f"got {self.training_data_percentage}."
            )
        self.random_state = int(random_state)
        self.rngs = nnx.Rngs(random_state)

    def checkpoint_metadata(self, *, n_logging_policy_features: int | None = None) -> dict:
        return {
            "class": type(self).__name__,
            "model_type": self.model_type,
            "policy_strength": float(self.policy_strength),
            "random_state": int(self.random_state),
            "training_data_percentage": float(self.training_data_percentage),
            "n_logging_policy_features": (
                None if n_logging_policy_features is None else int(n_logging_policy_features)
            ),
        }

    def fit(self, dataset: RatingDataset):
        train_dataset = dataset
        if self.training_data_percentage < 100.0:
            n_queries = len(dataset)
            n_keep = max(
                1,
                int(np.ceil(n_queries * (self.training_data_percentage / 100.0))),
            )
            rng = np.random.default_rng(self.random_state)
            selected_idx = np.sort(
                rng.choice(n_queries, size=n_keep, replace=False)
            )
            train_dataset = Subset(dataset, selected_idx.tolist())
            print(
                "Logging policy training subset: "
                f"{n_keep}/{n_queries} queries ({self.training_data_percentage:.2f}%)"
            )

        loader = DataLoader(
            train_dataset, batch_size=self.batch_size, collate_fn=dataset.collate_fn
        )

        self.model = DeepRelevanceTower(
            query_doc_features=dataset.n_logging_policy_features,
            layers=2,
            hidden_units=32,
            dropout=0.1,
            rngs=self.rngs,
        )

        optimizer = nnx.Optimizer(self.model, self.optimizer)
        early_stopping = EarlyStopping(min_delta=0.0005, patience=self.patience)
        best_state = None

        for epoch in tqdm(
            range(self.max_epochs),
            desc=(
                f"Training {self.model_type} logging policy on "
                f"{self.training_data_percentage:.2f}% of queries with "
                f"{dataset.n_logging_policy_features} available features"
            ),
        ):
            epoch_loss = 0

            for batch in loader:
                epoch_loss += self._train_step(self.model, optimizer, batch)

            epoch_loss = epoch_loss / len(loader)
            early_stopping = early_stopping.update(epoch_loss)

            if early_stopping.has_improved:
                best_state = nnx.state(self.model)

            if early_stopping.should_stop:
                nnx.update(self.model, best_state)
                break

    def __call__(
        self,
        *,
        lp_query_doc_features: np.ndarray,
        where: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        # Obtain model scores:
        self.model.eval()
        batch = {"query_doc_features": lp_query_doc_features}
        labels = self.model(batch)

        # Generate uniform random scores [0, max_label]
        random_scores = jax.random.uniform(
            self.rngs.params(),
            shape=labels.shape,
            maxval=self.max_label,
        )

        # Compute scores as an interpolation between random and label-based scores
        abs_strength = abs(self.policy_strength)
        ordered_scores = np.sign(self.policy_strength) * labels
        scores = abs_strength * ordered_scores + (1 - abs_strength) * random_scores

        return np.where(where, scores, -np.inf)

    @partial(nnx.jit, static_argnums=(0))
    def _train_step(
        self,
        model: nnx.Module,
        optimizer: nnx.Optimizer,
        batch: Dict,
    ):
        def loss_fn(model, batch):
            y_predict = model({"query_doc_features": batch["lp_query_doc_features"]})
            y = batch["labels"]
            return rax.pointwise_mse_loss(y_predict, y, where=batch["mask"])

        grad_fn = nnx.value_and_grad(loss_fn)
        loss, grads = grad_fn(model, batch)
        optimizer.update(grads)

        return loss

    def save_logging_policy(self, ckpt_dir: str = "checkpoint_lp"):
        """
        Save logging policy model parameters, excluding RNG state.
        """
        ckpt_path = save_nnx_model(self.model, ckpt_dir)
        save_logging_policy_metadata(
            self.checkpoint_metadata(),
            ckpt_dir,
        )
        print(f"Logging policy saved to {ckpt_path}")

    def load_logging_policy(
        self,
        *,
        ckpt_dir: str = "checkpoint_lp",
        n_logging_policy_features: int | None = None,
    ):
        """
        Load logging policy model parameters and recreate the model.
        """
        if n_logging_policy_features is None:
            raise ValueError(
                "n_logging_policy_features must be provided when loading a logging policy checkpoint."
            )
        # Recreate model architecture
        self.model = DeepRelevanceTower(
            query_doc_features=n_logging_policy_features,
            layers=2,
            hidden_units=32,
            dropout=0.1,
            rngs=self.rngs,
        )

        metadata = load_logging_policy_metadata(ckpt_dir)
        if metadata is not None:
            saved_model_type = metadata.get("model_type")
            if saved_model_type is not None and str(saved_model_type) != str(self.model_type):
                raise ValueError(
                    "Logging policy checkpoint model_type mismatch: "
                    f"expected '{self.model_type}', found '{saved_model_type}'."
                )

        load_nnx_model(self.model, ckpt_dir)

        print("Parameters successfully restored!")
        return
