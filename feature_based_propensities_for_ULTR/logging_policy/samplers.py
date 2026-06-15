"""Logging policy samplers for converting scores into rankings."""

from __future__ import annotations

import numpy as np


def _valid_and_invalid_indices(where: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mask = np.asarray(where, dtype=bool)
    return np.flatnonzero(mask), np.flatnonzero(~mask)


def plackett_luce_temperature_scale(
    scores: np.ndarray,
    *,
    policy_temperature: float,
) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    temperature = float(np.clip(policy_temperature, 0.0, 1.0))
    if temperature <= 0.0:
        return scores
    centered = scores - float(np.max(scores))
    return (1.0 - temperature) * centered


class EGreedySampler:
    """
    Displays the original ranking in 1 - epsilon of the cases,
    and in epsilon of the cases displays a uniform random ranking.
    """

    policy_family = "e_greedy"

    def __init__(
        self,
        random_state: int,
        policy_temperature: float,
    ):
        self.rng = np.random.RandomState(random_state)
        self.policy_temperature = float(np.clip(policy_temperature, 0.0, 1.0))

    def __call__(
        self,
        *,
        scores: np.ndarray,
        where: np.ndarray,
    ) -> np.ndarray:
        scores = np.asarray(scores, dtype=np.float64).copy()

        if self.rng.rand() < self.policy_temperature:
            scores = self.rng.random(scores.shape)

        scores = np.where(where, scores, -np.inf)
        # Rank in descending order:
        return np.argsort(-scores)


class PlackettLuceSampler:
    """
    Samples rankings from a Plackett-Luce distribution induced by the scores.

    `policy_temperature=0` yields a deterministic ranking. As the temperature
    approaches 1, the logits collapse toward a constant and the policy tends to
    a uniform random permutation over the valid documents.
    """

    policy_family = "plackett_luce"

    def __init__(
        self,
        random_state: int,
        policy_temperature: float,
    ):
        self.rng = np.random.RandomState(random_state)
        self.policy_temperature = float(np.clip(policy_temperature, 0.0, 1.0))

    def __call__(
        self,
        *,
        scores: np.ndarray,
        where: np.ndarray,
    ) -> np.ndarray:
        scores = np.asarray(scores, dtype=np.float64)
        valid_idx, invalid_idx = _valid_and_invalid_indices(where)
        if valid_idx.size == 0:
            return invalid_idx

        valid_scores = scores[valid_idx]
        if self.policy_temperature <= 0.0:
            sampled_valid = valid_idx[np.argsort(-valid_scores, kind="stable")]
        else:
            scaled_scores = plackett_luce_temperature_scale(
                valid_scores,
                policy_temperature=self.policy_temperature,
            )
            gumbels = self.rng.gumbel(size=scaled_scores.shape[0])
            sampled_valid = valid_idx[np.argsort(-(scaled_scores + gumbels), kind="stable")]

        if invalid_idx.size == 0:
            return sampled_valid
        return np.concatenate([sampled_valid, invalid_idx])
