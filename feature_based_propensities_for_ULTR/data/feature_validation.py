"""Shared feature-dimension validation helpers."""

from __future__ import annotations

from pathlib import Path

from omegaconf import DictConfig

from feature_based_propensities_for_ULTR.experiments.config_resolver import resolve_expected_feature_dim


def validate_expected_feature_dim(
    config: DictConfig,
    observed_feature_dim: int,
    *,
    source: str,
) -> None:
    expected_feature_dim = resolve_expected_feature_dim(config)
    if expected_feature_dim is None:
        return

    observed = int(observed_feature_dim)
    if observed == expected_feature_dim:
        return

    dataset_root = Path(config.dataset_dir).expanduser() / "dataset"
    raise ValueError(
        "Feature-dimension mismatch for configured dataset. "
        f"expected_feature_dim={expected_feature_dim}, observed={observed}, source={source}. "
        f"Check dataset files under '{dataset_root}' and the split parser."
    )


__all__ = ["validate_expected_feature_dim"]
