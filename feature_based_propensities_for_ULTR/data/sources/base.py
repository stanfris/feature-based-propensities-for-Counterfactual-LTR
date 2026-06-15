"""Base protocols and config helpers for click-data sources."""

from __future__ import annotations

from typing import Protocol

from omegaconf import DictConfig

from feature_based_propensities_for_ULTR.data.artifacts import ClickPreparationSpec, PreparedClickArtifacts, normalize_source_name


class ClickSource(Protocol):
    """Protocol implemented by all click-data sources."""

    name: str
    kind: str

    def prepare(self, spec: ClickPreparationSpec) -> PreparedClickArtifacts:
        ...


def infer_click_source_name(config: DictConfig) -> str:
    data_cfg = getattr(config, "data", None)
    artifact_prefix = getattr(data_cfg, "artifact_prefix", None) if data_cfg is not None else None
    if artifact_prefix:
        return normalize_source_name(str(artifact_prefix))

    dataset_cfg = getattr(data_cfg, "dataset", None) if data_cfg is not None else None
    target = getattr(dataset_cfg, "_target_", None) if dataset_cfg is not None else None
    if target:
        return normalize_source_name(str(target).split(".")[-1])

    mode = getattr(data_cfg, "click_data_mode", None) if data_cfg is not None else None
    if mode:
        return normalize_source_name(str(mode))

    return "dataset"
