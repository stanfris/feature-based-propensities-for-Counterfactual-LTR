"""Source abstractions for preparing canonical click artifacts."""

from __future__ import annotations

from omegaconf import DictConfig

from feature_based_propensities_for_ULTR.prebuilt import is_prebuilt_click_mode

from .base import infer_click_source_name
from .generated import load_or_generate_generated_click_datasets


def prepare_click_artifacts_from_config(config: DictConfig):
    """Prepare or resolve canonical click artifacts for supported sources.

    Returns ``None`` when the config uses the legacy generated-click flow. This
    keeps the first migration slice backwards compatible while routing prebuilt
    sources through the new abstraction boundary.
    """

    if not is_prebuilt_click_mode(config):
        return None

    source_name = infer_click_source_name(config)
    raise ValueError(
        "Unsupported prebuilt click source "
        f"'{source_name}'. Implement a dedicated source in feature_based_propensities_for_ULTR.data.sources."
    )


__all__ = [
    "infer_click_source_name",
    "load_or_generate_generated_click_datasets",
    "prepare_click_artifacts_from_config",
]
