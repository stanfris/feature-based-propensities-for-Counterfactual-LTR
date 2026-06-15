"""Canonical click-artifact contracts and helpers."""

from .contract import (
    CANONICAL_CLICK_ARTIFACT_SCHEMA_VERSION,
    CANONICAL_CLICK_DATASET_KEYS,
    CANONICAL_CLICK_SPLITS,
    ClickPreparationSpec,
    PreparedClickArtifacts,
)
from .registry import (
    build_preparation_fingerprint,
    normalize_prepared_split_paths,
    normalize_source_name,
    prepared_artifact_dir,
    prepared_manifest_path,
    prepared_split_artifact_path,
)
from .validation import validate_prepared_click_artifacts

__all__ = [
    "CANONICAL_CLICK_ARTIFACT_SCHEMA_VERSION",
    "CANONICAL_CLICK_DATASET_KEYS",
    "CANONICAL_CLICK_SPLITS",
    "ClickPreparationSpec",
    "PreparedClickArtifacts",
    "build_preparation_fingerprint",
    "normalize_prepared_split_paths",
    "normalize_source_name",
    "prepared_artifact_dir",
    "prepared_manifest_path",
    "prepared_split_artifact_path",
    "validate_prepared_click_artifacts",
]
