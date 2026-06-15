"""Validation helpers for prepared click artifacts."""

from __future__ import annotations

from .contract import CANONICAL_CLICK_ARTIFACT_SCHEMA_VERSION, PreparedClickArtifacts


def validate_prepared_click_artifacts(prepared: PreparedClickArtifacts) -> None:
    """Validate the prepared-artifact descriptor before runtime loading."""

    if prepared.schema_version != CANONICAL_CLICK_ARTIFACT_SCHEMA_VERSION:
        raise ValueError(
            "Unsupported prepared click-artifact schema version: "
            f"{prepared.schema_version}."
        )

    if not prepared.manifest_path.exists():
        raise FileNotFoundError(f"Prepared click-artifact manifest not found: {prepared.manifest_path}")

    for split, path in prepared.split_paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Prepared click artifact for split '{split}' not found: {path}")
