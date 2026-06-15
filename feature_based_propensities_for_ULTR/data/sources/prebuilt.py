"""Helpers for sources that resolve existing click artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from feature_based_propensities_for_ULTR.data.artifacts import PreparedClickArtifacts, normalize_prepared_split_paths


class PrebuiltClickSource:
    """Base helper for sources that expose already-prepared click artifacts."""

    kind = "prebuilt"

    def build_prepared_artifacts(
        self,
        *,
        source_name: str,
        artifact_dir: Path,
        manifest_path: Path,
        split_paths: Mapping[str, str | Path],
        max_positions: int,
        feature_dim: int,
        metadata: Mapping[str, Any] | None = None,
    ) -> PreparedClickArtifacts:
        return PreparedClickArtifacts(
            source_name=source_name,
            source_kind=self.kind,
            artifact_dir=artifact_dir,
            manifest_path=manifest_path,
            split_paths=normalize_prepared_split_paths(split_paths),
            max_positions=int(max_positions),
            feature_dim=int(feature_dim),
            metadata=metadata or {},
        )
