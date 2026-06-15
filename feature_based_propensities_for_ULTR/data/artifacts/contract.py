"""Canonical contracts for prepared click artifacts."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

CANONICAL_CLICK_ARTIFACT_SCHEMA_VERSION = "click_artifact.v1"
CANONICAL_CLICK_SPLITS = ("train", "val", "test")
CANONICAL_CLICK_DATASET_KEYS = frozenset(
    {
        "sessions",
        "clicks",
        "positions",
        "sessions_per_query",
        "sessions_per_doc_pos",
        "sessions_per_doc_pos_diag",
        "has_diagonal_only_sessions_per_doc_pos",
        "positions_are_dense_identity",
        "positions_are_one_based",
        "query",
        "query_doc_features",
        "lp_query_doc_features",
        "query_doc_ids",
        "labels",
        "mask",
        "n",
    }
)


def _normalize_mapping(mapping: Mapping[str, Any] | None) -> dict[str, Any]:
    if mapping is None:
        return {}
    return dict(mapping)


@dataclass(frozen=True)
class ClickPreparationSpec:
    """Source-agnostic specification for preparing click artifacts."""

    source_name: str
    artifact_root: Path
    overwrite: bool = False
    cutoff: int | None = None
    source_config: Mapping[str, Any] = field(default_factory=dict)
    generation_config: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        artifact_root = Path(self.artifact_root).expanduser().resolve()
        object.__setattr__(self, "artifact_root", artifact_root)
        object.__setattr__(self, "source_config", _normalize_mapping(self.source_config))
        object.__setattr__(self, "generation_config", _normalize_mapping(self.generation_config))
        if self.cutoff is not None and int(self.cutoff) <= 0:
            raise ValueError(f"cutoff must be > 0, got {self.cutoff}.")


@dataclass(frozen=True)
class PreparedClickArtifacts:
    """Canonical description of prepared train/val/test click artifacts."""

    source_name: str
    source_kind: str
    artifact_dir: Path
    manifest_path: Path
    split_paths: Mapping[str, Path]
    schema_version: str = CANONICAL_CLICK_ARTIFACT_SCHEMA_VERSION
    max_positions: int = 0
    feature_dim: int = 0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        artifact_dir = Path(self.artifact_dir).expanduser().resolve()
        manifest_path = Path(self.manifest_path).expanduser().resolve()
        split_paths = {
            str(split): Path(path).expanduser().resolve()
            for split, path in dict(self.split_paths).items()
        }

        missing = [split for split in CANONICAL_CLICK_SPLITS if split not in split_paths]
        if missing:
            raise ValueError(f"Prepared click artifacts missing split paths for: {missing}")

        if int(self.max_positions) <= 0:
            raise ValueError(f"max_positions must be > 0, got {self.max_positions}.")
        if int(self.feature_dim) <= 0:
            raise ValueError(f"feature_dim must be > 0, got {self.feature_dim}.")

        object.__setattr__(self, "artifact_dir", artifact_dir)
        object.__setattr__(self, "manifest_path", manifest_path)
        object.__setattr__(self, "split_paths", split_paths)
        object.__setattr__(self, "metadata", _normalize_mapping(self.metadata))
