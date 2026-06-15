"""Path and fingerprint helpers for canonical click artifacts."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping

from .contract import CANONICAL_CLICK_SPLITS, ClickPreparationSpec


def normalize_source_name(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9]+", "_", str(value).strip().lower()).strip("_")
    return normalized or "dataset"


def normalize_prepared_split_paths(paths: Mapping[str, str | Path]) -> dict[str, Path]:
    normalized = {
        str(split): Path(path).expanduser().resolve()
        for split, path in dict(paths).items()
    }
    missing = [split for split in CANONICAL_CLICK_SPLITS if split not in normalized]
    if missing:
        raise ValueError(f"Missing split paths for {missing}.")
    return normalized


def _stable_json_default(value: Any) -> str:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Unsupported value for fingerprinting: {type(value).__name__}")


def _stable_json_dumps(payload: Mapping[str, Any]) -> str:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=_stable_json_default,
    )


def build_preparation_fingerprint(spec: ClickPreparationSpec, *, digest_len: int = 12) -> str:
    payload = {
        "source_name": normalize_source_name(spec.source_name),
        "cutoff": spec.cutoff,
        "source_config": dict(spec.source_config),
        "generation_config": dict(spec.generation_config or {}),
    }
    return hashlib.md5(_stable_json_dumps(payload).encode("utf-8")).hexdigest()[:digest_len]


def prepared_artifact_dir(artifact_root: Path, source_name: str, fingerprint: str) -> Path:
    root = Path(artifact_root).expanduser().resolve()
    return root / normalize_source_name(source_name) / "click_artifacts" / f"prep_{fingerprint}"


def prepared_manifest_path(artifact_dir: Path) -> Path:
    return Path(artifact_dir).expanduser().resolve() / "manifest.json"


def prepared_split_artifact_path(artifact_dir: Path, split: str) -> Path:
    split_name = str(split).strip().lower()
    if split_name not in CANONICAL_CLICK_SPLITS:
        raise ValueError(f"Unknown click-artifact split: {split}")
    return Path(artifact_dir).expanduser().resolve() / f"{split_name}_click_dataset.npz"
