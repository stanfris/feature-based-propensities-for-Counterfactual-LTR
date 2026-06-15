"""Shared helpers for prebuilt click-dataset mode."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from omegaconf import DictConfig


@dataclass(frozen=True)
class PrebuiltClickPaths:
    train: Path
    val: Path
    test: Path
    test_clicks: Path | None = None


def resolve_prebuilt_cutoff(config: DictConfig) -> int | None:
    data_cfg = getattr(config, "data", None)
    raw_max_documents = getattr(data_cfg, "max_documents_per_query", None) if data_cfg is not None else None
    if raw_max_documents is not None:
        cutoff = int(raw_max_documents)
        if cutoff <= 0:
            raise ValueError(f"config.data.max_documents_per_query must be > 0, got {cutoff}.")
        return cutoff

    preprocessor_cfg = getattr(data_cfg, "preprocessor", None) if data_cfg is not None else None
    raw_preprocessor_max_documents = (
        getattr(preprocessor_cfg, "max_documents_per_query", None)
        if preprocessor_cfg is not None
        else None
    )
    if raw_preprocessor_max_documents is not None:
        cutoff = int(raw_preprocessor_max_documents)
        if cutoff <= 0:
            raise ValueError(
                f"config.data.preprocessor.max_documents_per_query must be > 0, got {cutoff}."
            )
        return cutoff

    raw_cutoff = getattr(data_cfg, "data_cutoff", None) if data_cfg is not None else None
    if raw_cutoff is None:
        return None
    cutoff = int(raw_cutoff)
    if cutoff <= 0:
        raise ValueError(f"config.data.data_cutoff must be > 0, got {cutoff}.")
    return cutoff


def add_cutoff_suffix(path: Path, cutoff: int | None) -> Path:
    if cutoff is None:
        return path
    suffix = f"_k{int(cutoff)}"
    if path.stem.endswith(suffix):
        return path
    return path.with_name(f"{path.stem}{suffix}{path.suffix}")


def prefer_existing_cutoff_variant(path: Path, cutoff: int | None) -> Path:
    """Prefer an existing suffixed artifact, then fall back to the base artifact."""
    suffixed = add_cutoff_suffix(path, cutoff)
    if cutoff is None or suffixed == path:
        return path
    if suffixed.exists():
        return suffixed
    if path.exists():
        return path
    return suffixed


def prebuilt_split_artifact_path(output_dir: Path, split: str, cutoff: int | None) -> Path:
    return add_cutoff_suffix(output_dir / f"{split}_click_dataset.npz", cutoff)


def prebuilt_manifest_path(output_dir: Path, cutoff: int | None) -> Path:
    return add_cutoff_suffix(output_dir / "manifest.json", cutoff)


def is_prebuilt_click_mode(config: DictConfig) -> bool:
    data_cfg = getattr(config, "data", None)
    mode = getattr(data_cfg, "click_data_mode", None) if data_cfg is not None else None
    mode_norm = str(mode).strip().lower().replace("_", "-") if mode is not None else ""
    return mode_norm == "prebuilt"


def resolve_prebuilt_click_path(
    config: DictConfig,
    raw_path: str | Path | None,
    *,
    split: str,
    must_exist: bool = False,
) -> Path:
    if raw_path is None:
        raise ValueError(
            f"Missing config.data.prebuilt_click_paths.{split} for prebuilt click mode."
        )

    cutoff = resolve_prebuilt_cutoff(config)
    path = Path(str(raw_path)).expanduser()
    if not path.is_absolute():
        path = Path(config.dataset_dir).expanduser() / path
    path = path.resolve()
    path = prefer_existing_cutoff_variant(path, cutoff)

    if must_exist and not path.exists():
        raise FileNotFoundError(
            f"Prebuilt click dataset for split '{split}' does not exist: {path}"
        )
    return path


def resolve_prebuilt_click_paths(
    config: DictConfig,
    *,
    must_exist: bool = False,
) -> PrebuiltClickPaths:
    data_cfg = getattr(config, "data", None)
    prebuilt_cfg = getattr(data_cfg, "prebuilt_click_paths", None) if data_cfg is not None else None
    if prebuilt_cfg is None:
        raise ValueError(
            "click_data_mode='prebuilt' requires config.data.prebuilt_click_paths "
            "with train/val/test NPZ paths."
        )

    return PrebuiltClickPaths(
        train=resolve_prebuilt_click_path(
            config,
            getattr(prebuilt_cfg, "train", None),
            split="train",
            must_exist=must_exist,
        ),
        val=resolve_prebuilt_click_path(
            config,
            getattr(prebuilt_cfg, "val", None),
            split="val",
            must_exist=must_exist,
        ),
        test=resolve_prebuilt_click_path(
            config,
            getattr(prebuilt_cfg, "test", None),
            split="test",
            must_exist=must_exist,
        ),
        test_clicks=resolve_prebuilt_click_path(
            config,
            getattr(prebuilt_cfg, "test_clicks", None),
            split="test_clicks",
            must_exist=must_exist,
        ) if getattr(prebuilt_cfg, "test_clicks", None) is not None else None,
    )


__all__ = [
    "PrebuiltClickPaths",
    "add_cutoff_suffix",
    "is_prebuilt_click_mode",
    "prefer_existing_cutoff_variant",
    "prebuilt_manifest_path",
    "prebuilt_split_artifact_path",
    "resolve_prebuilt_cutoff",
    "resolve_prebuilt_click_path",
    "resolve_prebuilt_click_paths",
]
