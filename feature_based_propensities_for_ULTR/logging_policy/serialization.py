"""Checkpoint helpers for logging policy models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import orbax.checkpoint as ocp
from flax import nnx


def save_nnx_model(model: nnx.Module, ckpt_dir: str) -> Path:
    """Save a Flax NNX model state without RNGs."""
    _, _, model_state = nnx.split(model, nnx.RngState, ...)

    ckpt_path = Path(ckpt_dir).resolve()
    ckpt_path.mkdir(parents=True, exist_ok=True)

    ckptr = ocp.StandardCheckpointer()
    ckptr.save(ckpt_path, model_state, force=True)
    ckptr.wait_until_finished()
    return ckpt_path


def load_nnx_model(model: nnx.Module, ckpt_dir: str) -> Path:
    """Restore a Flax NNX model state into an existing model."""
    _, _, other_state = nnx.split(model, nnx.RngState, ...)

    ckptr = ocp.StandardCheckpointer()
    ckpt_path = Path(ckpt_dir).resolve()
    restored_other_state = ckptr.restore(ckpt_path, other_state)
    nnx.update(model, restored_other_state)
    return ckpt_path


def save_logging_policy_metadata(metadata: dict[str, Any], ckpt_dir: str) -> Path:
    ckpt_path = Path(ckpt_dir).resolve()
    ckpt_path.mkdir(parents=True, exist_ok=True)
    metadata_path = ckpt_path / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return metadata_path


def load_logging_policy_metadata(ckpt_dir: str) -> dict[str, Any] | None:
    metadata_path = Path(ckpt_dir).resolve() / "metadata.json"
    if not metadata_path.exists():
        return None
    return json.loads(metadata_path.read_text(encoding="utf-8"))
