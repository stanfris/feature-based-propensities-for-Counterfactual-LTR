"""Generated click-data preparation for rating-dataset sources."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil

import numpy as np
from hydra.utils import instantiate
from omegaconf import DictConfig

from feature_based_propensities_for_ULTR.data.feature_validation import validate_expected_feature_dim
from feature_based_propensities_for_ULTR.data.runtime import (
    load_click_dataset_npz,
    load_rating_dataset_npz,
    save_rating_dataset_npz,
)
from feature_based_propensities_for_ULTR.experiments.config_resolver import (
    click_signature,
    prefixed_artifact_name,
    resolve_test_set_mode,
    session_document_percentage_override,
    session_click_override,
    split_session_budget,
    split_sessions,
    test_evaluation_signature,
)
from feature_based_propensities_for_ULTR.simulation import Simulator


@dataclass(frozen=True)
class ClickDatasetPaths:
    train_click_path: Path
    val_click_path: Path
    test_click_path: Path | None
    test_rating_path: Path
    save_dir: Path
    logging_policy_ckpt_dir: Path

def _delete_corrupt_checkpoint(ckpt_dir: Path) -> None:
    """Best-effort deletion of a corrupt logging-policy checkpoint."""
    try:
        if not ckpt_dir.exists():
            return
        if ckpt_dir.is_dir():
            shutil.rmtree(ckpt_dir)
        else:
            ckpt_dir.unlink()
        print(f"Deleted corrupt logging-policy checkpoint: {ckpt_dir}")
    except OSError as exc:
        print(
            "Could not delete corrupt logging-policy checkpoint "
            f"{ckpt_dir} ({type(exc).__name__}: {exc})"
        )


def build_click_dataset_paths(
    config: DictConfig,
    save_subdir: str,
    *,
    logging_policy_ckpt_dir: str | None = None,
) -> ClickDatasetPaths:
    dataset_dir = Path(config.dataset_dir).expanduser()
    save_dir = dataset_dir / save_subdir
    save_dir.mkdir(parents=True, exist_ok=True)

    test_set_mode = resolve_test_set_mode(config)
    train_click_sig = click_signature(config, "train")
    val_click_sig = click_signature(config, "val")
    test_artifact_sig = (
        click_signature(config, "test")
        if test_set_mode == "clicks"
        else test_evaluation_signature(config)
    )

    train_click_path = save_dir / prefixed_artifact_name(
        config,
        f"train_click_dataset_{train_click_sig}.npz",
    )
    val_click_path = save_dir / prefixed_artifact_name(
        config,
        f"val_click_dataset_{val_click_sig}.npz",
    )
    test_click_path = None
    if test_set_mode == "clicks":
        test_click_path = save_dir / prefixed_artifact_name(
            config,
            f"test_click_dataset_{test_artifact_sig}.npz",
        )
    test_rating_path = save_dir / prefixed_artifact_name(
        config,
        f"test_dataset_{test_artifact_sig}.npz",
    )
    if logging_policy_ckpt_dir is None:
        logging_policy_path = save_dir / prefixed_artifact_name(
            config,
            f"logging_policy_{train_click_sig}",
        )
    else:
        logging_policy_path = Path(logging_policy_ckpt_dir).expanduser()

    return ClickDatasetPaths(
        train_click_path=train_click_path,
        val_click_path=val_click_path,
        test_click_path=test_click_path,
        test_rating_path=test_rating_path,
        save_dir=save_dir,
        logging_policy_ckpt_dir=logging_policy_path,
    )


def train_val_test_datasets(
    config: DictConfig,
    varying: bool = False,
    *,
    logging_policy_ckpt_dir: str | None = None,
):
    """Simulate train/val/test click datasets from a rating dataset source."""

    del varying  # Reserved for compatibility with the legacy public API.
    test_set_mode = resolve_test_set_mode(config)

    dataset = instantiate(config.data.dataset)
    preprocessor = instantiate(config.data.preprocessor)

    train_dataset = preprocessor(dataset.load("train"), split="train")
    val_dataset = preprocessor(dataset.load("val"), split="val")
    test_dataset = preprocessor(dataset.load("test"), split="test")
    validate_expected_feature_dim(
        config,
        train_dataset.n_logging_policy_features,
        source="train_dataset.lp_query_doc_features",
    )

    logging_policy_ranker = instantiate(config.logging_policy_ranker)
    ckpt_dir = Path(logging_policy_ckpt_dir).expanduser() if logging_policy_ckpt_dir else Path("checkpoint_lp")
    if ckpt_dir.exists():
        print(f"Loading logging policy from checkpoint: {ckpt_dir}")
        try:
            logging_policy_ranker.load_logging_policy(
                ckpt_dir=str(ckpt_dir),
                n_logging_policy_features=train_dataset.n_logging_policy_features,
            )
        except Exception as exc:
            print(
                "Failed to load logging policy checkpoint "
                f"({type(exc).__name__}: {exc}). Deleting corrupt checkpoint and re-training."
            )
            _delete_corrupt_checkpoint(ckpt_dir)
            logging_policy_ranker.fit(train_dataset)
            logging_policy_ranker.save_logging_policy(ckpt_dir=str(ckpt_dir))
    else:
        print("Training logging policy...")
        logging_policy_ranker.fit(train_dataset)
        logging_policy_ranker.save_logging_policy(ckpt_dir=str(ckpt_dir))
    logging_policy_sampler = instantiate(config.logging_policy_sampler)

    simulator = Simulator(
        logging_policy_ranker=logging_policy_ranker,
        logging_policy_sampler=logging_policy_sampler,
        bias_strength=config.bias_strength,
        random_state=config.random_state,
    )

    query_sampling_ratios = getattr(config, "query_sampling_ratios", None)
    single_obs_mode = bool(
        getattr(getattr(config.data, "preprocessor", {}), "disjoint_query_chunk_mode", False)
    )
    force_single_sample = bool(getattr(getattr(config, "ips", {}), "force_single_sample", False))
    effective_single_obs_mode = single_obs_mode or force_single_sample
    top_x = getattr(preprocessor, "top_x", None)
    debug_enabled = bool(getattr(getattr(config, "ips", {}), "debug", False))
    debug_max_samples = int(getattr(getattr(config, "ips", {}), "debug_max_samples", 3))
    train_clicks = int(config.train_clicks)
    val_clicks = int(config.val_clicks)
    session_percentage_override = session_document_percentage_override(config)
    if session_percentage_override is not None:
        if not effective_single_obs_mode:
            raise ValueError(
                "ips.n_session_percentage requires single-sample query selection. "
                "Enable disjoint_query_chunk_mode or ips.force_single_sample."
            )
        total_queries = len(train_dataset) + len(val_dataset)
        total_target_queries = int(np.ceil(total_queries * (session_percentage_override / 100.0)))
        total_target_queries = min(total_target_queries, total_queries)
        _, train_clicks, val_clicks = split_sessions(
            total_target_queries,
            len(train_dataset),
            len(val_dataset),
        )
        print(
            "Using ips.n_session_percentage as the train/val pseudo-query override: "
            f"train_clicks={train_clicks}, val_clicks={val_clicks} "
            f"(requested percentage={session_percentage_override}%)"
        )
    n_sessions_override = session_click_override(config)
    if n_sessions_override is not None:
        train_clicks, val_clicks = split_session_budget(
            n_sessions_override,
            len(train_dataset),
            len(val_dataset),
        )
        print(
            "Using ips.n_sessions as the train/val simulated-session budget: "
            f"train_clicks={train_clicks}, val_clicks={val_clicks} "
            f"(requested n_sessions={n_sessions_override})"
        )

    train_click_dataset = simulator(
        train_dataset,
        train_clicks,
        top_x=top_x,
        query_sampling_ratios=query_sampling_ratios,
        single_observation_per_query_doc=effective_single_obs_mode,
        debug=debug_enabled,
        debug_label="train",
        debug_max_samples=debug_max_samples,
    )
    val_click_dataset = simulator(
        val_dataset,
        val_clicks,
        top_x=top_x,
        query_sampling_ratios=query_sampling_ratios,
        single_observation_per_query_doc=effective_single_obs_mode,
        debug=debug_enabled,
        debug_label="val",
        debug_max_samples=debug_max_samples,
    )
    test_click_dataset = None
    if test_set_mode == "clicks":
        test_click_dataset = simulator(
            test_dataset,
            config.test_clicks,
            top_x=top_x,
            query_sampling_ratios=query_sampling_ratios,
            debug=debug_enabled,
            debug_label="test",
            debug_max_samples=debug_max_samples,
        )

    return train_click_dataset, val_click_dataset, test_click_dataset, test_dataset


def load_or_generate_generated_click_datasets(
    config: DictConfig,
    save_subdir: str,
    *,
    varying: bool = False,
    logging_policy_ckpt_dir: str | None = None,
):
    persist_datasets = bool(getattr(config, "persist_datasets", True))
    test_set_mode = resolve_test_set_mode(config)
    needs_test_click_dataset = test_set_mode == "clicks"
    paths = build_click_dataset_paths(
        config,
        save_subdir,
        logging_policy_ckpt_dir=logging_policy_ckpt_dir,
    )
    train_click_path = paths.train_click_path
    val_click_path = paths.val_click_path
    test_click_path = paths.test_click_path
    test_rating_path = paths.test_rating_path
    save_dir = paths.save_dir
    logging_policy_ckpt_dir = str(paths.logging_policy_ckpt_dir)

    all_saved = (
        train_click_path.exists()
        and val_click_path.exists()
        and test_rating_path.exists()
        and (
            not needs_test_click_dataset
            or (test_click_path is not None and test_click_path.exists())
        )
    )

    if persist_datasets and all_saved:
        import zipfile

        try:
            _, train_click_dataset = load_click_dataset_npz(train_click_path)
            _, val_click_dataset = load_click_dataset_npz(val_click_path)
            test_click_dataset = None
            if needs_test_click_dataset:
                if test_click_path is None:
                    raise RuntimeError("Missing test click artifact path in legacy click mode.")
                _, test_click_dataset = load_click_dataset_npz(test_click_path)
            test_dataset = load_rating_dataset_npz(test_rating_path)
            validate_expected_feature_dim(
                config,
                int(np.asarray(train_click_dataset.lp_query_doc_features).shape[-1]),
                source="cached_train_click_dataset.lp_query_doc_features",
            )
            print(f"✅ Loaded saved datasets from {save_dir}")
            return train_click_dataset, val_click_dataset, test_click_dataset, test_dataset
        except (zipfile.BadZipFile, OSError, Exception) as exc:
            print(
                f"⚠️  Failed to load cached click datasets ({type(exc).__name__}: {exc}). "
                "Deleting corrupt files and regenerating..."
            )
            delete_paths = [train_click_path, val_click_path, test_rating_path]
            if needs_test_click_dataset and test_click_path is not None:
                delete_paths.append(test_click_path)
            for path in delete_paths:
                try:
                    if path.exists():
                        path.unlink()
                        print(f"   Deleted corrupt file: {path}")
                except OSError:
                    pass

    train_click_dataset, val_click_dataset, test_click_dataset, test_dataset = train_val_test_datasets(
        config,
        varying=varying,
        logging_policy_ckpt_dir=logging_policy_ckpt_dir,
    )
    validate_expected_feature_dim(
        config,
        int(np.asarray(train_click_dataset.lp_query_doc_features).shape[-1]),
        source="generated_train_click_dataset.lp_query_doc_features",
    )
    if persist_datasets:
        train_click_dataset.save_to_npz(str(train_click_path))
        val_click_dataset.save_to_npz(str(val_click_path))
        if test_click_dataset is not None and test_click_path is not None:
            test_click_dataset.save_to_npz(str(test_click_path))
        save_rating_dataset_npz(test_dataset, test_rating_path)
        print(f"✅ Saved datasets to {save_dir}")
    else:
        print("Dataset persistence disabled; skipping click dataset cache save/load.")
    return train_click_dataset, val_click_dataset, test_click_dataset, test_dataset


__all__ = [
    "ClickDatasetPaths",
    "build_click_dataset_paths",
    "load_or_generate_generated_click_datasets",
    "train_val_test_datasets",
]
