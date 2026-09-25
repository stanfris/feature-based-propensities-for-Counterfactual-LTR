"""Runtime aggregation cache helpers and orchestration."""

from __future__ import annotations

import logging
import time
import zipfile
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import DictConfig

from feature_based_propensities_for_ULTR.data.base import RatingDataset
from feature_based_propensities_for_ULTR.experiments.config_resolver import (
    IPSResolvedConfig,
    agg_signature,
    click_signature,
    prefixed_artifact_name,
    resolve_aggregation_session_budget,
    split_sessions,
)
from feature_based_propensities_for_ULTR.prebuilt import is_prebuilt_click_mode
from feature_based_propensities_for_ULTR.simulation.aggregation import _prepare_aggregation_inputs
from feature_based_propensities_for_ULTR.simulation import AggregatedClickDataset, Simulator

from .alignment import resolve_click_bundle_cutoff
from .bundles import AggregatedDataBundle, AggregatedDatasetPaths, ClickDatasetBundle

logger = logging.getLogger(__name__)


def build_aggregated_dataset_paths(
    config: DictConfig,
    save_subdir: str,
    *,
    n_sessions: int | None,
) -> AggregatedDatasetPaths:
    dataset_dir = Path(config.dataset_dir).expanduser()
    save_dir = dataset_dir / save_subdir
    save_dir.mkdir(parents=True, exist_ok=True)

    train_signature = agg_signature(config, "train", n_sessions)
    val_signature = agg_signature(config, "val", n_sessions)
    test_signature = agg_signature(config, "test", None)

    train_agg_path = save_dir / prefixed_artifact_name(
        config,
        f"train_aggregated_dataset_{train_signature}.npz",
    )
    val_agg_path = save_dir / prefixed_artifact_name(
        config,
        f"val_aggregated_dataset_{val_signature}.npz",
    )
    test_agg_path = save_dir / prefixed_artifact_name(
        config,
        f"test_aggregated_dataset_{test_signature}.npz",
    )
    logging_policy_ckpt_dir = save_dir / prefixed_artifact_name(
        config,
        f"logging_policy_{click_signature(config, 'train')}",
    )

    return AggregatedDatasetPaths(
        train_path=train_agg_path,
        val_path=val_agg_path,
        test_path=test_agg_path,
        save_dir=save_dir,
        logging_policy_ckpt_dir=logging_policy_ckpt_dir,
    )


def load_aggregated_datasets(paths: AggregatedDatasetPaths) -> AggregatedDataBundle | None:
    """Load cached aggregated datasets from disk."""

    logger.info("Loading aggregated datasets from %s...", paths.save_dir)
    try:
        train_data = AggregatedClickDataset.load_from_npz(str(paths.train_path))
        val_data = AggregatedClickDataset.load_from_npz(str(paths.val_path))
        test_data = AggregatedClickDataset.load_from_npz(str(paths.test_path))
    except (zipfile.BadZipFile, OSError, Exception) as exc:
        logger.warning(
            "Failed to load cached aggregated datasets (%s: %s). "
            "Deleting corrupt files and regenerating...",
            type(exc).__name__,
            exc,
        )
        for path in (paths.train_path, paths.val_path, paths.test_path):
            try:
                if path.exists():
                    path.unlink()
                    logger.info("Deleted corrupt file: %s", path)
            except OSError:
                pass
        return None

    n_sessions_used = int(len(train_data.sessions) + len(val_data.sessions))
    cutoff = int(train_data.cutoff)
    if int(val_data.cutoff) != cutoff or int(test_data.cutoff) != cutoff:
        raise ValueError("Loaded aggregated datasets have mismatched cutoff values.")

    return AggregatedDataBundle(
        train=train_data,
        val=val_data,
        test=test_data,
        cutoff=cutoff,
        n_sessions_used=n_sessions_used,
    )


def aggregate_datasets(
    *,
    config: DictConfig,
    resolved: IPSResolvedConfig,
    paths: AggregatedDatasetPaths,
    click_bundle: ClickDatasetBundle,
    test_rating_dataset: RatingDataset,
    n_sessions: int | None,
) -> AggregatedDataBundle:
    if click_bundle is None:
        raise RuntimeError("Click datasets are required to build aggregated datasets.")
    persist_datasets = bool(getattr(config, "persist_datasets", True))

    requested_n_sessions = n_sessions
    if requested_n_sessions is None:
        requested_n_sessions = resolve_aggregation_session_budget(
            config,
            len(click_bundle.train),
            len(click_bundle.val),
        )

    if requested_n_sessions is None:
        n_train = len(click_bundle.train)
        n_val = len(click_bundle.val)
        n_sessions_used = n_train + n_val
    else:
        n_sessions_used, n_train, n_val = split_sessions(
            requested_n_sessions,
            len(click_bundle.train),
            len(click_bundle.val),
        )

    cutoff = resolve_click_bundle_cutoff(config, click_bundle)

    rng = jax.random.PRNGKey(resolved.seed)
    rng_train, rng_val, rng_test = jax.random.split(rng, 3)

    hist_dir = resolved.display_histogram_dir
    if hist_dir is not None:
        hist_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Initializing simulator...")
    max_label = int(getattr(getattr(config, "simulation", {}), "max_label", 4))
    simulator = Simulator(
        logging_policy_ranker=lambda **_: None,
        logging_policy_sampler=lambda **_: None,
        bias_strength=resolved.bias_strength,
        random_state=resolved.seed,
        max_label=max_label,
    )
    use_query_doc_id_mapping = is_prebuilt_click_mode(config)
    if use_query_doc_id_mapping:
        logger.info(
            "Prebuilt mode: enabling shared query-document aggregation via (query, query_doc_id) keys."
        )

    start = time.time()
    logger.info("Aggregating train dataset...")
    train_data = simulator.aggregate(
        click_dataset=click_bundle.train,
        n_sessions=n_train,
        cutoff=cutoff,
        top_x=cutoff,
        use_query_doc_id_mapping=use_query_doc_id_mapping,
        display_histogram_max=resolved.display_histogram_max,
        display_histogram_path=str(hist_dir / "display_histogram_train.json")
        if hist_dir is not None
        else None,
        rng_key=rng_train,
        debug=resolved.debug,
    )
    logger.info("Aggregating validation dataset...")
    val_data = simulator.aggregate(
        click_dataset=click_bundle.val,
        n_sessions=n_val,
        cutoff=cutoff,
        top_x=cutoff,
        use_query_doc_id_mapping=use_query_doc_id_mapping,
        display_histogram_max=resolved.display_histogram_max,
        display_histogram_path=str(hist_dir / "display_histogram_val.json")
        if hist_dir is not None
        else None,
        rng_key=rng_val,
        debug=resolved.debug,
    )
    if click_bundle.test is None:
        logger.info("Aggregating test dataset from labels only...")
        test_data = _aggregate_label_only_test_dataset(
            rating_dataset=test_rating_dataset,
            cutoff=cutoff,
            use_query_doc_id_mapping=use_query_doc_id_mapping,
        )
    else:
        logger.info("Aggregating test dataset...")
        test_data = simulator.aggregate(
            click_dataset=click_bundle.test,
            n_sessions=None,
            cutoff=cutoff,
            top_x=cutoff,
            use_query_doc_id_mapping=use_query_doc_id_mapping,
            display_histogram_max=resolved.display_histogram_max,
            display_histogram_path=str(hist_dir / "display_histogram_test.json")
            if hist_dir is not None
            else None,
            rng_key=rng_test,
            debug=resolved.debug,
        )
    logger.info("Aggregation done in %.2f seconds", time.time() - start)

    actual_train_sessions = int(len(train_data.sessions))
    actual_val_sessions = int(len(val_data.sessions))
    actual_test_sessions = int(len(test_data.sessions))
    train_clicks_used = int(len(click_bundle.train))
    val_clicks_used = int(len(click_bundle.val))
    test_clicks_used = 0 if click_bundle.test is None else int(len(click_bundle.test))
    logger.info(
        "Session usage after aggregation: requested(train=%d,val=%d), actual(train=%d,val=%d,test=%d)",
        int(n_train),
        int(n_val),
        actual_train_sessions,
        actual_val_sessions,
        actual_test_sessions,
    )

    if persist_datasets:
        train_data.save_to_npz(
            str(paths.train_path),
            extra_metadata={
                "split": "train",
                "clicks": train_clicks_used,
                "n_sessions": int(requested_n_sessions) if requested_n_sessions is not None else "all",
                "n_sessions_used": actual_train_sessions,
            },
        )
        val_data.save_to_npz(
            str(paths.val_path),
            extra_metadata={
                "split": "val",
                "clicks": val_clicks_used,
                "n_sessions": int(requested_n_sessions) if requested_n_sessions is not None else "all",
                "n_sessions_used": actual_val_sessions,
            },
        )
        test_data.save_to_npz(
            str(paths.test_path),
            extra_metadata={
                "split": "test",
                "clicks": test_clicks_used,
                "n_sessions": "all",
                "n_sessions_used": actual_test_sessions,
            },
        )
        logger.info("Saved aggregated datasets to %s", paths.save_dir)
    else:
        logger.info("Dataset persistence disabled; skipping aggregated dataset cache save.")

    return AggregatedDataBundle(
        train=train_data,
        val=val_data,
        test=test_data,
        cutoff=cutoff,
        n_sessions_used=actual_train_sessions + actual_val_sessions,
    )


def check_aggregated_dataset(data, cutoff: int, name: str) -> None:
    clicks = np.asarray(data.clicks)
    displays = np.asarray(data.displays)
    if clicks.shape != displays.shape:
        raise ValueError(f"{name}: clicks/displays shape mismatch {clicks.shape} vs {displays.shape}")
    if clicks.shape[1] != cutoff:
        raise ValueError(f"{name}: cutoff mismatch {clicks.shape[1]} vs {cutoff}")
    if data.feature_matrix.shape[0] != data.num_docs():
        raise ValueError(f"{name}: feature_matrix/doc count mismatch")
    if data.doclist_ranges[-1] != data.num_docs():
        raise ValueError(f"{name}: doclist_ranges mismatch")
    if np.any(displays < clicks):
        raise ValueError(f"{name}: displays must be >= clicks per doc/rank")


def _aggregate_label_only_test_dataset(
    *,
    rating_dataset: RatingDataset,
    cutoff: int,
    use_query_doc_id_mapping: bool,
) -> AggregatedClickDataset:
    base = _prepare_aggregation_inputs(
        rating_dataset,
        use_query_doc_id_mapping=use_query_doc_id_mapping,
    )
    total_docs = int(base.doclist_ranges[-1])
    n_queries = int(base.doclist_ranges.shape[0] - 1)
    cutoff = int(cutoff)
    zero_matrix = jnp.zeros((total_docs, cutoff), dtype=jnp.int32)
    empty_sessions = jnp.zeros((0,), dtype=jnp.int32)
    return AggregatedClickDataset(
        query=base.query_out,
        feature_matrix=base.feature_matrix,
        lp_feature_matrix=base.lp_feature_matrix,
        label_vector=base.label_vector,
        doc_id_vector=base.doc_id_vector,
        doclist_ranges=base.doclist_ranges,
        doc_id_map=base.doc_id_map,
        query_index_per_doc=base.query_index_per_doc,
        clicks=zero_matrix,
        displays=zero_matrix,
        clicks_per_doc=jnp.zeros((total_docs,), dtype=jnp.int32),
        displays_per_doc=jnp.zeros((total_docs,), dtype=jnp.int32),
        query_freq=jnp.zeros((n_queries,), dtype=jnp.int32),
        sessions=empty_sessions,
        cutoff=cutoff,
        metadata={
            "use_query_doc_id_mapping": bool(use_query_doc_id_mapping),
            "has_reconstructible_query_tensors": True,
            "has_separate_lp_feature_matrix": bool(base.lp_feature_matrix is not None),
            "test_set_mode": "label_only",
        },
    )


def validate_aggregated_bundle(bundle: AggregatedDataBundle) -> None:
    check_aggregated_dataset(bundle.train, bundle.cutoff, "train")
    check_aggregated_dataset(bundle.val, bundle.cutoff, "validation")
    check_aggregated_dataset(bundle.test, bundle.cutoff, "test")


__all__ = [
    "aggregate_datasets",
    "build_aggregated_dataset_paths",
    "check_aggregated_dataset",
    "load_aggregated_datasets",
    "validate_aggregated_bundle",
]
