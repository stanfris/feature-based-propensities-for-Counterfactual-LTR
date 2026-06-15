import time
from pathlib import Path

import logging

logger = logging.getLogger(__name__)
from omegaconf import DictConfig

from feature_based_propensities_for_ULTR.data.runtime import (
    AggregatedDataBundle,
    AggregatedDatasetPaths,
    ClickDatasetBundle,
    DataBundle,
    aggregate_datasets,
    align_click_bundle_to_cutoff,
    build_aggregated_dataset_paths,
    check_aggregated_dataset,
    load_aggregated_datasets,
    resolve_click_bundle_cutoff,
    validate_aggregated_bundle,
)
from feature_based_propensities_for_ULTR.utils import load_or_generate_click_datasets

from .config_resolver import (
    IPSResolvedConfig,
    aggregation_session_override,
)


def load_click_datasets(
    config: DictConfig,
    ips_cfg: DictConfig,
    logging_policy_ckpt_dir: Path,
    resolved: "IPSResolvedConfig | None" = None,
) -> tuple[ClickDatasetBundle, object]:
    start = time.time()
    logger.info("Loading or generating click datasets...")

    train_click_dataset, val_click_dataset, test_click_dataset, test_rating_dataset = load_or_generate_click_datasets(
        config=config,
        save_subdir=ips_cfg.save_subdir,
        varying=False,
        logging_policy_ckpt_dir=str(logging_policy_ckpt_dir),
    )
    test_count = len(test_click_dataset) if test_click_dataset is not None else 0
    logger.info(
        "Loaded click datasets in %.2f seconds (train=%d, val=%d, test=%d)"
        % (time.time() - start, len(train_click_dataset), len(val_click_dataset), test_count)
    )
    if resolved is not None and resolved.test_set_mode == "label_only":
        logger.info("Test split is label-only by design; no test click dataset was loaded.")
    return (
        ClickDatasetBundle(
            train=train_click_dataset,
            val=val_click_dataset,
            test=test_click_dataset,
        ),
        test_rating_dataset,
    )


def load_aligned_click_datasets(
    config: DictConfig,
    ips_cfg: DictConfig,
    logging_policy_ckpt_dir: Path,
    resolved: "IPSResolvedConfig | None" = None,
) -> tuple[ClickDatasetBundle, object]:
    click_bundle, test_rating_dataset = load_click_datasets(
        config,
        ips_cfg,
        logging_policy_ckpt_dir,
        resolved=resolved,
    )
    effective_cutoff = resolve_click_bundle_cutoff(config, click_bundle)
    aligned_click_bundle = align_click_bundle_to_cutoff(click_bundle, effective_cutoff)
    return aligned_click_bundle, test_rating_dataset


def prepare_data_bundle(
    config: DictConfig,
    resolved: IPSResolvedConfig,
    *,
    force_click_datasets: bool = False,
) -> DataBundle:
    ips_cfg = config.ips
    persist_datasets = bool(getattr(config, "persist_datasets", True))
    paths = build_aggregated_dataset_paths(
        config,
        ips_cfg.save_subdir,
        n_sessions=aggregation_session_override(config),
    )

    agg_saved = (
        persist_datasets
        and paths.train_path.exists()
        and paths.val_path.exists()
        and paths.test_path.exists()
    )
    need_click_datasets = force_click_datasets or (
        (not agg_saved)
        or resolved.use_mlp_propensity
    )

    click_bundle = None
    test_rating_dataset = None
    if need_click_datasets:
        click_bundle, test_rating_dataset = load_aligned_click_datasets(
            config,
            ips_cfg,
            paths.logging_policy_ckpt_dir,
            resolved=resolved,
        )

    aggregated = None
    if agg_saved:
        aggregated = load_aggregated_datasets(paths)
        # None is returned when the cached files were corrupt and deleted.
        if aggregated is None and click_bundle is None:
            # We need click datasets to regenerate — load them now.
            click_bundle, test_rating_dataset = load_aligned_click_datasets(
                config,
                ips_cfg,
                paths.logging_policy_ckpt_dir,
                resolved=resolved,
            )

    if aggregated is not None and click_bundle is not None and int(aggregated.cutoff) != int(click_bundle.train.positions.shape[1]):
        raise ValueError(
            "Cached aggregated cutoff does not match the aligned click-dataset width: "
            f"aggregated={aggregated.cutoff}, click_width={click_bundle.train.positions.shape[1]}."
        )

    if aggregated is None:
        if click_bundle is None or test_rating_dataset is None:
            raise RuntimeError("Prepared train/val clicks and the test rating dataset are required to build aggregated datasets.")
        aggregated = aggregate_datasets(
            config=config,
            resolved=resolved,
            paths=paths,
            click_bundle=click_bundle,
            test_rating_dataset=test_rating_dataset,
            n_sessions=aggregation_session_override(config),
        )

    return DataBundle(
        aggregated=aggregated,
        clicks=click_bundle,
        test_rating_dataset=test_rating_dataset,
    )


__all__ = [
    "AggregatedDataBundle",
    "AggregatedDatasetPaths",
    "ClickDatasetBundle",
    "DataBundle",
    "aggregate_datasets",
    "build_aggregated_dataset_paths",
    "check_aggregated_dataset",
    "load_aggregated_datasets",
    "load_aligned_click_datasets",
    "load_click_datasets",
    "prepare_data_bundle",
    "validate_aggregated_bundle",
]
