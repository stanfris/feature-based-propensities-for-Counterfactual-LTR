"""Shared helpers to align click-dataset width with the effective runtime cutoff."""

from __future__ import annotations

import logging

import numpy as np
from omegaconf import DictConfig

from feature_based_propensities_for_ULTR.data.base import RatingDataset
from feature_based_propensities_for_ULTR.experiments.config_resolver import resolve_data_cutoff
from feature_based_propensities_for_ULTR.simulation import ClickDataset

from .bundles import ClickDatasetBundle

log = logging.getLogger(__name__)


def resolve_click_bundle_cutoff(
    config: DictConfig,
    click_bundle: ClickDatasetBundle,
) -> int:
    natural_cutoff = int(
        min(click_bundle.train.positions.shape[1], click_bundle.val.positions.shape[1])
    )
    configured_cutoff = resolve_data_cutoff(config)
    if configured_cutoff is None:
        return natural_cutoff

    cutoff = min(natural_cutoff, int(configured_cutoff))
    log.info(
        "Applying data cutoff override: configured=%d, natural=%d, effective=%d",
        int(configured_cutoff),
        natural_cutoff,
        cutoff,
    )
    return cutoff


def _slice_rating_dataset(dataset: RatingDataset, cutoff: int) -> RatingDataset:
    cutoff = int(cutoff)
    lp_features = (
        np.asarray(dataset.lp_query_doc_features[:, :cutoff, :])
        if dataset.has_separate_lp_query_doc_features
        else None
    )
    return RatingDataset(
        query=np.asarray(dataset.query),
        query_doc_ids=np.asarray(dataset.query_doc_ids[:, :cutoff]),
        query_doc_features=np.asarray(dataset.query_doc_features[:, :cutoff, :]),
        lp_query_doc_features=lp_features,
        labels=np.asarray(dataset.labels[:, :cutoff]),
        mask=np.asarray(dataset.mask[:, :cutoff]),
        n=np.minimum(np.asarray(dataset.n), cutoff),
    )


def _slice_sessions_per_doc_pos(click_dataset: ClickDataset, cutoff: int) -> np.ndarray:
    cutoff = int(cutoff)
    if click_dataset.has_diagonal_only_sessions_per_doc_pos:
        return np.asarray(click_dataset.sessions_per_doc_pos_diag[:, :cutoff])
    return np.asarray(click_dataset.sessions_per_doc_pos[:, :cutoff, :cutoff])


def slice_click_dataset_to_cutoff(click_dataset: ClickDataset, cutoff: int) -> ClickDataset:
    cutoff = int(cutoff)
    current_width = int(click_dataset.positions.shape[1])
    if cutoff <= 0:
        raise ValueError(f"cutoff must be > 0, got {cutoff}.")
    if cutoff > current_width:
        raise ValueError(
            f"Cannot expand click dataset width from {current_width} to {cutoff}."
        )
    if cutoff == current_width:
        return click_dataset

    return ClickDataset(
        # Keep the original rating tensors intact because click_dataset.positions
        # stores raw candidate indices, not remapped 0..cutoff-1 indices.
        rating_dataset=click_dataset,
        sessions=np.asarray(click_dataset.sessions),
        clicks=np.asarray(click_dataset.clicks[:, :cutoff]),
        positions=np.asarray(click_dataset.positions[:, :cutoff]),
        sessions_per_query=np.asarray(click_dataset.sessions_per_query),
        sessions_per_doc_pos=_slice_sessions_per_doc_pos(click_dataset, cutoff),
    )


def align_click_bundle_to_cutoff(
    click_bundle: ClickDatasetBundle,
    cutoff: int,
) -> ClickDatasetBundle:
    cutoff = int(cutoff)
    return ClickDatasetBundle(
        train=slice_click_dataset_to_cutoff(click_bundle.train, cutoff),
        val=slice_click_dataset_to_cutoff(click_bundle.val, cutoff),
        test=(
            None
            if click_bundle.test is None
            else slice_click_dataset_to_cutoff(click_bundle.test, cutoff)
        ),
    )


__all__ = [
    "align_click_bundle_to_cutoff",
    "resolve_click_bundle_cutoff",
    "slice_click_dataset_to_cutoff",
]
