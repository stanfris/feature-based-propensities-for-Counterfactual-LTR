"""Runtime bundles built from prepared click artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from feature_based_propensities_for_ULTR.data.artifacts import PreparedClickArtifacts
from feature_based_propensities_for_ULTR.data.base import RatingDataset
from feature_based_propensities_for_ULTR.simulation import AggregatedClickDataset, ClickDataset


@dataclass(frozen=True)
class ClickDatasetBundle:
    train: ClickDataset
    val: ClickDataset
    test: ClickDataset | None


@dataclass(frozen=True)
class AggregatedDataBundle:
    train: AggregatedClickDataset
    val: AggregatedClickDataset
    test: AggregatedClickDataset
    cutoff: int
    n_sessions_used: int


@dataclass(frozen=True)
class DataBundle:
    aggregated: AggregatedDataBundle
    clicks: ClickDatasetBundle | None
    test_rating_dataset: RatingDataset | None = None


@dataclass(frozen=True)
class AggregatedDatasetPaths:
    train_path: Path
    val_path: Path
    test_path: Path
    save_dir: Path
    logging_policy_ckpt_dir: Path


@dataclass(frozen=True)
class LoadedClickArtifacts:
    """Loaded train/val/test click datasets plus their prepared-artifact metadata."""

    prepared_artifacts: PreparedClickArtifacts
    train: ClickDataset
    val: ClickDataset
    test: ClickDataset | None
    test_rating_dataset: RatingDataset

    def as_click_bundle(self) -> ClickDatasetBundle:
        return ClickDatasetBundle(
            train=self.train,
            val=self.val,
            test=self.test,
        )

    def as_legacy_tuple(self):
        return self.train, self.val, self.test, self.test_rating_dataset
