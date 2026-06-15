"""Runtime helpers for consuming prepared click artifacts."""

from .alignment import (
    align_click_bundle_to_cutoff,
    resolve_click_bundle_cutoff,
    slice_click_dataset_to_cutoff,
)
from .aggregation import (
    aggregate_datasets,
    build_aggregated_dataset_paths,
    check_aggregated_dataset,
    load_aggregated_datasets,
    validate_aggregated_bundle,
)
from .bundles import (
    AggregatedDataBundle,
    AggregatedDatasetPaths,
    ClickDatasetBundle,
    DataBundle,
    LoadedClickArtifacts,
)
from .io import (
    load_click_dataset_npz,
    load_rating_dataset_npz,
    rating_dataset_from_click_dataset,
    save_rating_dataset_npz,
)
from .loader import load_click_bundle_from_prepared_artifacts

__all__ = [
    "AggregatedDataBundle",
    "AggregatedDatasetPaths",
    "ClickDatasetBundle",
    "DataBundle",
    "LoadedClickArtifacts",
    "align_click_bundle_to_cutoff",
    "aggregate_datasets",
    "build_aggregated_dataset_paths",
    "check_aggregated_dataset",
    "load_aggregated_datasets",
    "load_click_bundle_from_prepared_artifacts",
    "resolve_click_bundle_cutoff",
    "load_click_dataset_npz",
    "load_rating_dataset_npz",
    "rating_dataset_from_click_dataset",
    "save_rating_dataset_npz",
    "slice_click_dataset_to_cutoff",
    "validate_aggregated_bundle",
]
