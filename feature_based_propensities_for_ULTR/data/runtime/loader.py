"""Runtime loading for prepared click artifacts."""

from __future__ import annotations

import logging

import numpy as np

from feature_based_propensities_for_ULTR.data.artifacts import CANONICAL_CLICK_SPLITS, PreparedClickArtifacts, validate_prepared_click_artifacts

from .bundles import LoadedClickArtifacts

log = logging.getLogger(__name__)


def load_click_bundle_from_prepared_artifacts(
    prepared: PreparedClickArtifacts,
    *,
    include_test_click_dataset: bool = False,
) -> LoadedClickArtifacts:
    """Load canonical click artifacts into runtime datasets.

    Note: prebuilt dataset sources (baidu-ultr, cwrczech) have been removed.
    This function is retained for future prebuilt source implementations.
    """
    raise NotImplementedError(
        "No prebuilt dataset loaders are available. "
        "The baidu-ultr and cwrczech prebuilt sources have been removed from the codebase. "
        "To use prebuilt click artifacts, implement a new source in "
        "feature_based_propensities_for_ULTR.data.sources."
    )
