"""Compatibility layer for legacy simulation imports.

Use modules under feature_based_propensities_for_ULTR.simulation and
 feature_based_propensities_for_ULTR.logging_policy for the new structure.
"""

from feature_based_propensities_for_ULTR.logging_policy.ranker import NeuralRanker
from feature_based_propensities_for_ULTR.logging_policy.samplers import EGreedySampler, PlackettLuceSampler

from .aggregation import (
    _aggregate_click_dataset,
    _build_doc_index_mapping,
    _infer_positions_are_ranks_jax,
)
from .click_simulator import (
    Simulator,
    _parse_ratio_list,
    _sample_sessions_from_query_counts,
    get_position_bias,
    get_relevance,
)
from .datasets import AggregatedClickDataset, ClickDataset

__all__ = [
    "Simulator",
    "get_position_bias",
    "get_relevance",
    "ClickDataset",
    "AggregatedClickDataset",
    "NeuralRanker",
    "EGreedySampler",
    "PlackettLuceSampler",
    "_aggregate_click_dataset",
    "_build_doc_index_mapping",
    "_infer_positions_are_ranks_jax",
    "_parse_ratio_list",
    "_sample_sessions_from_query_counts",
]
