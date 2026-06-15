"""Simulation package: click simulation, aggregation, and dataset wrappers."""

from .click_simulator import Simulator, get_position_bias, get_relevance
from .datasets import AggregatedClickDataset, ClickDataset

__all__ = [
    "Simulator",
    "get_position_bias",
    "get_relevance",
    "ClickDataset",
    "AggregatedClickDataset",
]
