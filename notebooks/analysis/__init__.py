from .config import ExperimentConfig
from .data_loader import (
    aggregate_mean_std,
    load_baselines_from_folder,
    load_long_metrics,
    metric_only_baselines,
)
from .stats import make_combined_latex_table

_PLOTTING_EXPORTS = {
    "plot_dm_dr_ips_naiveho_frequency_based",
    "plot_grid",
    "plot_policy_models_train_histograms_ps1p0",
    "plot_temperature_analysis",
}

__all__ = [
    "ExperimentConfig",
    "load_long_metrics",
    "aggregate_mean_std",
    "metric_only_baselines",
    "load_baselines_from_folder",
    "plot_grid",
    "plot_dm_dr_ips_naiveho_frequency_based",
    "plot_policy_models_train_histograms_ps1p0",
    "plot_temperature_analysis",
    "make_combined_latex_table",
]


def __getattr__(name: str):
    if name in _PLOTTING_EXPORTS:
        from . import plotting as _plotting

        return getattr(_plotting, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
