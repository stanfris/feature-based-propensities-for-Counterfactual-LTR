# Architecture Overview

This codebase is organized so that the core responsibilities are easy to locate:

- Logging policy models and samplers live under `two_tower_confounding/logging_policy/`.
- Click simulation and aggregation live under `two_tower_confounding/simulation/`.
- Dataset wrappers for simulated clicks are in `two_tower_confounding/simulation/datasets.py`.
- Serialization/checkpoint helpers are in `two_tower_confounding/logging_policy/serialization.py`
  and `two_tower_confounding/simulation/serialization.py`.
- IPS/DM/DR experiment orchestration lives under `two_tower_confounding/experiments/`.
- Model training, metrics, and data loaders remain under `two_tower_confounding/models/`,
  `two_tower_confounding/metrics.py`, and `two_tower_confounding/data/`.

## Module Map

```
.
├── run.py
├── compare_propensity_param.py
├── two_tower_confounding/
│   ├── experiments/
│   │   ├── __init__.py
│   │   ├── ips_pipeline.py      # IPS/DM/DR orchestration
│   │   ├── config_resolver.py   # Config normalization + signatures
│   │   ├── data_pipeline.py     # Click/aggregated dataset loading + caching
│   │   ├── propensity_pipeline.py # Propensity estimation + oracle utilities
│   │   ├── regression_pipeline.py # DM/DR regression training
│   │   ├── model_pipeline.py    # IPS model training/eval
│   │   ├── diagnostics.py       # Stats/debug output helpers
│   │   ├── reporting.py         # Output dict + JSON writing
│   │   └── logging_policy.py    # Logging-policy helpers for analysis scripts
│   ├── logging_policy/
│   │   ├── __init__.py
│   │   ├── ranker.py            # Logging policy models + training loops
│   │   ├── samplers.py          # Score-to-ranking samplers
│   │   └── serialization.py     # Checkpoint helpers for logging policies
│   ├── simulation/
│   │   ├── __init__.py
│   │   ├── click_simulator.py   # Click simulation and Simulator orchestration
│   │   ├── aggregation.py       # Aggregation logic for document-level datasets
│   │   ├── datasets.py          # ClickDataset / AggregatedClickDataset wrappers
│   │   ├── serialization.py     # NPZ helpers for simulated datasets
│   │   └── simulator.py         # Legacy re-exports (compatibility)
│   ├── models/
│   │   ├── policy_models.py     # Policy estimators and selection helper
│   ├── data/
│   │   ├── parsers/
│   │   │   └── contracts.py       # NPZ schema/constants/validation contracts
│   │   └── ...
│   ├── ranking/
│   ├── trainer.py
│   ├── prebuilt.py                # Shared prebuilt-mode helpers
│   └── utils.py
```

## Entry Points (Canonical Flows)

This project is research-oriented and keeps multiple scripts for reproducibility. The
canonical entry points are:

1. `run.py`
   - Canonical IPS/DM/DR workflow. Delegates orchestration to `experiments/ips_pipeline.py`.
2. `compare_propensity_param.py`
   - Analysis script for propensity estimators and sensitivity checks.

The `two_tower_confounding/simulation/simulator.py` module
continues to re-export legacy symbols (e.g., `NeuralRanker`, `Simulator`) to preserve
existing Hydra configs and user imports.

## Dataset Selection And Artifact Names

- The Hydra data group supports `data=istella` via
  `two_tower_confounding.data.datasets.istella.IstellaSample`.
- Istella is loaded directly from the extracted directory
  `dataset/istella-s-letor/sample/{train.txt,vali.txt,test.txt}` (no archive download required).
- Istella drops feature columns `0` and `193` (1-based feature ids `1` and `194`, both with extreme sentinel
  values in the sample split) before normalization; the pipeline then validates
  `expected_feature_dim: 218` at runtime before click simulation.
- Cached artifacts are dataset-scoped by prefix when `data.artifact_prefix` is set.
  For Istella this produces names such as:
  `istella_logging_policy_...`, `istella_train_click_dataset_...`,
  and `istella_train_aggregated_dataset_...`.
- MLSR configs do not set a prefix, so existing MLSR artifact names remain unchanged.
