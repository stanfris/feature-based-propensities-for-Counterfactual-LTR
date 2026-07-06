# Feature-based Propensities for Counterfactual Learning to Rank

This repository contains the code for the paper *Feature-based Propensities for Counterfactual Learning to Rank*. It is partially based on the implementation of *Understanding Two-Tower Models for Unbiased Learning to Rank* (Hager et al., 2025), available [here](https://github.com/philipphager/two-tower-confounding).

## Setup

The project supports Python 3.11 and 3.12. Install [`uv`](https://docs.astral.sh/uv/), then create and activate the environment:

```bash
uv python install 3.11
uv venv --python 3.11
uv sync
source .venv/bin/activate
```

All commands below assume that `.venv` is active.

## Datasets

The supported datasets are [MSLR-WEB30K/10K](https://www.microsoft.com/en-us/research/project/mslr/), [Yahoo! Webscope](https://webscope.sandbox.yahoo.com/catalog.php?datatype=c), and [Istella-S](https://istella.ai/datasets/letor-dataset/), which must be downloaded manually.

Set the dataset root with `LTR_DATASET_DIR`. It defaults to `../ltr_datasets`.

```bash
export LTR_DATASET_DIR=/path/to/ltr_datasets
```

Expected layout:

- `download/MSLR-WEB30K.zip` for `data=mslr30k`
- `download/MSLR-WEB10K.zip` for `data=mslr10k`
- `download/ltrc_yahoo.tar.bz2` for `data=yahoo`
- `dataset/istella-s-letor/sample/` for `data=istella`

## Running Experiments

Make the script executable, then run it:

```bash
chmod +x scripts/*.sh
./scripts/1_example.sh
```

`1_example.sh` runs IPS on MSLR-WEB30K with an MLP propensity estimator, 10,000 simulated sessions, logging-policy temperature `0.5`, and a ranking cutoff of 25. 

Optionally, you can launch the job on a SLURM cluster to distribute training jobs:

```bash
./scripts/1_example.sh +launcher=slurm
```
You can edit the launch parameters for SLURM under: `config/launcher/slurm.yaml`.

The main experiment collections can then be started with:

```bash
./scripts/compare_propensity_Euclidean.sh
./scripts/compare_propensity_MLP.sh
./scripts/run_policy_models.sh
./scripts/run_baselines.sh
./scripts/estimated_pos_bias.sh
```

These are large multiruns which may cost a significant amount of compute. Please inspect their dataset, seed, model, and session-count lists to align with your needs before launching them.



## Relevant parameters

The scripts pass [Hydra](https://hydra.cc/) overrides to `run.py`. The most useful overrides are:

| Parameter | Purpose | Common values |
| --- | --- | --- |
| `experiment` | Names the directory below `results/` | `1_example`, `real_targets` |
| `data` | Selects the ranking dataset | `mslr30k`, `yahoo`, `istella` |
| `ips.model` | Selects the learning objective or baseline | `ips`, `dm`, `dr`, `max-score`, `logging-policy` |
| `propensity_model` | Selects the propensity estimator | `frequency-based`, `MLPregression`, `kmeans`, `knn`, `cosine`, `euclidean`, `true_propensity` |
| `ips.n_sessions` | Sets the simulated train/validation session budget | for example `1000` or `100000` |
| `random_state` | Controls random sampling and model initialization | any integer, we use `40-59` |
| `policy_temperature` | Controls stochasticity of the logging policy | `0.0` is deterministic; larger values add randomness until `1.0` |
| `data.preprocessor.top_x` | Sets the top-k ranking cutoff | We use `25` |
| `ips.position_bias.source` | Chooses the examination-bias curve | `oracle`, `estimate` |

Defaults and all available settings are defined in `config/config.yaml`, `config/ips/default.yaml`, and the files below `config/propensity_model/`.


An experiment can also estimate its position-bias curve directly:

```bash
./scripts/1_example.sh \
  ips.position_bias.source=estimate \
  ips.position_bias.estimator=global_all_pairs
```

Supported estimators are `ctr`, `pivot_one`, `adjacent_chain`, and `global_all_pairs`. Position-bias estimation requires [ultr_bias_toolkit](https://github.com/philipphager/ultr-bias-toolkit), either installed as a package or checked out next to this repository at `../ultr-bias-toolkit`.


## Results
We publish all simulation results under `results/`. All code for our visualizations, and the visualisations which are already generated, are under `result_parsing/`. Below, we include a brief description of how to handle results.

Each multirun is written to:

```text
results/<experiment>/<Hydra override directory>/
```

You can also generate the visualizations, for example using:

```bash
python result_parsing/visualization_scripts/plot_real_targets.py
```

The PDFs are written to `result_parsing/result_plots/`.

