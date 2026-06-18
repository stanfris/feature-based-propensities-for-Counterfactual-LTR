# feature-based-propensities-for-ULTR

This repository contains research code for doubly robust learning-to-rank experiments. The main workflow is a Hydra-driven experiment pipeline that loads ranking datasets, simulates or reuses click data, trains IPS/DM/DR-style models, and writes experiment outputs to the repository. The canonical setup in this repository is now `uv`-first and targets CPU execution by default.

## Python version

- Recommended: Python `3.11`
- Supported by project metadata: `>=3.11,<3.13`

## Install `uv`

Official installer:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Alternative on macOS with Homebrew:

```bash
brew install uv
```

## Canonical setup

From a fresh clone, run:

```bash
uv python install 3.11
uv venv --python 3.11
uv sync
```

This creates `.venv/` and installs the project in editable mode, so local modules such as `two_tower_confounding` resolve correctly.

## Optional dependency groups

The base environment is intentionally focused on the core experiment pipeline.

- `dev`: test runner
- `analysis`: plotting support for `compare_propensity_param.py`
- `slurm`: Hydra Submitit launcher plugin for `+launcher=slurm` and `+launcher=slurmcpu`
- `tracking`: optional Weights & Biases integration

Examples:

```bash
uv sync --group dev
uv sync --group analysis
uv sync --group slurm
uv sync --group tracking
```

## Dataset location

Set the dataset root with `LTR_DATASET_DIR`. If you do not set it, the code defaults to `./data`.

```bash
export LTR_DATASET_DIR=/absolute/path/to/ltr_datasets
```

Expected layout under `LTR_DATASET_DIR`:

- `download/MSLR-WEB30K.zip` for `data=mslr30k`
- `download/MSLR-WEB10K.zip` for `data=mslr10k`
- `download/ltrc_yahoo.tar.bz2` for `data=yahoo`
- `dataset/istella-s-letor/sample/` for `data=istella`

The code also creates and reuses:

- `dataset/` for extracted raw datasets
- `cache/` for parsed SVMLight caches
- experiment-generated click artifacts under the configured dataset root when `persist_datasets=true`

## Running core experiments

The main experiment entry point is:

```bash
uv run two-tower-run
```

Useful smoke-test style example with smaller budgets:

```bash
export LTR_DATASET_DIR=/absolute/path/to/ltr_datasets
uv run two-tower-run \
  data=mslr30k \
  random_state=42 \
  train_clicks=5000 \
  val_clicks=2500 \
  test_clicks=1000 \
  ips.n_sessions=1000 \
  use_wandb=false
```

Equivalent direct script form:

```bash
uv run python run.py data=mslr30k ips.n_sessions=1000 use_wandb=false
```

If you want to use the cluster launcher configs such as `+launcher=slurmcpu` or `+launcher=slurm`, install the SLURM group first:

```bash
uv sync --group slurm
uv run python run.py -m ... +launcher=slurmcpu
```

Without that plugin, Hydra only exposes the built-in `basic` launcher and will fail with `Could not find 'hydra/launcher/submitit_slurm'`.

Position-bias export entry point:

```bash
uv run two-tower-estimate-position-bias data=mslr30k ips.n_sessions=1000
```

IPS/DM/DR runs can also estimate a position-bias curve directly in each run:

```bash
uv run two-tower-run \
  data=mslr30k \
  ips.position_bias.source=estimate \
  ips.position_bias.estimator=pivot_one
```

When `source=estimate`, the run writes the selected curve to `position_bias.json`
next to `ips_results.json`. Supported estimators are `ctr`, `pivot_one`,
`adjacent_chain`, and `global_all_pairs`.

Important: `estimate_position_bias.py` depends on `ultr_bias_toolkit`, which is not vendored in this repository. The current code looks for either:

- an installed `ultr_bias_toolkit` package, or
- a sibling checkout at `../ultr-bias-toolkit`

That step is therefore not part of the minimal base environment.

## Analysis-only script

`compare_propensity_param.py` is not part of the base environment because it requires plotting dependencies. Install the analysis group first:

```bash
uv sync --group analysis
uv run two-tower-compare-propensity data=mslr30k
```

## Testing

Install the dev group and run the unit tests:

```bash
uv sync --group dev
uv run --group dev pytest
```

The repository tests are unit-style tests and do not require external ranking datasets.

## Compatibility fallback

`requirements.txt` is kept only as a compatibility shim for tools that insist on a pip-style requirements file:

```bash
uv pip install -r requirements.txt
```

The canonical workflow remains `uv sync`.

## Non-obvious prerequisites

- CPU execution is the default and the documented path here.
- Weights & Biases is disabled by default in config. If you want it, install the `tracking` group and authenticate separately.
- The project writes experiment outputs into the repository tree, especially under `results/`, checkpoints, and display histogram directories.
- `ultr_bias_toolkit` is required for position-bias estimation and is not part of the base environment.

## Reproducibility notes

- `uv.lock` is the version-pinned record for the environment resolved from `pyproject.toml`.
- The Python package metadata is the single source of truth for dependencies; `requirements.txt` delegates to it via editable install.
- Datasets are not included in this repository, so reproducibility depends on staging the same raw archives under `LTR_DATASET_DIR`.
- The default config no longer hard-codes a machine-specific dataset path.
- The default config no longer requires W&B for a local run.
