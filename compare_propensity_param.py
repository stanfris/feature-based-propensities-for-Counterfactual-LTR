import random

import hydra
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf

from feature_based_propensities_for_ULTR.data.runtime import (
    ClickDatasetBundle,
    align_click_bundle_to_cutoff,
    resolve_click_bundle_cutoff,
)
from feature_based_propensities_for_ULTR.experiments.logging_policy import (
    build_dominant_position_lookup,
    compute_dominant_pos_per_doc,
    load_logging_policy_ranker,
)
from feature_based_propensities_for_ULTR.experiments.config_resolver import effective_click_count
from feature_based_propensities_for_ULTR.experiments.propensity_pipeline import _subset_click_dataset
from feature_based_propensities_for_ULTR.models.propensity import (
    build_propensity_model_spec,
    prepare_propensity_model,
    FrequencyPropensityModel,
    PropensityPredictor,
)
from feature_based_propensities_for_ULTR.prebuilt import is_prebuilt_click_mode
from feature_based_propensities_for_ULTR.simulation import Simulator, get_position_bias
from feature_based_propensities_for_ULTR.utils import load_or_generate_click_datasets, set_device


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def aggregate_by_observation_count(
    dominant_pos: np.ndarray,
    values_per_doc: np.ndarray,
    displays_per_doc: np.ndarray,
    cutoff: int,
    obs_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Aggregate per-doc values by dominant position for a specific observation count."""
    mask = displays_per_doc == obs_count
    dom = dominant_pos[mask]
    vals = values_per_doc[mask]

    valid = (dom >= 0) & (dom < cutoff) & np.isfinite(vals)
    dom = dom[valid]
    vals = vals[valid]

    sum_per_pos = np.bincount(dom, weights=vals, minlength=cutoff)
    sumsq_per_pos = np.bincount(dom, weights=vals ** 2, minlength=cutoff)
    count_per_pos = np.bincount(dom, minlength=cutoff)

    mean = sum_per_pos / np.maximum(count_per_pos, 1)
    var = (sumsq_per_pos / np.maximum(count_per_pos, 1)) - (mean ** 2)
    var = np.maximum(var, 0.0)
    std = np.sqrt(var)
    return mean, std


def plot_multi_panel(
    csv_path: str,
    obs_counts: list[int],
    output_path: str,
) -> None:
    df = pd.read_csv(csv_path)
    positions = df["position"].to_numpy()
    effective_alpha = df["effective_alpha"].to_numpy()

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    axes = axes.ravel()

    for idx, obs_count in enumerate(obs_counts):
        ax = axes[idx]
        mean_pred = df[f"pred_mean_obs{obs_count}"].to_numpy()
        std_pred = df[f"pred_std_obs{obs_count}"].to_numpy()
        mean_expected = df[f"exp_mean_obs{obs_count}"].to_numpy()
        std_expected = df[f"exp_std_obs{obs_count}"].to_numpy()
        valid = np.isfinite(mean_pred)
        pos_valid = positions[valid]

        ax.plot(
            positions,
            effective_alpha,
            label="Effective alpha (logit)",
            color="black",
            linestyle="--",
            linewidth=2.0,
            zorder=3,
        )
        ax.plot(
            pos_valid,
            mean_pred[valid],
            label="Predicted propensity (mean, logit)",
            marker="x",
            color="tab:blue",
            alpha=0.75,
            zorder=2,
        )
        ax.fill_between(
            pos_valid,
            mean_pred[valid] - std_pred[valid],
            mean_pred[valid] + std_pred[valid],
            alpha=0.2,
            color="tab:blue",
            label="Predicted propensity ±1 std (logit)",
        )
        ax.plot(
            pos_valid,
            mean_expected[valid],
            label="Expected alpha (propensity_per_doc, logit)",
            marker="s",
            markersize=6,
            color="tab:orange",
            alpha=0.75,
            zorder=2,
        )
        ax.fill_between(
            pos_valid,
            mean_expected[valid] - std_expected[valid],
            mean_expected[valid] + std_expected[valid],
            alpha=0.2,
            color="tab:orange",
            label="Expected alpha ±1 std (logit)",
        )
        ax.set_title(f"Obs = {obs_count}")
        ax.grid(True, alpha=0.3)

    for idx in range(len(obs_counts), len(axes)):
        axes[idx].set_visible(False)

    axes[0].set_ylabel("Logit")
    axes[2].set_ylabel("Logit")
    axes[2].set_xlabel("Position")
    axes[3].set_xlabel("Position")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2)
    if obs_counts:
        obs_label = f"{obs_counts[0]}–{obs_counts[-1]}"
    else:
        obs_label = "N/A"
    fig.suptitle(f"Propensity by Observation Count ({obs_label})")
    fig.tight_layout(rect=[0, 0.07, 1, 0.92])

def click_signature(config: DictConfig) -> str:
    def _abbr_value(value) -> str:
        return str(value).replace(".", "p").replace(" ", "")

    max_label = int(getattr(getattr(config, "simulation", {}), "max_label", 4))
    return (
        f"tc{_abbr_value(effective_click_count(config, 'train'))}_"
        f"te{_abbr_value(config.test_clicks)}_"
        f"ps{_abbr_value(config.policy_strength)}_"
        f"pt{_abbr_value(config.policy_temperature)}_"
        f"bs{_abbr_value(config.bias_strength)}_"
        f"ml{_abbr_value(max_label)}"
    )


def propensity_model_signature(config: DictConfig) -> str:
    def _abbr_value(value) -> str:
        return str(value).replace(".", "p").replace(" ", "")

    model_type = str(config.propensity_model.type).lower()
    if model_type == "propensity_mlp_classifier":
        classifier_cfg = config.propensity_model.classifier
        return (
            f"clf_l{_abbr_value(classifier_cfg.layers)}_"
            f"h{_abbr_value(classifier_cfg.hidden_units)}_"
            f"d{_abbr_value(classifier_cfg.dropout)}"
        )
    if model_type == "kmeans":
        kmeans_cfg = config.propensity_model.kmeans
        return (
            f"kmeans_g{_abbr_value(kmeans_cfg.num_groups)}_"
            f"min{_abbr_value(kmeans_cfg.min_cluster_size)}_"
            f"it{_abbr_value(kmeans_cfg.kmeans_iters)}"
        )
    if model_type == "cosine":
        cosine_cfg = config.propensity_model.cosine
        return f"cos_t{_abbr_value(cosine_cfg.cosine_threshold)}_k{_abbr_value(cosine_cfg.knn_k)}"
    if model_type == "euclidean":
        euclidean_cfg = config.propensity_model.euclidean
        return (
            f"euc_t{_abbr_value(euclidean_cfg.euclidean_threshold)}_"
            f"k{_abbr_value(euclidean_cfg.knn_k)}"
        )
    if model_type == "knn":
        knn_cfg = config.propensity_model.knn
        return f"knn_k{_abbr_value(knn_cfg.knn_k)}"
    if model_type == "propensity_mlp_regression":
        regression_cfg = config.propensity_model.regression
        return (
            f"reg_l{_abbr_value(regression_cfg.layers)}_"
            f"h{_abbr_value(regression_cfg.hidden_units)}_"
            f"d{_abbr_value(regression_cfg.dropout)}"
        )
    return f"unknown_{_abbr_value(model_type)}"


def build_stats_frame(
    *,
    positions: np.ndarray,
    effective_alpha: np.ndarray,
    per_obs_stats: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    cutoff: int,
    obs_counts: list[int],
) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "position": positions,
            "effective_alpha": effective_alpha,
        }
    )

    for obs_count in obs_counts:
        mean_pred, std_pred, mean_expected, std_expected = per_obs_stats[obs_count]
        pred_full = np.full_like(positions, np.nan, dtype=np.float64)
        pred_std_full = np.full_like(positions, np.nan, dtype=np.float64)
        exp_full = np.full_like(positions, np.nan, dtype=np.float64)
        exp_std_full = np.full_like(positions, np.nan, dtype=np.float64)

        pred_full[:cutoff] = mean_pred
        pred_std_full[:cutoff] = std_pred
        exp_full[:cutoff] = mean_expected
        exp_std_full[:cutoff] = std_expected

        df[f"pred_mean_obs{obs_count}"] = pred_full
        df[f"pred_std_obs{obs_count}"] = pred_std_full
        df[f"exp_mean_obs{obs_count}"] = exp_full
        df[f"exp_std_obs{obs_count}"] = exp_std_full

    return df


@hydra.main(version_base="1.3", config_path="config/", config_name="config")
def main(config: DictConfig):
    config.test_set_mode = "clicks"
    set_device(config.device)
    print(OmegaConf.to_yaml(config))
    seed_all(int(config.random_state))

    train_click_dataset, val_click_dataset, test_click_dataset, test_dataset = (
        load_or_generate_click_datasets(
            config,
            save_subdir="lp_predictor_datasets",
            varying=False,
        )
    )
    click_bundle = ClickDatasetBundle(
        train=train_click_dataset,
        val=val_click_dataset,
        test=test_click_dataset,
    )
    cutoff = resolve_click_bundle_cutoff(config, click_bundle)
    click_bundle = align_click_bundle_to_cutoff(click_bundle, cutoff)
    train_click_dataset = click_bundle.train
    val_click_dataset = click_bundle.val
    test_click_dataset = click_bundle.test

    logging_policy_ranker = load_logging_policy_ranker(config, test_dataset)
    lookup = build_dominant_position_lookup(logging_policy_ranker, test_dataset)

    if test_click_dataset is None:
        raise RuntimeError("compare_propensity_param.py requires test_set_mode=clicks.")

    alpha_logits = get_position_bias(cutoff, strength=float(config.bias_strength))
    mean_alpha_logit = float(np.mean(alpha_logits))
    policy_temperature = float(config.policy_temperature)
    effective_alpha_logits = (
        (1 - policy_temperature) * alpha_logits
        + policy_temperature * mean_alpha_logit
    )

    bias_value_array = jnp.asarray(alpha_logits - alpha_logits[0])

    spec = build_propensity_model_spec(
        config=config,
        train_dataset=train_click_dataset,
        bias_value_array=bias_value_array,
        seed=int(config.random_state),
    )

    rng = jax.random.PRNGKey(int(config.random_state))
    simulator = Simulator(
        logging_policy_ranker=lambda **_: None,
        logging_policy_sampler=lambda **_: None,
        bias_strength=float(config.bias_strength),
        random_state=int(config.random_state),
        max_label=int(getattr(getattr(config, "simulation", {}), "max_label", 4)),
    )
    test_data = simulator.aggregate(
        click_dataset=test_click_dataset,
        n_sessions=None,
        cutoff=cutoff,
        top_x=cutoff,
        use_query_doc_id_mapping=is_prebuilt_click_mode(config),
        rng_key=rng,
    )

    dominant_pos = compute_dominant_pos_per_doc(test_data, lookup)
    displays_per_doc = np.asarray(test_data.displays_per_doc)

    expected_predictor = PropensityPredictor(
        model=FrequencyPropensityModel(),
        alpha=alpha_logits,
        apply_sigmoid=False,
    )
    expected_propensity_per_doc = expected_predictor.predict_per_doc(test_data)

    base_sig = click_signature(config)
    model_sig = propensity_model_signature(config)

    obs_groups = [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
    ]
    obs_counts = list(range(1, 13))

    propensity_cfg = getattr(getattr(config, "ips", {}), "propensity", None) or {}
    training_cfg = getattr(config.propensity_model, "training", None) or {}
    train_click_dataset_for_fit = train_click_dataset
    val_click_dataset_for_fit = val_click_dataset
    if spec.trainable:
        max_train_samples = getattr(propensity_cfg, "max_train_samples", None)
        train_click_dataset_for_fit = _subset_click_dataset(
            train_click_dataset,
            max_train_samples,
            int(config.random_state),
        )
        val_click_dataset_for_fit = _subset_click_dataset(
            val_click_dataset,
            max_train_samples,
            int(config.random_state) + 1,
        )
    dataloader_cfg = getattr(propensity_cfg, "dataloader", None) or {}

    propensity_model = prepare_propensity_model(
        spec=spec,
        train_click_dataset=train_click_dataset_for_fit,
        val_click_dataset=val_click_dataset_for_fit,
        batch_size=int(getattr(training_cfg, "batch_size", 512)),
        learning_rate=float(getattr(training_cfg, "learning_rate", 5e-3)),
        min_delta=float(getattr(training_cfg, "min_delta", 1e-5)),
        dataloader_num_workers=int(getattr(dataloader_cfg, "num_workers", 0)),
        dataloader_prefetch_factor=getattr(dataloader_cfg, "prefetch_factor", None),
        dataloader_persistent_workers=bool(getattr(dataloader_cfg, "persistent_workers", False)),
        dataloader_pin_memory=bool(getattr(dataloader_cfg, "pin_memory", False)),
    )

    predictor = PropensityPredictor(
        model=propensity_model,
        alpha=alpha_logits,
        apply_sigmoid=False,
    )
    predicted_per_doc = predictor.predict_per_doc(
        test_data,
        debug_groups=bool(getattr(config, "debug_groups", False)),
        debug_group_limit=int(getattr(config, "debug_group_limit", 5)),
    )

    per_obs_stats = {}
    for obs_count in obs_counts:
        mean_pred, std_pred = aggregate_by_observation_count(
            dominant_pos, predicted_per_doc, displays_per_doc, cutoff, obs_count
        )
        mean_expected, std_expected = aggregate_by_observation_count(
            dominant_pos, expected_propensity_per_doc, displays_per_doc, cutoff, obs_count
        )
        per_obs_stats[obs_count] = (
            mean_pred,
            std_pred,
            mean_expected,
            std_expected,
        )

    csv_path = f"compare_propensity_{base_sig}_{spec.name}_{model_sig}_obs1-12.csv"
    stats_df = build_stats_frame(
        positions=np.arange(cutoff),
        effective_alpha=effective_alpha_logits,
        per_obs_stats=per_obs_stats,
        cutoff=cutoff,
        obs_counts=obs_counts,
    )
    stats_df.to_csv(csv_path, index=False)
    print(f"Saved plot data to {csv_path}")

    for group in obs_groups:
        output_path = (
            f"compare_propensity_{base_sig}_{spec.name}_{model_sig}_obs{group[0]}-{group[-1]}.png"
        )
        plot_multi_panel(
            csv_path=csv_path,
            obs_counts=group,
            output_path=output_path,
        )


if __name__ == "__main__":
    main()
