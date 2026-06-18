import time
from dataclasses import dataclass
from pathlib import Path
from pprint import pformat
from typing import Any, Protocol

import logging
from feature_based_propensities_for_ULTR.experiments.tracking import Tracker

logger = logging.getLogger(__name__)

import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import DictConfig

from feature_based_propensities_for_ULTR.models.policy_models import build_policy_model
from feature_based_propensities_for_ULTR.data.runtime import AggregatedDataBundle, DataBundle, validate_aggregated_bundle
from feature_based_propensities_for_ULTR.prebuilt import is_prebuilt_click_mode
from feature_based_propensities_for_ULTR.simulation import get_position_bias

from .config_resolver import (
    IPSResolvedConfig,
    resolve_dataset_label,
    resolve_ips_config,
)
from .click_prediction import evaluate_click_prediction, flatten_click_prediction_metrics
from .data_pipeline import prepare_data_bundle
from .diagnostics import (
    log_aggregated_dataset_stats,
    log_aggregated_sizes,
    log_click_stats,
    log_doc_weight_stats,
    log_doc_weight_true_stats,
    log_missing_oracle_propensity,
    log_propensity_diagnostics,
    log_raw_click_dataset_stats,
    log_unique_values,
    propensity_compare,
    propensity_stats,
    weights_stats,
)
from .model_pipeline import (
    build_logging_policy_model,
    evaluate_model,
    train_policy_model,
)
from .position_bias_pipeline import (
    estimate_position_bias_values,
    load_position_bias_from_csv,
    write_estimated_position_bias_json,
)
from .propensity_pipeline import PropensityBundle, compute_propensity_clipping, predict_propensity
from .regression_pipeline import RegressionBundle, train_regression_bundle
from .reporting import build_report, write_results


class ScoringModel(Protocol):
    def score_feature_matrix(self, feature_matrix: np.ndarray) -> np.ndarray:
        ...


@dataclass
class DocWeights:
    train: np.ndarray
    val: np.ndarray
    train_true: np.ndarray | None = None
    val_true: np.ndarray | None = None


@dataclass
class IPSContext:
    config: DictConfig
    resolved: IPSResolvedConfig
    ips_cfg: DictConfig
    trainer_cfg: DictConfig
    dataset_label: str
    output_path: Path
    data_bundle: DataBundle | None = None
    aggregated: AggregatedDataBundle | None = None
    alpha_logits: jnp.ndarray | None = None
    alpha: jnp.ndarray | None = None
    mean_alpha_logit: float | None = None
    position_bias_csv_path: str | None = None
    position_bias_csv_method: str | None = None
    position_bias_json_path: str | None = None
    position_bias_estimator: str | None = None
    propensity_bundle: PropensityBundle | None = None
    dr_frequency_propensity_bundle: PropensityBundle | None = None
    propensity_clipping: float | None = None
    regression_bundle: RegressionBundle | None = None
    doc_weights: DocWeights | None = None
    model: ScoringModel | Any | None = None
    metrics: dict[str, Any] | None = None
    click_prediction: dict[str, Any] | None = None
    tracker: Tracker | None = None


def _pipeline_print(message: str) -> None:
    print(f"[IPSPipeline] {message}", flush=True)


def resolve_context(config: DictConfig) -> IPSContext:
    resolved = resolve_ips_config(config)
    _validate_prebuilt_mode_restrictions(config, resolved)
    ips_cfg = config.ips
    trainer_cfg = ips_cfg.trainer
    dataset_label = resolve_dataset_label(config)
    output_path = Path(ips_cfg.output_path)
    return IPSContext(
        config=config,
        resolved=resolved,
        ips_cfg=ips_cfg,
        trainer_cfg=trainer_cfg,
        dataset_label=dataset_label,
        output_path=output_path,
    )

def _validate_prebuilt_mode_restrictions(
    config: DictConfig,
    resolved: IPSResolvedConfig,
) -> None:
    if not is_prebuilt_click_mode(config):
        return

    if resolved.model_choice == "logging-policy":
        raise ValueError(
            "data.click_data_mode='prebuilt' does not support ips.model='logging-policy'. "
            "This mode is disabled because it would require loading/training a logging policy."
        )

    if resolved.use_true_propensity:
        raise ValueError(
            "data.click_data_mode='prebuilt' does not support propensity_model='true_propensity'. "
            "Use a data-driven propensity estimator "
            "(e.g. frequency-based, kmeans, knn, cosine, euclidean, or MLP)."
        )


def prepare_data(context: IPSContext) -> IPSContext:
    context.data_bundle = prepare_data_bundle(context.config, context.resolved)
    context.aggregated = context.data_bundle.aggregated
    if context.data_bundle.clicks is not None and context.data_bundle.clicks.test is None:
        logger.info("Test split uses label-only evaluation; raw test click diagnostics are skipped.")
    _log_aggregated_bundle(context.aggregated)
    if context.resolved.debug:
        if context.data_bundle.clicks is not None:
            log_raw_click_dataset_stats("train", context.data_bundle.clicks.train)
            log_raw_click_dataset_stats("val", context.data_bundle.clicks.val)
            if context.data_bundle.clicks.test is not None:
                log_raw_click_dataset_stats("test", context.data_bundle.clicks.test)
        log_aggregated_dataset_stats("train", context.aggregated.train)
        log_aggregated_dataset_stats("val", context.aggregated.val)
        log_aggregated_dataset_stats("test", context.aggregated.test)
    return context


def compute_position_bias(context: IPSContext) -> IPSContext:
    logger.info(f"Running estimator pipeline with model='{context.resolved.model_choice}'")
    logger.info("Computing position bias (alpha)...")
    position_bias_source = context.resolved.position_bias_source
    if (
        position_bias_source == "oracle"
        and is_prebuilt_click_mode(context.config)
        and context.resolved.model_choice in {"ips", "dm", "dr"}
    ):
        logger.warning(
            "Prebuilt-click mode uses an assumed position-bias curve from get_position_bias "
            "(controlled by bias_strength=%.4f). Oracle propensities are not available in this mode, "
            "so IPS/DM/DR results are sensitivity-dependent with respect to this assumption.",
            float(context.resolved.bias_strength),
        )
    aggregated = _require_aggregated(context)
    if position_bias_source == "oracle":
        alpha_logits = jnp.asarray(
            get_position_bias(aggregated.cutoff, strength=context.resolved.bias_strength)
        )
        context.position_bias_csv_path = None
        context.position_bias_csv_method = None
        context.position_bias_json_path = None
        context.position_bias_estimator = None
    elif position_bias_source == "csv":
        alpha_logits_np, alpha_np, csv_path, csv_method = load_position_bias_from_csv(
            config=context.config,
            dataset_label=context.dataset_label,
            cutoff=int(aggregated.cutoff),
            n_sessions_used=int(aggregated.n_sessions_used),
        )
        alpha_logits = jnp.asarray(alpha_logits_np)
        alpha = jnp.asarray(alpha_np)
        context.position_bias_csv_path = str(csv_path)
        context.position_bias_csv_method = csv_method
        context.position_bias_json_path = None
        context.position_bias_estimator = csv_method
        logger.info(
            "Loaded CSV position bias from %s (method=%s): %s",
            csv_path,
            csv_method,
            np.asarray(alpha_logits),
        )
    elif position_bias_source == "estimate":
        data_bundle = _require_data_bundle(context)
        if data_bundle.clicks is None:
            raise RuntimeError(
                "In-run position-bias estimation requires click datasets. "
                "Ensure click data is generated or loaded before compute_position_bias."
            )
        alpha_logits_np, alpha_np, position_bias_df, examination_np, estimator_name = (
            estimate_position_bias_values(
                config=context.config,
                click_bundle=data_bundle.clicks,
                cutoff=int(aggregated.cutoff),
                dataset_label=context.dataset_label,
            )
        )
        alpha_logits = jnp.asarray(alpha_logits_np)
        alpha = jnp.asarray(alpha_np)
        context.position_bias_csv_path = None
        context.position_bias_csv_method = None
        context.position_bias_estimator = estimator_name
        json_path = write_estimated_position_bias_json(
            output_path=context.output_path.parent / "position_bias.json",
            position_bias_df=position_bias_df,
            alpha_logits=np.asarray(alpha_logits_np),
            alpha=np.asarray(alpha_np),
            examination=np.asarray(examination_np),
        )
        context.position_bias_json_path = str(json_path)
        logger.info(
            "Estimated position bias in-run (estimator=%s): %s",
            estimator_name,
            np.asarray(alpha_logits),
        )
    else:
        raise ValueError(
            f"Unknown position_bias_source '{position_bias_source}'. "
            "Expected 'oracle', 'csv', or 'estimate'."
        )
    if position_bias_source == "oracle":
        alpha = jax.nn.sigmoid(alpha_logits)
    context.alpha_logits = alpha_logits
    context.alpha = alpha
    context.mean_alpha_logit = float(np.mean(np.asarray(alpha_logits)))
    _print_position_bias(
        source=position_bias_source,
        alpha_logits=np.asarray(alpha_logits, dtype=np.float64),
        alpha=np.asarray(alpha, dtype=np.float64),
    )
    return context


def maybe_compute_propensity(context: IPSContext) -> IPSContext:
    if not _needs_propensity(context.resolved):
        context.propensity_bundle = None
        context.propensity_clipping = None
        return context

    aggregated = _require_aggregated(context)
    alpha = _require_alpha(context)
    alpha_logits = _require_alpha_logits(context)
    data_bundle = _require_data_bundle(context)

    diagnose_true_propensity = bool(getattr(context.ips_cfg.propensity, "debug_compare_true", False))
    _pipeline_print(
        f"Starting propensity stage with method='{context.resolved.propensity_method}'"
    )
    context.propensity_bundle = predict_propensity(
        config=context.config,
        method=context.resolved.propensity_method,
        train_data=aggregated.train,
        val_data=aggregated.val,
        train_click_dataset=data_bundle.clicks.train if data_bundle.clicks else None,
        val_click_dataset=data_bundle.clicks.val if data_bundle.clicks else None,
        alpha=np.asarray(alpha),
        alpha_logits=np.asarray(alpha_logits),
        mean_alpha_logit=float(context.mean_alpha_logit),
        seed=context.resolved.seed,
        use_true_propensity=context.resolved.use_true_propensity,
        use_mlp_propensity=context.resolved.use_mlp_propensity,
        compute_true=diagnose_true_propensity,
    )
    if diagnose_true_propensity:
        log_propensity_diagnostics(
            tracker=context.tracker,
            propensity_bundle=context.propensity_bundle,
            debug=context.resolved.debug,
            diagnose_true_propensity=diagnose_true_propensity,
            use_true_propensity=context.resolved.use_true_propensity,
        )
    _pipeline_print("Propensity stage completed")

    context.propensity_clipping = compute_propensity_clipping(
        context.trainer_cfg,
        aggregated.n_sessions_used,
    )
    context.dr_frequency_propensity_bundle = None
    if context.resolved.model_choice == "dr":
        _pipeline_print("Computing auxiliary frequency-based propensity bundle for DR")
        context.dr_frequency_propensity_bundle = predict_propensity(
            config=context.config,
            method="frequency_based",
            train_data=aggregated.train,
            val_data=aggregated.val,
            train_click_dataset=data_bundle.clicks.train if data_bundle.clicks else None,
            val_click_dataset=data_bundle.clicks.val if data_bundle.clicks else None,
            alpha=np.asarray(alpha),
            alpha_logits=np.asarray(alpha_logits),
            mean_alpha_logit=float(context.mean_alpha_logit),
            seed=context.resolved.seed,
            use_true_propensity=False,
            use_mlp_propensity=False,
            compute_true=False,
        )
        _pipeline_print("Auxiliary DR propensity bundle completed")
    return context


def maybe_compute_regression(context: IPSContext) -> IPSContext:
    if not _needs_regression(context.resolved):
        context.regression_bundle = None
        return context

    if context.propensity_bundle is None:
        raise RuntimeError("DM/DR training requires propensity estimates.")

    aggregated = _require_aggregated(context)
    context.regression_bundle = train_regression_bundle(
        config=context.config,
        train_data=aggregated.train,
        val_data=aggregated.val,
        train_propensity=context.propensity_bundle.train,
        val_propensity=context.propensity_bundle.val,
        propensity_clipping=context.propensity_clipping,
        seed=context.resolved.seed,
        tracker=context.tracker,
    )
    return context


def compute_policy_weights(context: IPSContext) -> IPSContext:
    if context.resolved.model_choice == "logging-policy":
        context.doc_weights = None
        return context

    aggregated = _require_aggregated(context)
    policy_model = build_policy_model(context.resolved.model_choice)
    policy_label = {
        "naive": "Naive",
        "naive-ho": "Naive-HO",
        "dm": "DM",
        "dr": "DR",
        "ips": "IPS",
        "max-score": "MaxScore",
        "logging-policy": "LoggingPolicy",
    }[policy_model.name]

    logger.info(f"Computing {policy_label} doc weights...")
    start = time.time()
    train_propensity = context.propensity_bundle.train if context.propensity_bundle is not None else None
    val_propensity = context.propensity_bundle.val if context.propensity_bundle is not None else None
    train_propensity_for_denom = None
    val_propensity_for_denom = None
    if context.resolved.model_choice == "dr" and context.dr_frequency_propensity_bundle is not None:
        # DR spec: use frequency-based propensity in the correction term,
        # but keep denominator/clipping based on the selected propensity method.
        train_propensity_for_denom = train_propensity
        val_propensity_for_denom = val_propensity
        train_propensity = context.dr_frequency_propensity_bundle.train
        val_propensity = context.dr_frequency_propensity_bundle.val

    train_doc_weights = policy_model.compute_weights(
        aggregated.train,
        propensity=train_propensity,
        propensity_for_denom=train_propensity_for_denom,
        propensity_clipping=context.propensity_clipping,
        regression_pred=context.regression_bundle.train_pred if context.regression_bundle is not None else None,
    )
    val_doc_weights = policy_model.compute_weights(
        aggregated.val,
        propensity=val_propensity,
        propensity_for_denom=val_propensity_for_denom,
        propensity_clipping=context.propensity_clipping,
        regression_pred=context.regression_bundle.val_pred if context.regression_bundle is not None else None,
    )
    logger.info("Doc weights computed in %.2f seconds" % (time.time() - start))

    log_doc_weight_stats(
        tracker=context.tracker,
        debug=context.resolved.debug,
        train_doc_weights=train_doc_weights,
        val_doc_weights=val_doc_weights,
    )
    if context.resolved.debug:
        train_nonzero_ratio = float(np.mean(np.asarray(train_doc_weights) > 0))
        val_nonzero_ratio = float(np.mean(np.asarray(val_doc_weights) > 0))
        logger.info(
            "[Weights signal] nonzero_ratio(train=%.4f, val=%.4f)",
            train_nonzero_ratio,
            val_nonzero_ratio,
        )
        train_labels = np.asarray(aggregated.train.label_vector, dtype=np.float64)
        val_labels = np.asarray(aggregated.val.label_vector, dtype=np.float64)
        train_w = np.asarray(train_doc_weights, dtype=np.float64)
        val_w = np.asarray(val_doc_weights, dtype=np.float64)
        if train_w.size > 1 and np.std(train_w) > 0 and np.std(train_labels) > 0:
            corr_train = float(np.corrcoef(train_w, train_labels)[0, 1])
            logger.info("[Weights signal] corr(train_weights, train_labels)=%.4f", corr_train)
        if val_w.size > 1 and np.std(val_w) > 0 and np.std(val_labels) > 0:
            corr_val = float(np.corrcoef(val_w, val_labels)[0, 1])
            logger.info("[Weights signal] corr(val_weights, val_labels)=%.4f", corr_val)

    train_doc_weights_true = None
    val_doc_weights_true = None
    if (
        context.resolved.model_choice == "ips"
        and context.propensity_bundle is not None
        and context.propensity_bundle.true_train is not None
        and context.propensity_bundle.true_val is not None
    ):
        train_doc_weights_true = policy_model.compute_weights(
            aggregated.train,
            propensity_clipping=context.propensity_clipping,
            propensity=context.propensity_bundle.true_train,
        )
        val_doc_weights_true = policy_model.compute_weights(
            aggregated.val,
            propensity_clipping=context.propensity_clipping,
            propensity=context.propensity_bundle.true_val,
        )
        log_doc_weight_true_stats(
            tracker=context.tracker,
            debug=context.resolved.debug,
            train_doc_weights_true=train_doc_weights_true,
            val_doc_weights_true=val_doc_weights_true,
        )

    log_missing_oracle_propensity(
        debug=context.resolved.debug,
        model_choice=context.resolved.model_choice,
        propensity_bundle=context.propensity_bundle,
    )

    context.doc_weights = DocWeights(
        train=train_doc_weights,
        val=val_doc_weights,
        train_true=train_doc_weights_true,
        val_true=val_doc_weights_true,
    )
    return context


def train_model(context: IPSContext) -> IPSContext:
    aggregated = _require_aggregated(context)
    alpha = _require_alpha(context)

    if context.resolved.model_choice == "logging-policy":
        logger.info(
            "Using logging-policy ranker with temperature-based randomization "
            f"(policy_temperature={float(getattr(context.config, 'policy_temperature', 0.0)):.4f})."
        )
        logger.info("Note: ips.n_sessions does not affect logging-policy evaluation (no IPS/DM/DR training).")
        context.model = build_logging_policy_model(
            config=context.config,
            data_split=aggregated.test,
        )
        return context

    if context.doc_weights is None:
        raise RuntimeError("Policy training requires computed document weights.")

    context.model = train_policy_model(
        config=context.config,
        model_choice=context.resolved.model_choice,
        model_cfg=context.resolved.model_cfg,
        train_data=aggregated.train,
        val_data=aggregated.val,
        train_doc_weights=context.doc_weights.train,
        val_doc_weights=context.doc_weights.val,
        alpha=alpha,
        seed=context.resolved.seed,
    )
    return context


def evaluate_and_report(context: IPSContext) -> dict[str, Any]:
    aggregated = _require_aggregated(context)
    alpha = _require_alpha(context)
    if context.model is None:
        raise RuntimeError("Missing trained model.")

    model_output_multiplier = float(getattr(context.resolved.model_cfg, "model_output_multiplier", 1.0))
    eval_label_scale = float(getattr(context.ips_cfg, "eval_label_scale", 0.25))
    context.metrics = evaluate_model(
        model=context.model,
        test_data=aggregated.test,
        alpha=alpha,
        n_samples=context.trainer_cfg.n_eval_samples,
        model_output_multiplier=model_output_multiplier,
        eval_label_scale=eval_label_scale,
    )

    click_eval_cfg = getattr(context.ips_cfg, "click_prediction_eval", None)
    click_eval_enabled = bool(getattr(click_eval_cfg, "enabled", False)) if click_eval_cfg is not None else False
    context.click_prediction = None
    if click_eval_enabled:
        data_bundle = _require_data_bundle(context)
        if data_bundle.clicks is None or data_bundle.clicks.test is None:
            raise RuntimeError(
                "Click prediction evaluation requires a held-out test click dataset. "
                "Set test_set_mode='clicks' and ensure prebuilt test_clicks artifacts exist."
            )
        prediction_output_path = getattr(click_eval_cfg, "output_path", None)
        context.click_prediction = evaluate_click_prediction(
            model=context.model,
            click_dataset=data_bundle.clicks.test,
            alpha=np.asarray(alpha),
            model_output_multiplier=model_output_multiplier,
            prediction_output_path=prediction_output_path,
        )

    needs_propensity = _needs_propensity(context.resolved)
    report = build_report(
        dataset=context.dataset_label,
        model=context.resolved.model_choice,
        n_sessions_used=int(aggregated.n_sessions_used),
        cutoff=int(aggregated.cutoff),
        position_bias_source=context.resolved.position_bias_source,
        position_bias_csv_path=context.position_bias_csv_path,
        position_bias_csv_method=context.position_bias_csv_method,
        position_bias_json_path=context.position_bias_json_path,
        position_bias_estimator=context.position_bias_estimator,
        alpha_clip=None if context.propensity_clipping is None else float(context.propensity_clipping),
        propensity_method=context.resolved.propensity_method if needs_propensity else None,
        propensity_checkpoint_dir=None,
        metrics=context.metrics,
        click_prediction=context.click_prediction,
    )
    write_results(report, output_path=context.output_path)
    return report.results


def _needs_propensity(resolved: IPSResolvedConfig) -> bool:
    return resolved.model_choice in {"ips", "dm", "dr"}


def _needs_regression(resolved: IPSResolvedConfig) -> bool:
    return resolved.model_choice in {"dm", "dr"}


def _require_data_bundle(context: IPSContext) -> DataBundle:
    if context.data_bundle is None:
        raise RuntimeError("Missing data bundle. Call prepare_data first.")
    return context.data_bundle


def _require_aggregated(context: IPSContext) -> AggregatedDataBundle:
    if context.aggregated is None:
        raise RuntimeError("Missing aggregated data. Call prepare_data first.")
    return context.aggregated


def _require_alpha(context: IPSContext) -> jnp.ndarray:
    if context.alpha is None:
        raise RuntimeError("Missing alpha values. Call compute_position_bias first.")
    return context.alpha


def _require_alpha_logits(context: IPSContext) -> jnp.ndarray:
    if context.alpha_logits is None:
        raise RuntimeError("Missing alpha logits. Call compute_position_bias first.")
    return context.alpha_logits


def _print_position_bias(
    *,
    source: str,
    alpha_logits: np.ndarray,
    alpha: np.ndarray,
) -> None:
    logits_str = np.array2string(alpha_logits, precision=6, separator=", ")
    alpha_str = np.array2string(alpha, precision=6, separator=", ")
    _pipeline_print(f"Position bias source='{source}'")
    _pipeline_print(f"Position bias logits={logits_str}")
    _pipeline_print(f"Position bias probs={alpha_str}")


def _log_aggregated_bundle(aggregated: AggregatedDataBundle) -> None:
    validate_aggregated_bundle(aggregated)
    log_aggregated_sizes(
        aggregated.train,
        aggregated.val,
        aggregated.test,
        int(aggregated.cutoff),
    )
    log_click_stats("Train", aggregated.train)
    log_click_stats("Validation", aggregated.val)
    log_click_stats("Test", aggregated.test)


class IPSExperimentPipeline:
    def __init__(self, config: DictConfig):
        self.config = config

    def run(self) -> dict[str, Any]:
        from omegaconf import OmegaConf
        use_wandb = getattr(self.config, "use_wandb", False)
        
        tracker = Tracker(
            name=__name__,
            config=OmegaConf.to_container(self.config, resolve=True),
            use_wandb=use_wandb
        )
        if use_wandb and not tracker.use_wandb:
            logger.warning("Disabling wandb for trainer components because initialization failed.")
            self.config.use_wandb = False
            if "ips" in self.config and "trainer" in self.config.ips:
                self.config.ips.trainer.use_wandb = False

        context = resolve_context(self.config)
        context.tracker = tracker
        prepare_data(context)
        compute_position_bias(context)
        maybe_compute_propensity(context)
        maybe_compute_regression(context)
        compute_policy_weights(context)
        train_model(context)
        
        results = evaluate_and_report(context)
        logger.info("Final evaluation results:\n%s", pformat(results, sort_dicts=True))
        print("Final evaluation results:")
        print(pformat(results, sort_dicts=True))

        metrics = results.get("metrics", {})
        if isinstance(metrics, dict):
            tracker.log_metrics({"eval/" + k: float(v) for k, v in metrics.items()})
        else:
            tracker.log_metrics({"eval/metrics": metrics})
        click_prediction = results.get("click_prediction")
        if isinstance(click_prediction, dict):
            tracker.log_metrics(flatten_click_prediction_metrics(click_prediction))
        tracker.finish()
                
        return results


__all__ = [
    "IPSContext",
    "IPSExperimentPipeline",
    "compute_policy_weights",
    "compute_position_bias",
    "evaluate_and_report",
    "maybe_compute_propensity",
    "maybe_compute_regression",
    "prepare_data",
    "resolve_context",
    "train_model",
]
