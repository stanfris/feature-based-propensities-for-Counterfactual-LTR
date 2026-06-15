import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from omegaconf import DictConfig
from feature_based_propensities_for_ULTR.data.feature_validation import validate_expected_feature_dim
from feature_based_propensities_for_ULTR.data.runtime import (
    load_click_bundle_from_prepared_artifacts,
)
from feature_based_propensities_for_ULTR.experiments.config_resolver import resolve_test_set_mode
from feature_based_propensities_for_ULTR.data.sources import (
    load_or_generate_generated_click_datasets,
    prepare_click_artifacts_from_config,
)


def set_device(device_name: str | None = None):
    """
    Sets the default JAX device for execution.
    If device_name is null, JAX will use the best automatically (GPU/MPS > CPU).
    Otherwise, it forces JAX to use the specified platform (e.g., 'cpu', 'mps', 'cuda').
    """
    if device_name is not None:
        jax.config.update("jax_platforms", device_name)
    
    print(f"JAX will use device backend: {jax.default_backend().upper()} - Array Devices: {jax.devices()}")


def reduce_per_query(loss: Array, where: Array) -> Array:
    loss = loss.reshape(len(loss), -1)
    where = where.reshape(len(where), -1)

    # Adopt Rax safe_reduce as jnp.mean can return NaN if all inputs are 0,
    # which happens easily for pairwise loss functions without any valid pair.
    # Replace NaNs with 0 after reduce, but propagate if the loss already contains NaNs:
    is_input_valid = jnp.logical_not(jnp.any(jnp.isnan(loss)))
    output = jnp.mean(loss, where=where, axis=1)
    output = jnp.where(jnp.isnan(output) & is_input_valid, 0.0, output)

    return output


def _load_prebuilt_click_datasets(config: DictConfig, prepared_artifacts=None):
    if prepared_artifacts is None:
        prepared_artifacts = prepare_click_artifacts_from_config(config)
    include_test_click_dataset = resolve_test_set_mode(config) == "clicks"
    loaded = load_click_bundle_from_prepared_artifacts(
        prepared_artifacts,
        include_test_click_dataset=include_test_click_dataset,
    )

    validate_expected_feature_dim(
        config,
        int(np.asarray(loaded.train.lp_query_doc_features).shape[-1]),
        source="prebuilt_train_click_dataset.lp_query_doc_features",
    )
    print(
        "✅ Loaded prebuilt click datasets: "
        f"train={loaded.prepared_artifacts.split_paths['train']}, "
        f"val={loaded.prepared_artifacts.split_paths['val']}, "
        f"test(labels)={loaded.prepared_artifacts.split_paths['test']}"
    )
    return loaded.as_legacy_tuple()


def load_or_generate_click_datasets(
    config: DictConfig,
    save_subdir: str,
    *,
    varying: bool = False,
    logging_policy_ckpt_dir: str | None = None,
):
    resolve_test_set_mode(config)
    prepared_artifacts = prepare_click_artifacts_from_config(config)
    if prepared_artifacts is not None:
        return _load_prebuilt_click_datasets(config, prepared_artifacts=prepared_artifacts)
    return load_or_generate_generated_click_datasets(
        config,
        save_subdir=save_subdir,
        varying=varying,
        logging_policy_ckpt_dir=logging_policy_ckpt_dir,
    )
