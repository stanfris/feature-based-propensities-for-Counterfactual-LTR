import hashlib
import logging
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from omegaconf import DictConfig

from feature_based_propensities_for_ULTR.prebuilt import is_prebuilt_click_mode, resolve_prebuilt_click_paths

logger = logging.getLogger(__name__)
_IGNORED_TEST_CLICKS_WARNED: set[int] = set()


@dataclass(frozen=True)
class IPSResolvedConfig:
    model_choice: str
    model_cfg: DictConfig | dict
    position_bias_source: str
    train_clicks: int
    val_clicks: int
    test_set_mode: str
    propensity_method: str
    use_true_propensity: bool
    use_mlp_propensity: bool
    seed: int
    bias_strength: float
    debug: bool
    display_histogram_max: int
    display_histogram_dir: Path | None


def session_click_override(config: DictConfig) -> int | None:
    ips_cfg = getattr(config, "ips", None)
    if ips_cfg is None:
        return None

    if session_document_percentage_override(config) is not None:
        return None

    raw_n_sessions = getattr(ips_cfg, "n_sessions", None)
    if raw_n_sessions is None:
        return None

    n_sessions = int(raw_n_sessions)
    if n_sessions <= 0:
        raise ValueError(f"ips.n_sessions must be > 0, got {n_sessions}.")
    return n_sessions


def session_document_percentage_override(config: DictConfig) -> float | None:
    ips_cfg = getattr(config, "ips", None)
    if ips_cfg is None:
        return None

    raw_percentage = getattr(ips_cfg, "n_session_percentage", None)
    if raw_percentage is None:
        return None

    percentage = float(raw_percentage)
    if not np.isfinite(percentage):
        raise ValueError(f"ips.n_session_percentage must be finite, got {raw_percentage}.")
    if percentage <= 0.0 or percentage > 100.0:
        raise ValueError(
            f"ips.n_session_percentage must be in (0, 100], got {percentage}."
        )
    return percentage


def aggregation_session_override(config: DictConfig) -> int | None:
    if session_document_percentage_override(config) is not None:
        return None
    return session_click_override(config)


def resolve_aggregation_session_budget(
    config: DictConfig,
    train_len: int,
    val_len: int,
) -> int | None:
    percentage = session_document_percentage_override(config)
    if percentage is not None:
        total_sessions = max(int(train_len) + int(val_len), 0)
        if total_sessions <= 0:
            return 0
        requested = int(np.ceil(total_sessions * (percentage / 100.0)))
        return min(requested, total_sessions)
    return aggregation_session_override(config)


def resolve_test_set_mode(config: DictConfig) -> str:
    raw_mode = getattr(config, "test_set_mode", "label_only")
    mode = str(raw_mode).strip().lower().replace("-", "_")
    if mode not in {"label_only", "clicks"}:
        raise ValueError(
            f"Unknown test_set_mode '{raw_mode}'. Expected 'label_only' or 'clicks'."
        )
    if mode == "label_only":
        raw_test_clicks = getattr(config, "test_clicks", None)
        config_key = id(config)
        if (
            raw_test_clicks is not None
            and int(raw_test_clicks) > 0
            and config_key not in _IGNORED_TEST_CLICKS_WARNED
        ):
            logger.warning(
                "test_clicks=%s is ignored because test_set_mode='label_only'. "
                "Set test_set_mode='clicks' to enable legacy click-based test behavior.",
                raw_test_clicks,
            )
            _IGNORED_TEST_CLICKS_WARNED.add(config_key)
    return mode


def effective_click_count(config: DictConfig, split: str) -> int:
    split_norm = str(split).strip().lower()
    if split_norm == "test":
        return int(config.test_clicks)

    override = session_click_override(config)
    if override is not None and split_norm in {"train", "val"}:
        return override

    if split_norm == "train":
        return int(config.train_clicks)
    if split_norm == "val":
        return int(config.val_clicks)
    raise ValueError(f"Unknown split: {split}")


def split_sessions(n_sessions: int, train_len: int, val_len: int) -> tuple[int, int, int]:
    total_sessions = train_len + val_len
    if n_sessions >= total_sessions:
        print("Using all available sessions.")
        return total_sessions, train_len, val_len

    train_ratio = train_len / max(total_sessions, 1)
    n_train = int(np.floor(n_sessions * train_ratio))
    n_val = n_sessions - n_train
    n_train = min(n_train, train_len)
    n_val = min(n_val, val_len)
    print(f"Splitting sessions: {n_train} train, {n_val} val, original: {total_sessions}")
    return n_train + n_val, n_train, n_val


def split_session_budget(n_sessions: int, train_len: int, val_len: int) -> tuple[int, int]:
    """
    Allocate a requested train/val session budget by split ratio without capping.

    This is used when generating simulated click sessions, where queries can be
    sampled with replacement. In that setting, the number of unique queries is
    not a hard upper bound on the number of simulated sessions.
    """
    total_sessions = train_len + val_len
    if total_sessions <= 0 or n_sessions <= 0:
        return 0, 0

    train_ratio = train_len / total_sessions
    n_train = int(np.floor(n_sessions * train_ratio))
    n_val = n_sessions - n_train
    return n_train, n_val


def _normalize_model_choice(value) -> str:
    norm = str(value).strip().lower().replace("_", "-")
    if norm in {"ips"}:
        return "ips"
    if norm in {"naive"}:
        return "naive"
    if norm in {"naive-ho", "naive-2022"}:
        return "naive-ho"
    if norm in {"dm"}:
        return "dm"
    if norm in {"dr"}:
        return "dr"
    if norm in {"max-score"}:
        return "max-score"
    if norm in {"logging-policy"}:
        return "logging-policy"
    raise ValueError(
        f"Unknown ips.model '{value}'. Expected one of 'ips', 'naive', 'naive-ho', 'dm', 'dr', 'max-score', or 'logging-policy'."
    )


def resolve_ips_model_config(ips_cfg: DictConfig):
    model_choice = getattr(ips_cfg, "model", "ips")
    model_params = getattr(ips_cfg, "model_params", None)
    model_choice = _normalize_model_choice(model_choice)
    return model_choice, model_params


def normalize_propensity_method(method) -> str:
    return str(method).replace("-", "_").strip().lower()


def normalize_position_bias_source(value) -> str:
    norm = str(value or "oracle").strip().lower().replace("_", "-")
    if norm == "oracle":
        return "oracle"
    if norm == "csv":
        return "csv"
    if norm == "estimate":
        return "estimate"
    raise ValueError(
        f"Unknown ips.position_bias.source '{value}'. "
        "Expected 'oracle', 'csv', or 'estimate'."
    )


def resolve_ips_config(config: DictConfig) -> IPSResolvedConfig:
    ips_cfg = config.ips
    model_choice, model_cfg = resolve_ips_model_config(ips_cfg)
    if model_cfg is None:
        model_cfg = {}
    position_bias_cfg = getattr(ips_cfg, "position_bias", None)
    position_bias_source = normalize_position_bias_source(
        getattr(position_bias_cfg, "source", "oracle") if position_bias_cfg is not None else "oracle"
    )

    train_clicks = effective_click_count(config, "train")
    val_clicks = effective_click_count(config, "val")
    test_set_mode = resolve_test_set_mode(config)

    propensity_method = normalize_propensity_method(ips_cfg.propensity.method)
    use_true_propensity = propensity_method == "true_propensity"
    use_mlp_propensity = propensity_method in {"propensity_mlp_classifier", "propensity_mlp_regression"}

    display_histogram_dir = getattr(ips_cfg, "display_histogram_dir", None)
    hist_dir = Path(display_histogram_dir) if display_histogram_dir else None

    return IPSResolvedConfig(
        model_choice=model_choice,
        model_cfg=model_cfg,
        position_bias_source=position_bias_source,
        train_clicks=train_clicks,
        val_clicks=val_clicks,
        test_set_mode=test_set_mode,
        propensity_method=propensity_method,
        use_true_propensity=use_true_propensity,
        use_mlp_propensity=use_mlp_propensity,
        seed=int(getattr(config, "random_state", 0)),
        bias_strength=float(config.bias_strength),
        debug=bool(getattr(ips_cfg, "debug", False)),
        display_histogram_max=int(getattr(ips_cfg, "display_histogram_max", 100)),
        display_histogram_dir=hist_dir,
    )


def _abbr_value(value) -> str:
    return str(value).replace(".", "p").replace(" ", "")


def _disjoint_chunk_signature(config: DictConfig, split: str) -> str:
    split_norm = str(split).strip().lower()
    if split_norm not in {"train", "val"}:
        return ""

    data_cfg = getattr(config, "data", None)
    preprocessor_cfg = getattr(data_cfg, "preprocessor", None) if data_cfg is not None else None
    if preprocessor_cfg is None:
        return ""

    enabled = bool(getattr(preprocessor_cfg, "disjoint_query_chunk_mode", False))
    return "_dqchunk1" if enabled else ""


def _drop_zero_relevance_query_signature(config: DictConfig) -> str:
    data_cfg = getattr(config, "data", None)
    preprocessor_cfg = getattr(data_cfg, "preprocessor", None) if data_cfg is not None else None
    if preprocessor_cfg is None:
        return ""

    enabled = bool(
        getattr(preprocessor_cfg, "drop0rel", True)
    )
    return "_drop0rel1" if enabled else "_drop0rel0"


def resolve_dataset_artifact_prefix(config: DictConfig) -> str:
    data_cfg = getattr(config, "data", None)
    raw_prefix = getattr(data_cfg, "artifact_prefix", "") if data_cfg is not None else ""
    if raw_prefix is None:
        return ""

    normalized = re.sub(r"[^a-zA-Z0-9]+", "_", str(raw_prefix).strip().lower()).strip("_")
    if not normalized:
        return ""
    return f"{normalized}_"


def resolve_dataset_label(config: DictConfig) -> str:
    data_cfg = getattr(config, "data", None)
    raw_prefix = getattr(data_cfg, "artifact_prefix", "") if data_cfg is not None else ""
    normalized_prefix = re.sub(r"[^a-zA-Z0-9]+", "_", str(raw_prefix).strip().lower()).strip("_")
    if normalized_prefix:
        return normalized_prefix

    dataset_cfg = getattr(data_cfg, "dataset", None) if data_cfg is not None else None
    target = getattr(dataset_cfg, "_target_", "") if dataset_cfg is not None else ""
    if target:
        return str(target).strip().split(".")[-1]
    return "dataset"


def prefixed_artifact_name(config: DictConfig, base_name: str) -> str:
    return f"{resolve_dataset_artifact_prefix(config)}{base_name}"


def resolve_expected_feature_dim(config: DictConfig) -> int | None:
    data_cfg = getattr(config, "data", None)
    raw_dim = getattr(data_cfg, "expected_feature_dim", None) if data_cfg is not None else None
    if raw_dim is None:
        return None
    return int(raw_dim)


def resolve_data_cutoff(config: DictConfig) -> int | None:
    data_cfg = getattr(config, "data", None)
    preprocessor_cfg = getattr(data_cfg, "preprocessor", None) if data_cfg is not None else None
    raw_top_x = getattr(preprocessor_cfg, "top_x", None) if preprocessor_cfg is not None else None
    if raw_top_x is not None:
        cutoff = int(raw_top_x)
        if cutoff <= 0:
            raise ValueError(f"config.data.preprocessor.top_x must be > 0, got {cutoff}.")
        return cutoff

    raw_cutoff = getattr(data_cfg, "data_cutoff", None) if data_cfg is not None else None
    if raw_cutoff is None:
        return None
    cutoff = int(raw_cutoff)
    if cutoff <= 0:
        raise ValueError(f"config.data.data_cutoff must be > 0, got {cutoff}.")
    return cutoff


def _ranking_signature(config: DictConfig) -> str:
    data_cfg = getattr(config, "data", None)
    preprocessor_cfg = getattr(data_cfg, "preprocessor", None) if data_cfg is not None else None
    if preprocessor_cfg is None:
        preprocessor_cfg = {}

    top_x = getattr(preprocessor_cfg, "top_x", None)
    max_documents_per_query = getattr(preprocessor_cfg, "max_documents_per_query", None)
    if max_documents_per_query is None and data_cfg is not None:
        max_documents_per_query = getattr(data_cfg, "max_documents_per_query", None)
    top_x_sig = "" if top_x is None else f"_topx{_abbr_value(top_x)}"
    max_docs_sig = (
        "" if max_documents_per_query is None else f"_maxdocs{_abbr_value(max_documents_per_query)}"
    )
    return f"_candfullv2{top_x_sig}{max_docs_sig}"


def _logging_policy_training_signature(config: DictConfig) -> str:
    ranker_cfg = getattr(config, "logging_policy_ranker", None)
    sampler_cfg = getattr(config, "logging_policy_sampler", None)
    if ranker_cfg is None and sampler_cfg is None:
        return ""
    parts: list[str] = []
    if ranker_cfg is not None:
        model_type = str(getattr(ranker_cfg, "model_type", "ranker")).lower()
        parts.append(f"_lpr{model_type}")
    if sampler_cfg is not None:
        sampler_target = str(getattr(sampler_cfg, "_target_", "sampler")).split(".")[-1]
        parts.append(f"_lps{sampler_target.lower()}")
    if ranker_cfg is not None:
        raw_pct = getattr(ranker_cfg, "training_data_percentage", 100.0)
        pct = float(raw_pct)
        if not np.isclose(pct, 100.0):
            parts.append(f"_lppct{_abbr_value(pct)}")
    return "".join(parts)


def _force_single_sample_signature(config: DictConfig) -> str:
    ips_cfg = getattr(config, "ips", None)
    if ips_cfg is None:
        return ""
    enabled = bool(getattr(ips_cfg, "force_single_sample", False))
    return "_fsample1" if enabled else ""


def _session_percentage_signature(config: DictConfig, split: str) -> str:
    split_norm = str(split).strip().lower()
    if split_norm not in {"train", "val"}:
        return ""
    percentage = session_document_percentage_override(config)
    if percentage is None:
        return ""
    return f"_spct{_abbr_value(percentage)}"


def _session_override_signature(config: DictConfig, split: str) -> str:
    split_norm = str(split).strip().lower()
    if split_norm not in {"train", "val"}:
        return ""
    if session_click_override(config) is None:
        return ""
    return "_sessallocv2"


def click_signature(config: DictConfig, split: str) -> str:
    split_norm = str(split).strip().lower()
    disjoint_chunk_sig = _disjoint_chunk_signature(config, split)
    drop_zero_relevance_sig = _drop_zero_relevance_query_signature(config)
    ranking_sig = _ranking_signature(config)
    logging_policy_sig = _logging_policy_training_signature(config)
    force_single_sample_sig = _force_single_sample_signature(config)
    session_percentage_sig = _session_percentage_signature(config, split)
    session_override_sig = _session_override_signature(config, split)

    if is_prebuilt_click_mode(config):
        prebuilt_paths = resolve_prebuilt_click_paths(config, must_exist=False)
        tokens: list[str] = []
        for split_name, path in (
            ("train", prebuilt_paths.train),
            ("val", prebuilt_paths.val),
            ("test", prebuilt_paths.test),
        ):
            token = f"{split_name}={path}"
            if path.exists():
                stat = path.stat()
                token += f":size={stat.st_size}:mtime_ns={stat.st_mtime_ns}"
            tokens.append(token)
        if split_norm == "test" and prebuilt_paths.test_clicks is not None:
            token = f"test_clicks={prebuilt_paths.test_clicks}"
            if prebuilt_paths.test_clicks.exists():
                stat = prebuilt_paths.test_clicks.stat()
                token += f":size={stat.st_size}:mtime_ns={stat.st_mtime_ns}"
            tokens.append(token)
        normalized = "|".join(tokens)
        digest = hashlib.md5(normalized.encode()).hexdigest()[:10]
        return f"prebuilt_{digest}{disjoint_chunk_sig}{drop_zero_relevance_sig}{logging_policy_sig}"

    query_sampling_ratios = getattr(config, "query_sampling_ratios", None)
    ratio_sig = ""
    if query_sampling_ratios is not None:
        ratio_sig = (
            f"_qs{hashlib.md5(str(query_sampling_ratios).encode()).hexdigest()[:8]}"
        )
    clicks = effective_click_count(config, split_norm)
    return (
        f"c{_abbr_value(clicks)}_"
        f"ps{_abbr_value(config.policy_strength)}_"
        f"pt{_abbr_value(config.policy_temperature)}_"
        f"bs{_abbr_value(config.bias_strength)}"
        f"{drop_zero_relevance_sig}"
        f"{ratio_sig}"
        f"{disjoint_chunk_sig}"
        f"{force_single_sample_sig}"
        f"{session_percentage_sig}"
        f"{session_override_sig}"
        f"{ranking_sig}"
        f"{logging_policy_sig}"
    )


def test_evaluation_signature(config: DictConfig) -> str:
    drop_zero_relevance_sig = _drop_zero_relevance_query_signature(config)
    ranking_sig = _ranking_signature(config)
    if is_prebuilt_click_mode(config):
        prebuilt_paths = resolve_prebuilt_click_paths(config, must_exist=False)
        path = prebuilt_paths.test
        token = f"test={path}"
        if path.exists():
            stat = path.stat()
            token += f":size={stat.st_size}:mtime_ns={stat.st_mtime_ns}"
        digest = hashlib.md5(token.encode()).hexdigest()[:10]
        return f"labelonly_prebuilt_{digest}"
    return f"labelonly{drop_zero_relevance_sig}{ranking_sig}"


def agg_signature(config: DictConfig, split: str, n_sessions: int | None) -> str:
    percentage = session_document_percentage_override(config)
    split_norm = str(split).strip().lower()
    if percentage is not None and split_norm in {"train", "val"}:
        n_sessions_sig = f"docpct{_abbr_value(percentage)}"
    elif n_sessions is None:
        n_sessions_sig = "all"
    else:
        n_sessions_sig = _abbr_value(n_sessions)
    prebuilt_shared_doc_sig = "_qdsharedv1" if is_prebuilt_click_mode(config) else ""
    cutoff = resolve_data_cutoff(config)
    cutoff_sig = f"_k{cutoff}" if cutoff is not None else ""
    if split_norm == "test" and resolve_test_set_mode(config) == "label_only":
        base_signature = test_evaluation_signature(config)
    else:
        base_signature = click_signature(config, split)
    return (
        f"{base_signature}_ns{n_sessions_sig}"
        f"{prebuilt_shared_doc_sig}{cutoff_sig}"
    )


__all__ = [
    "IPSResolvedConfig",
    "agg_signature",
    "aggregation_session_override",
    "resolve_aggregation_session_budget",
    "click_signature",
    "effective_click_count",
    "normalize_position_bias_source",
    "normalize_propensity_method",
    "prefixed_artifact_name",
    "resolve_test_set_mode",
    "resolve_dataset_artifact_prefix",
    "resolve_dataset_label",
    "resolve_data_cutoff",
    "test_evaluation_signature",
    "resolve_expected_feature_dim",
    "resolve_ips_config",
    "resolve_ips_model_config",
    "session_click_override",
    "session_document_percentage_override",
    "split_session_budget",
    "split_sessions",
]
