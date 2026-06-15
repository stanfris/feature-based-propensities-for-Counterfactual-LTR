import logging
from typing import Optional, TYPE_CHECKING
if TYPE_CHECKING:
    from feature_based_propensities_for_ULTR.experiments.propensity_pipeline import PropensityBundle
    from feature_based_propensities_for_ULTR.experiments.tracking import Tracker
import numpy as np

logger = logging.getLogger(__name__)
import jax.numpy as jnp

try:
    import wandb
except ImportError:
    wandb = None


def log_aggregated_sizes(train_data, val_data, test_data, cutoff: int) -> None:
    logger.info(
        f"[Data] train_docs={train_data.num_docs()}, val_docs={val_data.num_docs()}, "
        f"test_docs={test_data.num_docs()}, cutoff={cutoff}"
    )
    if wandb is None or wandb.run is None:
        return
    wandb.run.summary["dataset/train_docs"] = train_data.num_docs()
    wandb.run.summary["dataset/val_docs"] = val_data.num_docs()
    wandb.run.summary["dataset/test_docs"] = test_data.num_docs()
    wandb.run.summary["dataset/cutoff"] = cutoff


def log_click_stats(name: str, data) -> None:
    clicks = jnp.asarray(getattr(data, "clicks", 0.0))
    displays = jnp.asarray(getattr(data, "displays", 0.0))
    clicks_per_doc = jnp.asarray(getattr(data, "clicks_per_doc", 0.0))
    displays_per_doc = jnp.asarray(getattr(data, "displays_per_doc", 0.0))

    total_clicks = float(jnp.sum(clicks))
    total_displays = float(jnp.sum(displays))
    total_clicks_per_doc = float(jnp.sum(clicks_per_doc))
    total_displays_per_doc = float(jnp.sum(displays_per_doc))
    ctr = total_clicks / max(total_displays, 1.0)
    ctr_per_doc = total_clicks_per_doc / max(total_displays_per_doc, 1.0)

    logger.info(
        f"[{name}] clicks={total_clicks:.0f}, displays={total_displays:.0f}, "
        f"ctr={ctr:.4f}, ctr_per_doc={ctr_per_doc:.4f}"
    )
    if wandb is None or wandb.run is None:
        return
    wandb.log({
        f"{name}/total_clicks": total_clicks,
        f"{name}/total_displays": total_displays,
        f"{name}/ctr": ctr,
        f"{name}/clicks_per_doc": total_clicks_per_doc,
        f"{name}/displays_per_doc": total_displays_per_doc,
        f"{name}/ctr_per_doc": ctr_per_doc,
    })


def log_raw_click_dataset_stats(name: str, click_dataset) -> None:
    query = np.asarray(getattr(click_dataset, "query", []), dtype=np.int64).reshape(-1)
    mask = np.asarray(getattr(click_dataset, "mask", []), dtype=bool)
    n = np.asarray(getattr(click_dataset, "n", []), dtype=np.int64).reshape(-1)
    query_doc_ids = np.asarray(getattr(click_dataset, "query_doc_ids", []), dtype=np.int64)

    if query.size == 0 or mask.size == 0:
        logger.info("[%s raw] empty click dataset; skipping raw stats", name)
        return

    n_sessions = int(query.shape[0])
    docs_per_session = n if n.size == n_sessions else mask.sum(axis=1, dtype=np.int64)
    unique_queries, sessions_per_query = np.unique(query, return_counts=True)

    logger.info(
        "[%s raw] sessions=%d unique_queries=%d docs/session(mean=%.2f p50=%d p95=%d max=%d)",
        name,
        n_sessions,
        int(unique_queries.shape[0]),
        float(np.mean(docs_per_session)),
        int(np.percentile(docs_per_session, 50)),
        int(np.percentile(docs_per_session, 95)),
        int(np.max(docs_per_session)),
    )
    logger.info(
        "[%s raw] sessions/query(mean=%.2f p50=%d p95=%d max=%d)",
        name,
        float(np.mean(sessions_per_query)),
        int(np.percentile(sessions_per_query, 50)),
        int(np.percentile(sessions_per_query, 95)),
        int(np.max(sessions_per_query)),
    )

    flat_valid = mask.reshape(-1) & (query_doc_ids.reshape(-1) >= 0)
    if not np.any(flat_valid):
        logger.info("[%s raw] no valid (query,doc) pairs under mask", name)
        return

    repeated_query = np.repeat(query, mask.shape[1])[flat_valid]
    repeated_doc = query_doc_ids.reshape(-1)[flat_valid]
    order = np.lexsort((repeated_doc, repeated_query))
    q_sorted = repeated_query[order]
    d_sorted = repeated_doc[order]
    is_new = np.empty(q_sorted.shape[0], dtype=bool)
    is_new[0] = True
    is_new[1:] = (q_sorted[1:] != q_sorted[:-1]) | (d_sorted[1:] != d_sorted[:-1])
    starts = np.flatnonzero(is_new)
    pair_displays = np.diff(np.append(starts, q_sorted.shape[0]))

    hist_cap = 5
    hist = {k: int(np.sum(pair_displays == k)) for k in range(1, hist_cap + 1)}
    logger.info(
        "[%s raw] query-doc displays: pairs=%d share_gt1=%.4f max=%d hist(1..%d)=%s",
        name,
        int(pair_displays.shape[0]),
        float(np.mean(pair_displays > 1)),
        int(np.max(pair_displays)),
        hist_cap,
        hist,
    )


def log_aggregated_dataset_stats(name: str, data) -> None:
    displays_per_doc = np.asarray(getattr(data, "displays_per_doc", []), dtype=np.int64).reshape(-1)
    clicks_per_doc = np.asarray(getattr(data, "clicks_per_doc", []), dtype=np.int64).reshape(-1)
    query_freq = np.asarray(getattr(data, "query_freq", []), dtype=np.int64).reshape(-1)
    sessions = np.asarray(getattr(data, "sessions", []), dtype=np.int64).reshape(-1)

    if displays_per_doc.size == 0:
        logger.info("[%s agg] empty aggregated dataset; skipping aggregated stats", name)
        return

    nonzero = displays_per_doc > 0
    logger.info(
        "[%s agg] docs=%d sessions=%d queries=%d displays/doc(mean=%.2f p50=%d p95=%d max=%d) nonzero_display_ratio=%.4f",
        name,
        int(displays_per_doc.shape[0]),
        int(sessions.shape[0]),
        int(query_freq.shape[0]),
        float(np.mean(displays_per_doc)),
        int(np.percentile(displays_per_doc, 50)),
        int(np.percentile(displays_per_doc, 95)),
        int(np.max(displays_per_doc)),
        float(np.mean(nonzero)),
    )
    logger.info(
        "[%s agg] query_freq(mean=%.2f p50=%d p95=%d max=%d) ctr=%.4f",
        name,
        float(np.mean(query_freq)) if query_freq.size else 0.0,
        int(np.percentile(query_freq, 50)) if query_freq.size else 0,
        int(np.percentile(query_freq, 95)) if query_freq.size else 0,
        int(np.max(query_freq)) if query_freq.size else 0,
        float(np.sum(clicks_per_doc) / max(np.sum(displays_per_doc), 1)),
    )


def propensity_stats(name: str, values: np.ndarray) -> None:
    values = np.asarray(values, dtype=np.float64)
    valid = np.isfinite(values)
    if not np.any(valid):
        return
    v = values[valid]
    pcts = np.percentile(v, [0, 1, 5, 50, 95, 99, 100])
    logger.info(
        f"[{name}] propensity: n={v.size}, mean={v.mean():.6f}, std={v.std():.6f}, "
        f"min={pcts[0]:.6f}, p01={pcts[1]:.6f}, p50={pcts[3]:.6f}, p99={pcts[5]:.6f}, max={pcts[6]:.6f}"
    )
    if wandb is None or wandb.run is None:
        return
    wandb.log({
        f"{name}/propensity_n": v.size,
        f"{name}/propensity_min": pcts[0],
        f"{name}/propensity_p01": pcts[1],
        f"{name}/propensity_p05": pcts[2],
        f"{name}/propensity_p50": pcts[3],
        f"{name}/propensity_p95": pcts[4],
        f"{name}/propensity_p99": pcts[5],
        f"{name}/propensity_max": pcts[6],
        f"{name}/propensity_mean": v.mean(),
        f"{name}/propensity_std": v.std(),
    })


def propensity_compare(name: str, pred: np.ndarray, true: np.ndarray) -> None:
    pred = np.asarray(pred, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
    valid = np.isfinite(pred) & np.isfinite(true)
    if not np.any(valid):
        return
    p = pred[valid]
    t = true[valid]
    diff = p - t
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    rel_mae = float(np.mean(np.abs(diff) / np.maximum(np.abs(t), 1e-8)))
    if p.size > 1 and np.std(p) > 0 and np.std(t) > 0:
        corr = float(np.corrcoef(p, t)[0, 1])
    else:
        corr = float("nan")

    logger.info(
        f"[{name}] propensity comparison: mae={mae:.6f}, rmse={rmse:.6f}, "
        f"rel_mae={rel_mae:.6f}, corr={corr:.4f}"
    )
    if wandb is None or wandb.run is None:
        return
    wandb.log({
        f"{name}/propensity_mae": mae,
        f"{name}/propensity_rmse": rmse,
        f"{name}/propensity_rel_mae": rel_mae,
        f"{name}/propensity_corr": corr,
    })


def weights_stats(name: str, weights: np.ndarray) -> None:
    weights = np.asarray(weights, dtype=np.float64)
    valid = np.isfinite(weights)
    if not np.any(valid):
        return
    w = weights[valid]
    pcts = np.percentile(w, [0, 1, 5, 50, 95, 99, 100])
    finite_ratio = float(np.mean(valid))

    logger.info(
        f"[{name}] weights: n={w.size}, mean={w.mean():.6f}, std={w.std():.6f}, "
        f"min={pcts[0]:.6f}, p50={pcts[3]:.6f}, max={pcts[6]:.6f}, finite_ratio={finite_ratio:.4f}"
    )
    if wandb is None or wandb.run is None:
        return
    wandb.log({
        f"{name}/weights_n": w.size,
        f"{name}/weights_min": pcts[0],
        f"{name}/weights_p01": pcts[1],
        f"{name}/weights_p05": pcts[2],
        f"{name}/weights_p50": pcts[3],
        f"{name}/weights_p95": pcts[4],
        f"{name}/weights_p99": pcts[5],
        f"{name}/weights_max": pcts[6],
        f"{name}/weights_mean": w.mean(),
        f"{name}/weights_std": w.std(),
        f"{name}/weights_finite_ratio": finite_ratio,
    })


def log_unique_values(name: str, values: np.ndarray, limit: int = 25) -> None:
    """Print the top-`limit` most frequent unique values in `values`."""
    values = np.asarray(values).reshape(-1)
    unique, counts = np.unique(values, return_counts=True)
    order = np.argsort(-counts)[:limit]
    pairs = [(float(unique[i]), int(counts[i])) for i in order]
    logger.info(f"[{name}] top-{limit} unique values (value: count): {pairs}")


def log_propensity_diagnostics(
    tracker: Optional["Tracker"],
    propensity_bundle: Optional["PropensityBundle"],
    debug: bool,
    diagnose_true_propensity: bool,
    use_true_propensity: bool,
) -> None:
    if propensity_bundle is None:
        return
    if debug:
        propensity_stats("Train", propensity_bundle.train)
        propensity_stats("Validation", propensity_bundle.val)
        
    if tracker is not None:
        tracker.log_metrics({
            "propensity/train_mean": float(np.mean(propensity_bundle.train)),
            "propensity/val_mean": float(np.mean(propensity_bundle.val)),
            "propensity/train_min": float(np.min(propensity_bundle.train)),
            "propensity/val_min": float(np.min(propensity_bundle.val)),
            "propensity/train_max": float(np.max(propensity_bundle.train)),
            "propensity/val_max": float(np.max(propensity_bundle.val)),
        }, commit=False)
        tracker.log_histogram("propensity/train_dist", np.asarray(propensity_bundle.train), commit=False)
        tracker.log_histogram("propensity/val_dist", np.asarray(propensity_bundle.val), commit=False)
    
    if diagnose_true_propensity and not use_true_propensity:
        if propensity_bundle.true_train is not None and propensity_bundle.true_val is not None:
            propensity_stats("Train (oracle)", propensity_bundle.true_train)
            propensity_stats("Validation (oracle)", propensity_bundle.true_val)
            propensity_compare(
                "Train",
                propensity_bundle.train,
                propensity_bundle.true_train,
            )
            propensity_compare(
                "Validation",
                propensity_bundle.val,
                propensity_bundle.true_val,
            )
            log_unique_values("Train propensity (pred)", propensity_bundle.train)
            log_unique_values("Train propensity (oracle)", propensity_bundle.true_train)
            log_unique_values("Validation propensity (pred)", propensity_bundle.val)
            log_unique_values("Validation propensity (oracle)", propensity_bundle.true_val)


def log_doc_weight_stats(
    tracker: Optional["Tracker"],
    debug: bool,
    train_doc_weights: np.ndarray,
    val_doc_weights: np.ndarray,
) -> None:
    if debug:
        weights_stats("Train (pred)", train_doc_weights)
        weights_stats("Validation (pred)", val_doc_weights)

    if tracker is not None:
        tracker.log_metrics({
            "weights/train_mean": float(np.mean(train_doc_weights)),
            "weights/val_mean": float(np.mean(val_doc_weights)),
            "weights/train_min": float(np.min(train_doc_weights)),
            "weights/val_min": float(np.min(val_doc_weights)),
            "weights/train_max": float(np.max(train_doc_weights)),
            "weights/val_max": float(np.max(val_doc_weights)),
        }, commit=False)
        tracker.log_histogram("weights/train_dist", np.asarray(train_doc_weights), commit=False)
        tracker.log_histogram("weights/val_dist", np.asarray(val_doc_weights), commit=False)


def log_doc_weight_true_stats(
    tracker: Optional["Tracker"],
    debug: bool,
    train_doc_weights_true: np.ndarray | None,
    val_doc_weights_true: np.ndarray | None,
) -> None:
    if debug and train_doc_weights_true is not None and val_doc_weights_true is not None:
        weights_stats("Train (oracle)", train_doc_weights_true)
        weights_stats("Validation (oracle)", val_doc_weights_true)

    if tracker is not None and train_doc_weights_true is not None and val_doc_weights_true is not None:
        tracker.log_metrics({
            "weights_oracle/train_mean": float(np.mean(train_doc_weights_true)),
            "weights_oracle/val_mean": float(np.mean(val_doc_weights_true)),
            "weights_oracle/train_min": float(np.min(train_doc_weights_true)),
            "weights_oracle/val_min": float(np.min(val_doc_weights_true)),
            "weights_oracle/train_max": float(np.max(train_doc_weights_true)),
            "weights_oracle/val_max": float(np.max(val_doc_weights_true)),
        }, commit=False)
        tracker.log_histogram("weights_oracle/train_dist", np.asarray(train_doc_weights_true), commit=False)
        tracker.log_histogram("weights_oracle/val_dist", np.asarray(val_doc_weights_true), commit=False)


def log_missing_oracle_propensity(
    debug: bool,
    model_choice: str,
    propensity_bundle: Optional["PropensityBundle"],
) -> None:
    if (
        debug
        and model_choice == "ips"
        and (propensity_bundle is None or propensity_bundle.true_train is None)
    ):
        if propensity_bundle is not None:
            propensity_stats("Train (pred)", propensity_bundle.train)
            propensity_stats("Validation (pred)", propensity_bundle.val)
            log_unique_values("Train propensity (pred)", propensity_bundle.train)
            log_unique_values("Validation propensity (pred)", propensity_bundle.val)


__all__ = [
    "log_aggregated_dataset_stats",
    "log_aggregated_sizes",
    "log_click_stats",
    "log_raw_click_dataset_stats",
    "log_doc_weight_stats",
    "log_doc_weight_true_stats",
    "log_missing_oracle_propensity",
    "log_propensity_diagnostics",
    "log_unique_values",
    "propensity_compare",
    "propensity_stats",
    "weights_stats",
]
