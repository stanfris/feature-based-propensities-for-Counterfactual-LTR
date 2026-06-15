from copy import deepcopy
from functools import partial
import json
from pathlib import Path
import time
from typing import Dict

import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
import pandas as pd
from flax import nnx
from flax.training.early_stopping import EarlyStopping
from optax._src.base import GradientTransformation
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

try:
    import wandb
except ImportError:
    wandb = None

from feature_based_propensities_for_ULTR.metrics import Metric, Average
from feature_based_propensities_for_ULTR.ranking import plackettluce as pl
from feature_based_propensities_for_ULTR.ranking.exposure_loss import exposure_objective

class Trainer:
    def __init__(
        self,
        optimizer: GradientTransformation,
        metrics: Dict[str, Metric] = None,
        click_metrics: Dict[str, Metric] = None,
        epochs: int = 50,
        patience: int = 2,  # Note that NNX patience is off by 1, so 2 means 3 epochs.
        min_delta: float = 1e-5,
        n_features: int = 100,
        use_wandb: bool = False,
        log_freq: int = 100,
    ):
        self.optimizer = optimizer
        self.metrics = metrics if metrics is not None else {}
        self.click_metrics = click_metrics if click_metrics is not None else {}
        self.epochs = epochs
        self.patience = patience
        self.min_delta = min_delta
        self.click_metrics["loss"] = Average("loss")
        self.n_features = n_features
        self.use_wandb = use_wandb
        self.log_freq = log_freq

    def train(
        self,
        model: nnx.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ):

        optim = self.optimizer
        optimizer = nnx.Optimizer(model, optim)

        click_metrics = nnx.MultiMetric(**deepcopy(self.click_metrics))
        early_stopping = EarlyStopping(patience=self.patience, min_delta=self.min_delta)
        best_state = nnx.state(model)

        global_step = 0
        for epoch in range(self.epochs):
            # Enable non-deterministic operations:
            model.train()

            for batch in tqdm(train_loader, desc=f"Train - Epoch: {epoch}"):
                should_log_step = (
                    self.use_wandb
                    and wandb is not None
                    and wandb.run is not None
                    and global_step % self.log_freq == 0
                )
                if should_log_step:
                    loss, output, grads = self._train_step_with_diagnostics(
                        model,
                        optimizer,
                        click_metrics,
                        batch,
                    )
                    log_dict = {"train/loss_step": float(loss)}

                    # Log output distributions
                    log_dict["train/click_scores"] = wandb.Histogram(np.asarray(output.click))
                    log_dict["train/relevance_scores"] = wandb.Histogram(np.asarray(output.relevance))
                    log_dict["train/examination_scores"] = wandb.Histogram(np.asarray(output.examination))

                    # Log gradient norms and dists
                    flat_grads, _ = jax.tree_util.tree_flatten(grads)
                    grad_norms = [np.linalg.norm(np.asarray(g)) for g in flat_grads]
                    log_dict["gradients/global_norm"] = np.linalg.norm(grad_norms)
                    for i, g in enumerate(flat_grads):
                        log_dict[f"gradients/layer_{i}_norm"] = grad_norms[i]
                        if i < 10:  # limit histograms
                            log_dict[f"gradients/layer_{i}_dist"] = wandb.Histogram(np.asarray(g))

                    # Model weights
                    params = nnx.state(model, nnx.Param)
                    flat_params, _ = jax.tree_util.tree_flatten(params)
                    for i, p in enumerate(flat_params):
                        if i < 10:
                            log_dict[f"weights/layer_{i}_dist"] = wandb.Histogram(np.asarray(p))

                    wandb.log(log_dict, step=global_step)
                else:
                    self._train_step(model, optimizer, click_metrics, batch)

                global_step += 1

            train_metrics = click_metrics.compute()
            click_metrics.reset()

            # Disable random operations, such as dropout:
            model.eval()

            for batch in tqdm(val_loader, desc=f"Val - Epoch: {epoch}"):
                self._eval_click_step(model, click_metrics, batch)

            val_metrics = click_metrics.compute()
            early_stopping = early_stopping.update(val_metrics["loss"])
            click_metrics.reset()

            print(
                f"Epoch {epoch} - "
                f"Train loss: {train_metrics['loss']:.8f}, "
                f"Val loss: {val_metrics['loss']:.8f}, "
                f"has improved: {early_stopping.has_improved}\n"
            )


            if early_stopping.has_improved:
                best_state = nnx.state(model)

            if early_stopping.should_stop:
                print("Stopping early, loading best model state")
                nnx.update(model, best_state)
                break

    def test_clicks(
        self,
        model: nnx.Module,
        test_loader: DataLoader,
        save_outputs: bool = False,
        output_path: str | None = None,
    ):
        click_metrics = nnx.MultiMetric(**deepcopy(self.click_metrics))
        model.eval()
        all_outputs = [] 
        
        for batch in tqdm(test_loader, desc="Test"):
            outputs = self._test_click_step(model, click_metrics, batch)
            if save_outputs:
                outputs_dict = {
                    "click": jnp.array(outputs.click).tolist(),
                    "relevance": jnp.array(outputs.relevance).tolist(),
                    "examination": jnp.array(outputs.examination).tolist(),
                }
                all_outputs.append(outputs_dict)

        test_metrics = click_metrics.compute()
        click_metrics.reset()
        print(f"Test: {jax.tree.map(float, test_metrics)}")

        if save_outputs:
            if output_path is not None:
                with open(output_path, "w") as f:
                    json.dump(all_outputs, f)
                print(f"Saved outputs to {output_path}")
            else:
                print("Outputs were collected but not saved (no path specified).")

            return pd.DataFrame(test_metrics, index=[0]), all_outputs

        return pd.DataFrame(test_metrics, index=[0])

    def test_relevance(
        self,
        model: nnx.Module,
        test_loader: DataLoader,
    ):
        metrics = nnx.MultiMetric(**deepcopy(self.metrics))
        model.eval()

        for batch in tqdm(test_loader, desc="Test"):
            self._test_relevance_step(model, metrics, batch)

        test_metrics = metrics.compute()
        metrics.reset()
        return pd.DataFrame(test_metrics, index=[0])

    def get_position_bias(self, model, positions: int, unique_list: list = None, bias_csv_name: str = "bias"):
        positions = jnp.arange(positions)
        examination = model.bias_tower({"positions": positions}).squeeze()
        df = pd.DataFrame({"position": positions, "examination": examination-examination[0]})
        df.to_csv(f"{bias_csv_name}.csv", index=False)
        print(f"Saved position bias to {bias_csv_name}.csv")

    def get_relevance_scores(self, model, features: int):
        if hasattr(model, "relevance_tower"):
            feature_vectors = jnp.eye(features)

            relevance = model.relevance_tower({"query_doc_features": feature_vectors}).squeeze()

            return pd.DataFrame(
                {
                    "feature": jnp.arange(features),
                    "relevance": relevance,
                }
            )
        else:
            return pd.DataFrame({})
        
    def save_model_params(self, model, ckpt_dir="checkpoint"):
        """
        Save model parameters, excluding RNG state.
        """
        # Split model into RNG state and other parameters
        _, _, other_state = nnx.split(model, nnx.RngState, ...)
        
        # Ensure checkpoint directory exists
        ckpt_path = Path(ckpt_dir).resolve()
        ckpt_path.mkdir(parents=True, exist_ok=True)
        
        # Save using Orbax
        ckptr = ocp.StandardCheckpointer()
        ckptr.save(ckpt_path, other_state, force=True)
        ckptr.wait_until_finished()

        print(f"Model parameters saved to {ckpt_path}")

    def test_logging_policy(self, test_loader: DataLoader):
        metrics = nnx.MultiMetric(**deepcopy(self.metrics))

        @nnx.jit()
        def metric_fn(metrics):
            metrics.update(
                relevance=1 / (1 + batch["positions"]),
                relevance_labels=batch["labels"],
                mask=batch["mask"],
            )

        for batch in tqdm(test_loader, desc="Test logging policy"):
            metric_fn(metrics)
        test_metrics = metrics.compute()
        print(f"Test logging policy: {jax.tree.map(float, test_metrics)}")
        return pd.DataFrame(test_metrics, index=[0])

    @partial(nnx.jit, static_argnums=(0))
    def _train_step(
        self,
        model: nnx.Module,
        optimizer: nnx.Optimizer,
        metrics: nnx.MultiMetric,
        batch,
    ):

        def loss_fn(model, batch):
            output = model(batch)
            loss = model.compute_loss(output, batch).mean()
            return loss, output
        
        grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
        (loss, output), grads = grad_fn(model, batch)

        metrics.update(
            loss=loss,
            click=output.click,
            click_labels=batch["clicks"],
            mask=batch["mask"],
        )

        optimizer.update(grads)
        return loss

    @partial(nnx.jit, static_argnums=(0))
    def _train_step_with_diagnostics(
        self,
        model: nnx.Module,
        optimizer: nnx.Optimizer,
        metrics: nnx.MultiMetric,
        batch,
    ):

        def loss_fn(model, batch):
            output = model(batch)
            loss = model.compute_loss(output, batch).mean()
            return loss, output

        grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
        (loss, output), grads = grad_fn(model, batch)

        metrics.update(
            loss=loss,
            click=output.click,
            click_labels=batch["clicks"],
            mask=batch["mask"],
        )

        optimizer.update(grads)
        return loss, output, grads

    @partial(nnx.jit, static_argnums=(0))
    def _eval_click_step(
        self,
        model: nnx.Module,
        click_metrics: nnx.MultiMetric,
        batch,
    ):
        output = model(batch)
        loss = model.compute_loss(output, batch)
        click_metrics.update(
            loss=loss,
            click=output.click,
            click_labels=batch["clicks"],
            mask=batch["mask"],
        )
        return loss

    @partial(nnx.jit, static_argnums=(0))
    def _test_click_step(
        self,
        model: nnx.Module,
        click_metrics: nnx.MultiMetric,
        batch,
    ):
        output = model(batch)
        loss = model.compute_loss(output, batch)
        click_metrics.update(
            loss=loss,
            click=output.click,
            click_labels=batch["clicks"],
            mask=batch["mask"],
        )
        return output

    @partial(nnx.jit, static_argnums=(0))
    def _test_relevance_step(
        self,
        model: nnx.Module,
        metrics: nnx.MultiMetric,
        batch,
    ):
        relevance = model.predict_relevance(batch)
        # cast to same shape as batch["labels"]
        relevance = jnp.broadcast_to(relevance, batch["labels"].shape)
        metrics.update(
            relevance=relevance,
            relevance_labels=batch["labels"],
            mask=batch["mask"],
        )


class PolicyTrainer:
    """
    IPS policy trainer using the Plackett-Luce gradient estimator, mirroring
    the training loop in `train_logging_policy.optimize_policy`.
    """

    def __init__(
        self,
        optimizer: GradientTransformation,
        n_grad_samples: int | str = "dynamic",
        n_eval_samples: int = 100,
        eval_max_queries: int | None = None,
        max_epochs: int = 500,
        early_stop_diff: float = 0.001,
        early_stop_per_epochs: int = 3,
        early_stop_min_epochs: int = 0,
        patience: int = 3,
        n_grad_samples_dynamic_fallback: int = 100,
        print_updates: bool = True,
        use_wandb: bool = True,
        log_freq: int = 100,
        eval_first_epoch: bool = False,
        debug_eval_train_split: bool = False,
    ):
        self.optimizer = optimizer
        self.n_grad_samples = n_grad_samples
        self.n_eval_samples = n_eval_samples
        self.eval_max_queries = (
            int(eval_max_queries) if eval_max_queries is not None and int(eval_max_queries) > 0 else None
        )
        self.max_epochs = max_epochs
        self.early_stop_diff = early_stop_diff
        self.early_stop_per_epochs = early_stop_per_epochs
        self.early_stop_min_epochs = max(0, int(early_stop_min_epochs))
        self.patience = max(1, int(patience))
        self.n_grad_samples_dynamic_fallback = max(1, int(n_grad_samples_dynamic_fallback))
        self.print_updates = print_updates
        self.use_wandb = use_wandb
        self.log_freq = log_freq
        self.eval_first_epoch = bool(eval_first_epoch)
        self.debug_eval_train_split = bool(debug_eval_train_split)

    def train(
        self,
        model: nnx.Module,
        data_train,
        train_doc_weights: np.ndarray,
        train_alpha: np.ndarray,
        data_vali,
        vali_doc_weights: np.ndarray,
        vali_alpha: np.ndarray,
    ):
        optimizer = nnx.Optimizer(model, self.optimizer)

        train_alpha = jnp.asarray(train_alpha)
        vali_alpha = jnp.asarray(vali_alpha)
        stacked_alphas, stacked_additions = self._stack_alphas(train_alpha, vali_alpha)

        eval_cache = self._prepare_eval_cache(data_vali)
        if self.print_updates:
            total_rows = int(np.asarray(data_vali.mask).shape[0])
            active_rows = int(eval_cache[0].shape[0])
            self._log(
                f"Evaluation cache (validation): active_queries={active_rows}/{total_rows}"
            )
        baseline_eval_start = time.time()
        metrics = self._evaluate_policy(
            model,
            data_vali,
            stacked_alphas,
            stacked_additions,
            vali_doc_weights,
            eval_cache=eval_cache,
        )
        metrics = jax.block_until_ready(metrics)
        if self.print_updates:
            self._log(
                "Epoch 0 baseline eval time (validation): %.2fs"
                % (time.time() - baseline_eval_start)
            )
        train_eval_cache = None
        train_metrics = None
        if self.debug_eval_train_split:
            train_eval_cache = self._prepare_eval_cache(data_train)
            if self.print_updates:
                total_rows = int(np.asarray(data_train.mask).shape[0])
                active_rows = int(train_eval_cache[0].shape[0])
                self._log(
                    f"Evaluation cache (train debug): active_queries={active_rows}/{total_rows}"
                )
            baseline_train_eval_start = time.time()
            train_metrics = self._evaluate_policy(
                model,
                data_train,
                stacked_alphas,
                stacked_additions,
                train_doc_weights,
                eval_cache=train_eval_cache,
            )
            train_metrics = jax.block_until_ready(train_metrics)
            if self.print_updates:
                self._log(
                    "Epoch 0 baseline eval time (train debug): %.2fs"
                    % (time.time() - baseline_train_eval_start)
                )
        self._log_initial_metrics(metrics, train_metrics=train_metrics)

        first_metric_value = float(metrics[1])
        last_metric_value = float(metrics[1])
        best_metric_value = float(metrics[1])
        best_state = nnx.state(model)
        best_epoch = 0

        train_doc_weights = jnp.asarray(train_doc_weights)
        vali_doc_weights = jnp.asarray(vali_doc_weights)
        train_q_feat, train_q_doc_weights, train_q_mask = self._prepare_query_tensors(
            data_train, train_doc_weights
        )
        train_rank_weights = self._prepare_rank_weights(train_q_mask, train_alpha)

        qids = self._select_training_queries(
            np.asarray(train_q_doc_weights),
            np.asarray(train_q_mask),
        )
        n_queries = int(qids.shape[0])
        grad_sampler = self._make_grad_sampler(n_queries)

        eval_interval = min(self.early_stop_per_epochs, self.max_epochs)
        start_time = time.time()
        global_step = 0
        no_improve_count = 0

        for epoch in range(self.max_epochs):
            if self.print_updates:
                self._log(f"Epoch {epoch + 1}/{self.max_epochs} - start")
            epoch_skipped_updates = 0
            epoch_loss_sum = 0.0
            epoch_update_count = 0
            use_jitted_epoch = False
            fixed_n_grad_samples = None
            if n_queries > 0:
                sample_0 = int(grad_sampler(global_step))
                sample_1 = int(grad_sampler(global_step + (1 if n_queries > 1 else 0)))
                if sample_0 == sample_1:
                    use_jitted_epoch = True
                    fixed_n_grad_samples = sample_0
            qids = np.random.permutation(qids)
            if use_jitted_epoch and fixed_n_grad_samples is not None:
                if self.print_updates:
                    self._log(f"Epoch {epoch + 1} - jitted train over {n_queries} queries")
                qids_jax = jnp.asarray(qids, dtype=jnp.int32)
                model, optimizer, skipped_updates, loss_sum, update_count = self._train_epoch(
                    model,
                    optimizer,
                    train_q_feat,
                    train_q_doc_weights,
                    train_q_mask,
                    train_rank_weights,
                    qids_jax,
                    fixed_n_grad_samples,
                )
                self._block_on_model_params(model)
                epoch_skipped_updates += int(skipped_updates)
                epoch_loss_sum += float(jax.device_get(loss_sum))
                epoch_update_count += int(jax.device_get(update_count))
                global_step += n_queries
            else:
                for qid in tqdm(qids, desc=f"IPS Train - Epoch {epoch + 1}", disable=not self.print_updates):
                    qid = int(qid)
                    q_doc_weights = train_q_doc_weights[qid]
                    q_feat = train_q_feat[qid]
                    q_mask = train_q_mask[qid]
                    q_rank_weights = train_rank_weights[qid]
                    n_grad_samples = grad_sampler(global_step)
                    loss, scores, grads, did_update = self._train_step(
                        model,
                        optimizer,
                        q_feat,
                        q_doc_weights,
                        q_mask,
                        q_rank_weights,
                        n_grad_samples,
                    )
                    if not bool(did_update):
                        epoch_skipped_updates += 1
                        global_step += 1
                        continue

                    epoch_loss_sum += float(loss)
                    epoch_update_count += 1
                    
                    if (
                        self.use_wandb
                        and wandb is not None
                        and wandb.run is not None
                        and global_step % self.log_freq == 0
                    ):
                        log_dict = {"train_policy/loss_step": float(loss)}
                        log_dict["train_policy/scores"] = wandb.Histogram(np.asarray(scores))
                        
                        flat_grads, _ = jax.tree_util.tree_flatten(grads)
                        grad_norms = [np.linalg.norm(np.asarray(g)) for g in flat_grads]
                        log_dict["gradients_policy/global_norm"] = np.linalg.norm(grad_norms)
                        for i, g in enumerate(flat_grads):
                            log_dict[f"gradients_policy/layer_{i}_norm"] = grad_norms[i]
                            if i < 10:
                                log_dict[f"gradients_policy/layer_{i}_dist"] = wandb.Histogram(np.asarray(g))
                                
                        params = nnx.state(model, nnx.Param)
                        flat_params, _ = jax.tree_util.tree_flatten(params)
                        for i, p in enumerate(flat_params):
                            if i < 10:
                                log_dict[f"weights_policy/layer_{i}_dist"] = wandb.Histogram(np.asarray(p))
                        wandb.log(log_dict, step=global_step)
                    
                    global_step += 1
            epoch_train_loss = epoch_loss_sum / max(epoch_update_count, 1)
            if self.print_updates:
                self._log(
                    "Epoch %d - train loss: %.8f (updates=%d, skipped=%d)"
                    % (
                        epoch + 1,
                        epoch_train_loss,
                        epoch_update_count,
                        epoch_skipped_updates,
                    )
                )
            if self.print_updates and epoch_skipped_updates > 0:
                self._log(
                    "Epoch %d - skipped %d non-finite updates (safety guard)"
                    % (epoch + 1, epoch_skipped_updates)
                )

            should_eval = ((epoch + 1) % eval_interval == 0) or (
                self.eval_first_epoch and epoch == 0
            )
            if not should_eval:
                if self.print_updates:
                    self._log(f"Epoch {epoch + 1} - skipped eval (interval={eval_interval})")
                continue
            if self.print_updates and self.eval_first_epoch and epoch == 0 and eval_interval != 1:
                self._log(
                    f"Epoch {epoch + 1} - forced eval (eval_first_epoch=True, interval={eval_interval})"
                )

            eval_start = time.time()
            metrics = self._evaluate_policy(
                model,
                data_vali,
                stacked_alphas,
                stacked_additions,
                vali_doc_weights,
                eval_cache=eval_cache,
            )
            metrics = jax.block_until_ready(metrics)
            val_eval_time = time.time() - eval_start
            train_metrics = None
            train_eval_time = None
            if self.debug_eval_train_split and train_eval_cache is not None:
                eval_start = time.time()
                train_metrics = self._evaluate_policy(
                    model,
                    data_train,
                    stacked_alphas,
                    stacked_additions,
                    train_doc_weights,
                    eval_cache=train_eval_cache,
                )
                train_metrics = jax.block_until_ready(train_metrics)
                train_eval_time = time.time() - eval_start
            if self.print_updates:
                msg = f"Epoch {epoch + 1} - eval time: val={val_eval_time:.2f}s"
                if train_eval_time is not None:
                    msg += f", train={train_eval_time:.2f}s"
                self._log(msg)
            abs_improvement = float(metrics[1] - last_metric_value)
            self._log_epoch_metrics(
                epoch=epoch + 1,
                metrics=metrics,
                train_metrics=train_metrics,
                abs_improvement=abs_improvement,
                first_metric_value=first_metric_value,
                last_metric_value=last_metric_value,
                start_time=start_time,
                n_queries=n_queries,
                global_step=global_step,
            )
            if (
                self.print_updates
                and epoch <= eval_interval
                and first_metric_value > 0.0
                and float(metrics[1]) < 0.2 * first_metric_value
            ):
                self._log(
                    "Warning: early validation metric dropped by >80% from epoch-0 baseline. "
                    "This can happen with sparse IPS weights and/or unstable first updates; "
                    "inspect epoch-1 forced eval and consider lower learning rate or more grad samples."
                )

            last_metric_value = float(metrics[1])
            if best_metric_value < metrics[1]:
                best_state = nnx.state(model)
                best_metric_value = float(metrics[1])
                best_epoch = epoch + 1
            if (epoch + 1) >= self.early_stop_min_epochs and abs_improvement < self.early_stop_diff:
                no_improve_count += 1
                if self.print_updates:
                    self._log(
                        f"Epoch {epoch + 1}: no improvement "
                        f"(abs_improvement={abs_improvement:.6f} < {self.early_stop_diff:.6f}), "
                        f"patience {no_improve_count}/{self.patience}"
                    )
                if no_improve_count >= self.patience:
                    if self.print_updates:
                        self._log(
                            f"Early stop triggered at epoch {epoch + 1} "
                            f"(patience={self.patience} exhausted)"
                        )
                    break
            else:
                no_improve_count = 0

        nnx.update(model, best_state)
        if self.print_updates:
            self._log(
                "Training finished: restored best checkpoint from epoch %d "
                "(best alpha_val_metric=%0.6f)"
                % (best_epoch, best_metric_value)
            )
        return model, last_metric_value

    @staticmethod
    def _stack_alphas(train_alpha: jnp.ndarray, vali_alpha: jnp.ndarray):
        stacked_alphas = jnp.stack([train_alpha, vali_alpha], axis=-1)
        stacked_additions = jnp.zeros_like(stacked_alphas)
        return stacked_alphas, stacked_additions

    def _evaluate_policy(
        self,
        model: nnx.Module,
        data_split,
        stacked_alphas: jnp.ndarray,
        stacked_additions: jnp.ndarray,
        doc_weights: jnp.ndarray,
        *,
        eval_cache: tuple[jnp.ndarray, jnp.ndarray] | None = None,
    ):
        policy_scores = model.score_feature_matrix(jnp.asarray(data_split.feature_matrix))
        if eval_cache is None:
            return pl.datasplit_metrics(
                data_split,
                policy_scores,
                stacked_alphas,
                stacked_additions,
                doc_weights,
                query_norm_factors=None,
                n_samples=self.n_eval_samples,
            )
        doc_id_map, mask = eval_cache
        return self._datasplit_metrics_cached(
            policy_scores,
            stacked_alphas,
            stacked_additions,
            doc_weights,
            doc_id_map,
            mask,
            n_samples=self.n_eval_samples,
        )

    def _prepare_eval_cache(self, data_split) -> tuple[jnp.ndarray, jnp.ndarray]:
        doc_id_map_np = np.asarray(data_split.doc_id_map)
        mask_np = np.asarray(data_split.mask, dtype=bool)
        active_rows = np.any(mask_np, axis=1)
        if not np.all(active_rows):
            doc_id_map_np = doc_id_map_np[active_rows]
            mask_np = mask_np[active_rows]
        if (
            self.eval_max_queries is not None
            and doc_id_map_np.shape[0] > self.eval_max_queries
        ):
            n_active = int(doc_id_map_np.shape[0])
            rng = np.random.default_rng(0)
            keep_idx = np.sort(
                rng.choice(n_active, size=self.eval_max_queries, replace=False)
            )
            doc_id_map_np = doc_id_map_np[keep_idx]
            mask_np = mask_np[keep_idx]
            if self.print_updates:
                self._log(
                    "Evaluation cache capped to %d/%d active queries "
                    "(deterministic sample; set ips.trainer.eval_max_queries to adjust)."
                    % (self.eval_max_queries, n_active)
                )
        return jnp.asarray(doc_id_map_np), jnp.asarray(mask_np)

    @staticmethod
    def _datasplit_metrics_cached(
        policy_scores: jnp.ndarray,
        weight_per_rank: jnp.ndarray,
        addition_per_rank: jnp.ndarray,
        weight_per_doc: jnp.ndarray,
        doc_id_map: jnp.ndarray,
        mask: jnp.ndarray,
        *,
        n_samples: int,
    ) -> jnp.ndarray:
        policy_scores = jnp.asarray(policy_scores).reshape(-1)
        weight_per_doc = jnp.asarray(weight_per_doc).reshape(-1)
        weight_per_rank = jnp.asarray(weight_per_rank)
        addition_per_rank = jnp.asarray(addition_per_rank)
        n_queries = int(doc_id_map.shape[0])
        n_metrics = int(weight_per_rank.shape[1])
        if n_queries == 0:
            return jnp.zeros((n_metrics,), dtype=weight_per_rank.dtype)
        dummy_norm = jnp.ones((n_queries, n_metrics), dtype=weight_per_rank.dtype)
        return pl._datasplit_metrics_jax(
            policy_scores,
            weight_per_rank,
            addition_per_rank,
            weight_per_doc,
            doc_id_map,
            mask,
            dummy_norm,
            n_samples=n_samples,
            use_norm=False,
        )

    @staticmethod
    def _prepare_query_tensors(data_split, doc_weights: jnp.ndarray):
        q_feat = jnp.asarray(data_split.query_doc_features)
        q_mask = jnp.asarray(data_split.mask).astype(bool)
        q_feat = jnp.where(q_mask[..., None], q_feat, 0.0)
        q_doc_weights = pl.pad_by_doc_id(
            doc_weights,
            jnp.asarray(data_split.doc_id_map),
            fill_value=0.0,
        )
        q_doc_weights = jnp.where(q_mask, q_doc_weights, 0.0)
        return q_feat, q_doc_weights, q_mask

    @staticmethod
    def _prepare_rank_weights(q_mask: jnp.ndarray, train_alpha: jnp.ndarray) -> jnp.ndarray:
        n_valid = jnp.sum(q_mask, axis=1, dtype=jnp.int32)
        ranks = jnp.arange(train_alpha.shape[0])[None, :]
        rank_mask = ranks < n_valid[:, None]
        return train_alpha[None, :] * rank_mask

    @staticmethod
    def _select_training_queries(
        train_q_doc_weights: np.ndarray,
        train_q_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        query_weights = np.abs(np.asarray(train_q_doc_weights))
        if query_weights.ndim != 2:
            raise ValueError(
                f"Expected 2D query-doc weight matrix, got shape={query_weights.shape}"
            )
        if train_q_mask is not None:
            query_weights = np.where(np.asarray(train_q_mask, dtype=bool), query_weights, 0.0)
        query_weights = np.sum(query_weights, axis=1)
        qids = np.flatnonzero(query_weights > 0)
        if qids.size == 0:
            raise ValueError("No queries with non-zero weights available for training.")
        return qids

    def _make_grad_sampler(self, n_queries: int):
        if self.n_grad_samples == "dynamic":
            fixed_samples = self.n_grad_samples_dynamic_fallback

            def sampler(step_idx: int) -> int:
                return fixed_samples

            return sampler

        fixed_samples = int(self.n_grad_samples)

        def sampler(step_idx: int) -> int:
            return fixed_samples

        return sampler

    @partial(nnx.jit, static_argnums=(0, 7))
    def _train_step(
        self,
        model: nnx.Module,
        optimizer: nnx.Optimizer,
        q_feat: jnp.ndarray,
        q_doc_weights: jnp.ndarray,
        q_mask: jnp.ndarray,
        q_rank_weights: jnp.ndarray,
        n_grad_samples: int,
    ):
        def loss_fn(model):
            scores = model.score_query(q_feat)
            scores = jnp.where(q_mask, scores, -1e9)
            return model.compute_loss(
                scores, q_doc_weights, q_rank_weights, n_grad_samples
            ), scores

        (loss, scores), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)
        loss_is_finite = jnp.isfinite(loss) & jnp.all(jnp.isfinite(scores))
        grads_are_finite = self._tree_all_finite(grads)
        should_update = loss_is_finite & grads_are_finite
        safe_grads = jax.tree_util.tree_map(
            lambda g: jnp.where(should_update, g, jnp.zeros_like(g)),
            grads,
        )
        optimizer.update(safe_grads)
        return loss, scores, grads, should_update

    def _log(self, message: str) -> None:
        if self.print_updates:
            print(message)

    @staticmethod
    def _block_on_model_params(model: nnx.Module) -> None:
        params = nnx.state(model, nnx.Param)
        leaves, _ = jax.tree_util.tree_flatten(params)
        if leaves:
            jax.block_until_ready(leaves[0])

    @staticmethod
    def _tree_all_finite(tree) -> jax.Array:
        leaves = jax.tree_util.tree_leaves(tree)
        if not leaves:
            return jnp.array(True)
        finite_flags = [jnp.all(jnp.isfinite(leaf)) for leaf in leaves]
        return jnp.all(jnp.stack(finite_flags))

    @partial(nnx.jit, static_argnums=(0, 8))
    def _train_epoch(
        self,
        model: nnx.Module,
        optimizer: nnx.Optimizer,
        q_feat: jnp.ndarray,
        q_doc_weights: jnp.ndarray,
        q_mask: jnp.ndarray,
        q_rank_weights: jnp.ndarray,
        qids: jnp.ndarray,
        n_grad_samples: int,
    ):
        @partial(nnx.scan, in_axes=(nnx.Carry, 0), out_axes=nnx.Carry)
        def body(carry, qid):
            model, optimizer, skipped_updates, loss_sum, update_count = carry
            q_feat_i = q_feat[qid]
            q_doc_weights_i = q_doc_weights[qid]
            q_mask_i = q_mask[qid]
            q_rank_weights_i = q_rank_weights[qid]

            def loss_fn(model):
                scores = model.score_query(q_feat_i)
                scores = jnp.where(q_mask_i, scores, -1e9)
                return model.compute_loss(
                    scores, q_doc_weights_i, q_rank_weights_i, n_grad_samples
                )

            loss, grads = nnx.value_and_grad(loss_fn)(model)
            loss_is_finite = jnp.isfinite(loss)
            grads_are_finite = self._tree_all_finite(grads)
            should_update = loss_is_finite & grads_are_finite
            safe_grads = jax.tree_util.tree_map(
                lambda g: jnp.where(should_update, g, jnp.zeros_like(g)),
                grads,
            )
            optimizer.update(safe_grads)
            skipped_updates = skipped_updates + jnp.where(should_update, 0, 1)
            loss_sum = loss_sum + jnp.where(
                should_update,
                jnp.asarray(loss, dtype=jnp.float32),
                jnp.asarray(0.0, dtype=jnp.float32),
            )
            update_count = update_count + jnp.where(should_update, 1, 0)
            return (model, optimizer, skipped_updates, loss_sum, update_count)

        model, optimizer, skipped_updates, loss_sum, update_count = body(
            (
                model,
                optimizer,
                jnp.array(0, dtype=jnp.int32),
                jnp.array(0.0, dtype=jnp.float32),
                jnp.array(0, dtype=jnp.int32),
            ),
            qids,
        )
        return model, optimizer, skipped_updates, loss_sum, update_count

    def _log_initial_metrics(
        self,
        metrics: np.ndarray,
        *,
        train_metrics: np.ndarray | None = None,
    ) -> None:
        self._log(
            "epoch 0 baseline (validation split): alpha_train_metric=%0.04f alpha_val_metric=%0.04f"
            % (metrics[0], metrics[1])
        )
        if train_metrics is not None:
            self._log(
                "epoch 0 baseline (train split): alpha_train_metric=%0.04f alpha_val_metric=%0.04f"
                % (train_metrics[0], train_metrics[1])
            )

    def _log_epoch_metrics(
        self,
        epoch: int,
        metrics: np.ndarray,
        train_metrics: np.ndarray | None,
        abs_improvement: float,
        first_metric_value: float,
        last_metric_value: float,
        start_time: float,
        n_queries: int,
        global_step: int,
    ) -> None:
        if not self.print_updates:
            return
        improvement = metrics[1] / last_metric_value - 1.0 if last_metric_value != 0 else 0.0
        total_improvement = metrics[1] / first_metric_value - 1.0 if first_metric_value != 0 else 0.0
        average_time = (time.time() - start_time) / max(global_step, 1) * n_queries
        line = (
            "epoch %d: "
            "val(alpha_train_metric=%0.04f alpha_val_metric=%0.04f) "
            "epoch-time %0.04f "
            "abs-improvement %.8f "
            "improvement %.8f "
            "total-improvement %.8f "
            % (
                epoch,
                metrics[0],
                metrics[1],
                average_time,
                abs_improvement,
                improvement,
                total_improvement,
            )
        )
        if train_metrics is not None:
            line += "train(alpha_train_metric=%0.04f alpha_val_metric=%0.04f) " % (
                train_metrics[0],
                train_metrics[1],
            )
        self._log(line)

class PropensityTrainer:
    def __init__(
        self,
        optimizer: GradientTransformation = None,
        epochs: int = 50,
        patience: int = 3,
        learning_rate: float = 5e-3,
        min_delta: float = 1e-5,
    ):
        self.optimizer = optimizer
        self.epochs = epochs
        self.patience = patience
        self.learning_rate = float(learning_rate)
        self.min_delta = float(min_delta)

    def train(
        self,
        model: nnx.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ):
        # Setup optimizer if not provided
        optimizer = nnx.Optimizer(model, self.optimizer or optax.adamw(self.learning_rate))

        early_stopping = EarlyStopping(patience=self.patience, min_delta=self.min_delta)
        best_state = nnx.state(model)
        best_val_loss = float("inf")

        @nnx.jit
        def train_step(model, optimizer, batch):
            def loss_fn(model):
                output = model(batch)
                return model.compute_loss(output, batch)
            loss, grads = nnx.value_and_grad(loss_fn)(model)
            optimizer.update(grads)
            return loss

        @nnx.jit
        def val_step(model, batch):
            output = model(batch)
            return model.compute_loss(output, batch)

        epoch_start = time.time()
        for epoch in range(self.epochs):
            t0 = time.time()

            # ----- Train -----
            model.train()
            train_loss_acc = jnp.zeros(())
            train_loss_count = 0
            for batch in train_loader:
                loss = train_step(model, optimizer, batch)
                train_loss_acc = train_loss_acc + loss  # stays on device
                train_loss_count += 1
            # One device→host sync per epoch (not per batch)
            train_loss = float(train_loss_acc / max(train_loss_count, 1))

            # ----- Validation -----
            model.eval()
            val_loss_acc = jnp.zeros(())
            val_loss_count = 0
            last_val_batch = None
            for batch in val_loader:
                loss = val_step(model, batch)
                val_loss_acc = val_loss_acc + loss  # stays on device
                val_loss_count += 1
                last_val_batch = batch
            val_loss = float(val_loss_acc / max(val_loss_count, 1))

            early_stopping = early_stopping.update(val_loss)

            epoch_secs = time.time() - t0
            print(
                f"[PropensityMLP] Epoch {epoch + 1}/{self.epochs} "
                f"({epoch_secs:.1f}s) — "
                f"train: {train_loss:.6f}  val: {val_loss:.6f}  "
                f"improved: {early_stopping.has_improved}"
            )

            # Debug: print targets vs. masked mean predictions (regression models only)
            if last_val_batch is not None and hasattr(model, "alpha"):
                targets = np.asarray(model.alpha.value)
                preds = np.asarray(model(last_val_batch))  # (B, P)
                mask = np.asarray(last_val_batch.get("mask", np.ones_like(preds, dtype=bool))).astype(bool)
                valid_counts = mask.sum(axis=0)
                safe_denom = np.maximum(valid_counts, 1)
                masked_sum = np.where(mask, preds, 0.0).sum(axis=0)
                mean_preds = masked_sum / safe_denom
                np.set_printoptions(precision=4, suppress=True)
                print(f"  [debug] targets : {targets}")
                print(f"  [debug] mean_out_masked: {mean_preds}")
                print(f"  [debug] valid_count: {valid_counts}")

            if early_stopping.has_improved:
                best_val_loss = val_loss
                best_state = nnx.state(model)

            if early_stopping.should_stop:
                print("Early stopping triggered, restoring best model parameters.")
                nnx.update(model, best_state)
                break

        # Restore best model state at the end
        nnx.update(model, best_state)
        return model, best_val_loss

    def run_propensity_inference(
        self,
        model: nnx.Module,
        data_loader: DataLoader,
        output_path: str = "propensity_test_outputs.npz",
    ):
        """
        Run inference of the propensity model over a DataLoader and save outputs to an NPZ file.
        PropensityMLP(batch) -> [batch_size] (after squeeze), so the final array is [N,].
        """
        model.eval()
        all_outputs = []

        @nnx.jit
        def forward_step(model, batch):
            return model.compute_output(batch)  # returns 1D array [batch_size]

        for batch in data_loader:
            preds = forward_step(model, batch)
            preds_np = jnp.asarray(jax.device_get(preds))  # move to host NumPy
            all_outputs.append(preds_np)

        if not all_outputs:
            raise ValueError("No batches found in data_loader; cannot run inference.")

        outputs = jnp.concatenate(all_outputs, axis=0)  # shape [N,]

        jnp.savez(output_path, propensity=outputs)
        print(
            f"Saved propensity outputs for {outputs.shape[0]} examples "
            f"to {output_path}"
        )

        return outputs
    
    def save_model_params(self, model, ckpt_dir="checkpoint"):
        """
        Save model parameters, excluding RNG state.
        """
        # Split model into RNG state and other parameters
        _, _, other_state = nnx.split(model, nnx.RngState, ...)
        
        # Ensure checkpoint directory exists
        ckpt_path = Path(ckpt_dir).resolve()
        ckpt_path.mkdir(parents=True, exist_ok=True)
        
        # Save using Orbax
        ckptr = ocp.StandardCheckpointer()
        ckptr.save(ckpt_path, other_state, force=True)
        ckptr.wait_until_finished()

        print(f"Model parameters saved to {ckpt_path}")


class ExposurePolicyTrainer:
    """
    Policy trainer using the exposure-based Monte Carlo policy gradient.

    Replaces the Plackett-Luce gradient estimator in `PolicyTrainer` with the
    `exposure_objective` from `exposure_loss.py`.  The objective is:

        L = sum_d  E_pi[exposure_d] * reward_d

    where `exposure_d` is the expected exposure of document d under the current
    policy (estimated via Gumbel sampling) and `reward_d` is a per-document
    weight such as an IPS/DR/DM-corrected click signal.

    The gradient is computed via a REINFORCE-style policy gradient with a
    leave-one-out variance-reduction baseline (see exposure_loss.py for details).

    The training loop, evaluation, and early-stopping logic are identical to
    `PolicyTrainer`.  The only structural differences are:
    - A JAX PRNGKey is maintained in the trainer and split each training step.
    - The jitted epoch scan carries `(model, optimizer, rng_key)` as its carry.
    - `n_samples` (MC samples for the exposure estimator) replaces `n_grad_samples`.
    """

    def __init__(
        self,
        optimizer: GradientTransformation,
        n_samples: int = 100,
        n_eval_samples: int = 100,
        eval_max_queries: int | None = None,
        max_epochs: int = 500,
        early_stop_diff: float = 0.001,
        early_stop_per_epochs: int = 3,
        early_stop_min_epochs: int = 0,
        patience: int = 3,
        print_updates: bool = True,
        use_wandb: bool = True,
        log_freq: int = 100,
        rng_seed: int = 42,
        eval_first_epoch: bool = False,
        debug_eval_train_split: bool = False,
    ):
        self.optimizer = optimizer
        self.n_samples = int(n_samples)
        self.n_eval_samples = int(n_eval_samples)
        self.eval_max_queries = (
            int(eval_max_queries) if eval_max_queries is not None and int(eval_max_queries) > 0 else None
        )
        self.max_epochs = max_epochs
        self.early_stop_diff = early_stop_diff
        self.early_stop_per_epochs = early_stop_per_epochs
        self.early_stop_min_epochs = max(0, int(early_stop_min_epochs))
        self.patience = max(1, int(patience))
        self.print_updates = print_updates
        self.use_wandb = use_wandb
        self.log_freq = log_freq
        self.rng_seed = rng_seed
        self.eval_first_epoch = bool(eval_first_epoch)
        self.debug_eval_train_split = bool(debug_eval_train_split)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def train(
        self,
        model: nnx.Module,
        data_train,
        train_doc_weights: np.ndarray,
        train_alpha: np.ndarray,
        data_vali,
        vali_doc_weights: np.ndarray,
        vali_alpha: np.ndarray,
    ):
        """
        Train `model` using the exposure policy gradient.

        Parameters mirror `PolicyTrainer.train` exactly so that the two
        trainers are interchangeable in experiment configs.

        Args:
            model: A `PolicyModelBase` subclass with `score_query` defined.
            data_train: Training data split with `feature_matrix`, `doclist_ranges`, etc.
            train_doc_weights: Per-document reward weights (IPS / DR / DM), shape [N_train_docs].
            train_alpha: Per-rank exposure weights (e.g. 1/log2(k+1)), shape [K].
            data_vali: Validation data split.
            vali_doc_weights: Per-document reward weights for validation.
            vali_alpha: Per-rank exposure weights for validation.

        Returns:
            (model, best_metric_value) — model is updated in-place to best checkpoint.
        """
        optimizer = nnx.Optimizer(model, self.optimizer)
        rng_key = jax.random.PRNGKey(self.rng_seed)

        train_alpha = jnp.asarray(train_alpha)
        vali_alpha = jnp.asarray(vali_alpha)
        stacked_alphas, stacked_additions = self._stack_alphas(train_alpha, vali_alpha)

        eval_cache = self._prepare_eval_cache(data_vali)
        if self.print_updates:
            total_rows = int(np.asarray(data_vali.mask).shape[0])
            active_rows = int(eval_cache[0].shape[0])
            self._log(
                f"Evaluation cache (validation): active_queries={active_rows}/{total_rows}"
            )
        baseline_eval_start = time.time()
        metrics = self._evaluate_policy(
            model,
            data_vali,
            stacked_alphas,
            stacked_additions,
            vali_doc_weights,
            eval_cache=eval_cache,
        )
        metrics = jax.block_until_ready(metrics)
        if self.print_updates:
            self._log(
                "Epoch 0 baseline eval time (validation): %.2fs"
                % (time.time() - baseline_eval_start)
            )
        train_eval_cache = None
        train_metrics = None
        if self.debug_eval_train_split:
            train_eval_cache = self._prepare_eval_cache(data_train)
            if self.print_updates:
                total_rows = int(np.asarray(data_train.mask).shape[0])
                active_rows = int(train_eval_cache[0].shape[0])
                self._log(
                    f"Evaluation cache (train debug): active_queries={active_rows}/{total_rows}"
                )
            baseline_train_eval_start = time.time()
            train_metrics = self._evaluate_policy(
                model,
                data_train,
                stacked_alphas,
                stacked_additions,
                train_doc_weights,
                eval_cache=train_eval_cache,
            )
            train_metrics = jax.block_until_ready(train_metrics)
            if self.print_updates:
                self._log(
                    "Epoch 0 baseline eval time (train debug): %.2fs"
                    % (time.time() - baseline_train_eval_start)
                )
        self._log_initial_metrics(metrics, train_metrics=train_metrics)

        first_metric_value = float(metrics[1])
        last_metric_value = float(metrics[1])
        best_metric_value = float(metrics[1])
        best_state = nnx.state(model)
        best_epoch = 0

        train_doc_weights = jnp.asarray(train_doc_weights)
        train_q_feat, train_q_doc_weights, train_q_mask = self._prepare_query_tensors(
            data_train, train_doc_weights
        )
        train_rank_weights = self._prepare_rank_weights(train_q_mask, train_alpha)

        qids = self._select_training_queries(
            np.asarray(train_q_doc_weights),
            np.asarray(train_q_mask),
        )
        n_queries = int(qids.shape[0])

        eval_interval = min(self.early_stop_per_epochs, self.max_epochs)
        start_time = time.time()
        global_step = 0
        no_improve_count = 0

        for epoch in range(self.max_epochs):
            if self.print_updates:
                self._log(f"Epoch {epoch + 1}/{self.max_epochs} - start")

            qids = np.random.permutation(qids)
            epoch_loss_sum = 0.0
            epoch_update_count = 0

            # Attempt the fully-jitted scan path when n_samples is constant
            # (it always is here, but we keep the guard for consistency with
            # PolicyTrainer so that the two are easy to compare).
            use_jitted_epoch = True  # n_samples is always a fixed int
            if use_jitted_epoch:
                if self.print_updates:
                    self._log(f"Epoch {epoch + 1} - jitted train over {n_queries} queries")
                qids_jax = jnp.asarray(qids, dtype=jnp.int32)
                rng_key, epoch_key = jax.random.split(rng_key)
                model, optimizer, rng_key_after, loss_sum, update_count = self._train_epoch(
                    model,
                    optimizer,
                    train_q_feat,
                    train_q_doc_weights,
                    train_q_mask,
                    train_rank_weights,
                    qids_jax,
                    epoch_key,
                    self.n_samples,
                )
                self._block_on_model_params(model)
                # `rng_key_after` is the key state after the last query in the
                # scan; carry it forward so RNG is deterministic across epochs.
                rng_key = rng_key_after
                epoch_loss_sum += float(jax.device_get(loss_sum))
                epoch_update_count += int(jax.device_get(update_count))
                global_step += n_queries
            else:
                # Fallback: Python loop (kept for debugging / variable n_samples)
                for qid in tqdm(
                    qids,
                    desc=f"Exposure Train - Epoch {epoch + 1}",
                    disable=not self.print_updates,
                ):
                    qid = int(qid)
                    q_doc_weights = train_q_doc_weights[qid]
                    q_feat = train_q_feat[qid]
                    q_mask = train_q_mask[qid]
                    q_rank_weights = train_rank_weights[qid]

                    rng_key, subkey = jax.random.split(rng_key)
                    loss, scores, grads = self._train_step(
                        model,
                        optimizer,
                        q_feat,
                        q_doc_weights,
                        q_mask,
                        q_rank_weights,
                        subkey,
                        self.n_samples,
                    )
                    epoch_loss_sum += float(loss)
                    epoch_update_count += 1

                    if (
                        self.use_wandb
                        and wandb is not None
                        and wandb.run is not None
                        and global_step % self.log_freq == 0
                    ):
                        self._wandb_log_step(model, loss, scores, grads, global_step)

                    global_step += 1
            epoch_train_loss = epoch_loss_sum / max(epoch_update_count, 1)
            if self.print_updates:
                self._log(
                    "Epoch %d - train loss: %.8f (updates=%d)"
                    % (
                        epoch + 1,
                        epoch_train_loss,
                        epoch_update_count,
                    )
                )

            should_eval = ((epoch + 1) % eval_interval == 0) or (
                self.eval_first_epoch and epoch == 0
            )
            if not should_eval:
                if self.print_updates:
                    self._log(f"Epoch {epoch + 1} - skipped eval (interval={eval_interval})")
                continue
            if self.print_updates and self.eval_first_epoch and epoch == 0 and eval_interval != 1:
                self._log(
                    f"Epoch {epoch + 1} - forced eval (eval_first_epoch=True, interval={eval_interval})"
                )

            eval_start = time.time()
            metrics = self._evaluate_policy(
                model,
                data_vali,
                stacked_alphas,
                stacked_additions,
                vali_doc_weights,
                eval_cache=eval_cache,
            )
            metrics = jax.block_until_ready(metrics)
            val_eval_time = time.time() - eval_start
            train_metrics = None
            train_eval_time = None
            if self.debug_eval_train_split and train_eval_cache is not None:
                eval_start = time.time()
                train_metrics = self._evaluate_policy(
                    model,
                    data_train,
                    stacked_alphas,
                    stacked_additions,
                    train_doc_weights,
                    eval_cache=train_eval_cache,
                )
                train_metrics = jax.block_until_ready(train_metrics)
                train_eval_time = time.time() - eval_start
            if self.print_updates:
                msg = f"Epoch {epoch + 1} - eval time: val={val_eval_time:.2f}s"
                if train_eval_time is not None:
                    msg += f", train={train_eval_time:.2f}s"
                self._log(msg)
            abs_improvement = float(metrics[1] - last_metric_value)
            self._log_epoch_metrics(
                epoch=epoch + 1,
                metrics=metrics,
                train_metrics=train_metrics,
                abs_improvement=abs_improvement,
                first_metric_value=first_metric_value,
                last_metric_value=last_metric_value,
                start_time=start_time,
                n_queries=n_queries,
                global_step=global_step,
            )
            if (
                self.print_updates
                and epoch <= eval_interval
                and first_metric_value > 0.0
                and float(metrics[1]) < 0.2 * first_metric_value
            ):
                self._log(
                    "Warning: early validation metric dropped by >80% from epoch-0 baseline. "
                    "This can happen with sparse IPS weights and/or unstable first updates; "
                    "inspect epoch-1 forced eval and consider lower learning rate or more grad samples."
                )

            last_metric_value = float(metrics[1])
            if best_metric_value < metrics[1]:
                best_state = nnx.state(model)
                best_metric_value = float(metrics[1])
                best_epoch = epoch + 1

            if (epoch + 1) >= self.early_stop_min_epochs and abs_improvement < self.early_stop_diff:
                no_improve_count += 1
                if self.print_updates:
                    self._log(
                        f"Epoch {epoch + 1}: no improvement "
                        f"(abs_improvement={abs_improvement:.6f} < {self.early_stop_diff:.6f}), "
                        f"patience {no_improve_count}/{self.patience}"
                    )
                if no_improve_count >= self.patience:
                    if self.print_updates:
                        self._log(
                            f"Early stop triggered at epoch {epoch + 1} "
                            f"(patience={self.patience} exhausted)"
                        )
                    break
            else:
                no_improve_count = 0

        nnx.update(model, best_state)
        if self.print_updates:
            self._log(
                "Training finished: restored best checkpoint from epoch %d "
                "(best alpha_val_metric=%0.6f)"
                % (best_epoch, best_metric_value)
            )
        return model, best_metric_value

    # ------------------------------------------------------------------
    # JIT-compiled training primitives
    # ------------------------------------------------------------------

    @partial(nnx.jit, static_argnums=(0, 8))
    def _train_step(
        self,
        model: nnx.Module,
        optimizer: nnx.Optimizer,
        q_feat: jnp.ndarray,
        q_doc_weights: jnp.ndarray,
        q_mask: jnp.ndarray,
        q_rank_weights: jnp.ndarray,
        rng_key: jax.Array,
        n_samples: int,
    ):
        """
        Single-query training step.

        The loss is the *negative* exposure objective (we maximise expected
        exposure weighted by doc rewards, so we minimise its negation):

            loss = -sum_d  E_pi[exposure_d] * reward_d

        `q_rank_weights` acts as `exposure_per_rank` (the [K] alpha vector
        already masked to the query's valid document count).
        `q_doc_weights` acts as the per-document reward signal.
        """
        def loss_fn(model):
            scores = model.score_query(q_feat)
            scores = jnp.where(q_mask, scores, -1e9)
            # exposure_objective returns a scalar; we negate because we minimise.
            obj = exposure_objective(
                rng_key,
                scores,
                q_rank_weights,      # [K]  per-rank exposure weights
                q_doc_weights,       # [D]  per-doc rewards (gradient stopped inside)
                where=q_mask,
                num_samples=n_samples,
            )
            return -obj, scores

        (loss, scores), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)
        optimizer.update(grads)
        return loss, scores, grads

    @partial(nnx.jit, static_argnums=(0, 9))
    def _train_epoch(
        self,
        model: nnx.Module,
        optimizer: nnx.Optimizer,
        q_feat: jnp.ndarray,
        q_doc_weights: jnp.ndarray,
        q_mask: jnp.ndarray,
        q_rank_weights: jnp.ndarray,
        qids: jnp.ndarray,
        epoch_rng_key: jax.Array,
        n_samples: int,
    ):
        """
        Fully-jitted training epoch via `nnx.scan`.

        The RNG key is carried through the scan so each query gets a unique
        subkey, giving unbiased MC gradient estimates without Python overhead.

        Returns (model, optimizer, final_rng_key).
        """
        @partial(nnx.scan, in_axes=(nnx.Carry, 0), out_axes=nnx.Carry)
        def body(carry, qid):
            model, optimizer, rng_key, loss_sum, update_count = carry
            rng_key, subkey = jax.random.split(rng_key)

            q_feat_i = q_feat[qid]
            q_doc_weights_i = q_doc_weights[qid]
            q_mask_i = q_mask[qid]
            q_rank_weights_i = q_rank_weights[qid]

            def loss_fn(model):
                scores = model.score_query(q_feat_i)
                scores = jnp.where(q_mask_i, scores, -1e9)
                obj = exposure_objective(
                    subkey,
                    scores,
                    q_rank_weights_i,
                    q_doc_weights_i,
                    where=q_mask_i,
                    num_samples=n_samples,
                )
                return -obj

            loss, grads = nnx.value_and_grad(loss_fn)(model)
            optimizer.update(grads)
            loss_sum = loss_sum + jnp.asarray(loss, dtype=jnp.float32)
            update_count = update_count + 1
            return (model, optimizer, rng_key, loss_sum, update_count)

        model, optimizer, final_rng_key, loss_sum, update_count = body(
            (
                model,
                optimizer,
                epoch_rng_key,
                jnp.array(0.0, dtype=jnp.float32),
                jnp.array(0, dtype=jnp.int32),
            ),
            qids,
        )
        return model, optimizer, final_rng_key, loss_sum, update_count

    # ------------------------------------------------------------------
    # Evaluation (shared logic with PolicyTrainer)
    # ------------------------------------------------------------------

    def _evaluate_policy(
        self,
        model: nnx.Module,
        data_split,
        stacked_alphas: jnp.ndarray,
        stacked_additions: jnp.ndarray,
        doc_weights,
        *,
        eval_cache: tuple | None = None,
    ):
        doc_weights = jnp.asarray(doc_weights)
        policy_scores = model.score_feature_matrix(jnp.asarray(data_split.feature_matrix))
        if eval_cache is None:
            return pl.datasplit_metrics(
                data_split,
                policy_scores,
                stacked_alphas,
                stacked_additions,
                doc_weights,
                query_norm_factors=None,
                n_samples=self.n_eval_samples,
            )
        doc_id_map, mask = eval_cache
        return self._datasplit_metrics_cached(
            policy_scores,
            stacked_alphas,
            stacked_additions,
            doc_weights,
            doc_id_map,
            mask,
            n_samples=self.n_eval_samples,
        )

    def _prepare_eval_cache(self, data_split) -> tuple:
        doc_id_map_np = np.asarray(data_split.doc_id_map)
        mask_np = np.asarray(data_split.mask, dtype=bool)
        active_rows = np.any(mask_np, axis=1)
        if not np.all(active_rows):
            doc_id_map_np = doc_id_map_np[active_rows]
            mask_np = mask_np[active_rows]
        if (
            self.eval_max_queries is not None
            and doc_id_map_np.shape[0] > self.eval_max_queries
        ):
            n_active = int(doc_id_map_np.shape[0])
            rng = np.random.default_rng(0)
            keep_idx = np.sort(
                rng.choice(n_active, size=self.eval_max_queries, replace=False)
            )
            doc_id_map_np = doc_id_map_np[keep_idx]
            mask_np = mask_np[keep_idx]
            if self.print_updates:
                self._log(
                    "Evaluation cache capped to %d/%d active queries "
                    "(deterministic sample; set ips.trainer.eval_max_queries to adjust)."
                    % (self.eval_max_queries, n_active)
                )
        return jnp.asarray(doc_id_map_np), jnp.asarray(mask_np)

    @staticmethod
    def _datasplit_metrics_cached(
        policy_scores: jnp.ndarray,
        weight_per_rank: jnp.ndarray,
        addition_per_rank: jnp.ndarray,
        weight_per_doc: jnp.ndarray,
        doc_id_map: jnp.ndarray,
        mask: jnp.ndarray,
        *,
        n_samples: int,
    ) -> jnp.ndarray:
        policy_scores = jnp.asarray(policy_scores).reshape(-1)
        weight_per_doc = jnp.asarray(weight_per_doc).reshape(-1)
        weight_per_rank = jnp.asarray(weight_per_rank)
        addition_per_rank = jnp.asarray(addition_per_rank)
        n_queries = int(doc_id_map.shape[0])
        n_metrics = int(weight_per_rank.shape[1])
        if n_queries == 0:
            return jnp.zeros((n_metrics,), dtype=weight_per_rank.dtype)
        dummy_norm = jnp.ones((n_queries, n_metrics), dtype=weight_per_rank.dtype)
        return pl._datasplit_metrics_jax(
            policy_scores,
            weight_per_rank,
            addition_per_rank,
            weight_per_doc,
            doc_id_map,
            mask,
            dummy_norm,
            n_samples=n_samples,
            use_norm=False,
        )

    # ------------------------------------------------------------------
    # Data preparation helpers (identical to PolicyTrainer)
    # ------------------------------------------------------------------

    @staticmethod
    def _stack_alphas(train_alpha: jnp.ndarray, vali_alpha: jnp.ndarray):
        stacked_alphas = jnp.stack([train_alpha, vali_alpha], axis=-1)
        stacked_additions = jnp.zeros_like(stacked_alphas)
        return stacked_alphas, stacked_additions

    @staticmethod
    def _prepare_query_tensors(data_split, doc_weights: jnp.ndarray):
        q_feat = jnp.asarray(data_split.query_doc_features)
        q_mask = jnp.asarray(data_split.mask).astype(bool)
        q_feat = jnp.where(q_mask[..., None], q_feat, 0.0)
        q_doc_weights = pl.pad_by_doc_id(
            doc_weights,
            jnp.asarray(data_split.doc_id_map),
            fill_value=0.0,
        )
        q_doc_weights = jnp.where(q_mask, q_doc_weights, 0.0)
        return q_feat, q_doc_weights, q_mask

    @staticmethod
    def _prepare_rank_weights(q_mask: jnp.ndarray, train_alpha: jnp.ndarray) -> jnp.ndarray:
        n_valid = jnp.sum(q_mask, axis=1, dtype=jnp.int32)
        ranks = jnp.arange(train_alpha.shape[0])[None, :]
        rank_mask = ranks < n_valid[:, None]
        return train_alpha[None, :] * rank_mask

    @staticmethod
    def _select_training_queries(
        train_q_doc_weights: np.ndarray,
        train_q_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        query_weights = np.abs(np.asarray(train_q_doc_weights))
        if query_weights.ndim != 2:
            raise ValueError(
                f"Expected 2D query-doc weight matrix, got shape={query_weights.shape}"
            )
        if train_q_mask is not None:
            query_weights = np.where(np.asarray(train_q_mask, dtype=bool), query_weights, 0.0)
        query_weights = np.sum(query_weights, axis=1)
        qids = np.flatnonzero(query_weights > 0)
        if qids.size == 0:
            raise ValueError("No queries with non-zero weights available for training.")
        return qids

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _log(self, message: str) -> None:
        if self.print_updates:
            print(message)

    @staticmethod
    def _block_on_model_params(model: nnx.Module) -> None:
        params = nnx.state(model, nnx.Param)
        leaves, _ = jax.tree_util.tree_flatten(params)
        if leaves:
            jax.block_until_ready(leaves[0])

    def _log_initial_metrics(
        self,
        metrics: np.ndarray,
        *,
        train_metrics: np.ndarray | None = None,
    ) -> None:
        self._log(
            "epoch 0 baseline (validation split): alpha_train_metric=%0.04f alpha_val_metric=%0.04f"
            % (metrics[0], metrics[1])
        )
        if train_metrics is not None:
            self._log(
                "epoch 0 baseline (train split): alpha_train_metric=%0.04f alpha_val_metric=%0.04f"
                % (train_metrics[0], train_metrics[1])
            )

    def _log_epoch_metrics(
        self,
        epoch: int,
        metrics: np.ndarray,
        train_metrics: np.ndarray | None,
        abs_improvement: float,
        first_metric_value: float,
        last_metric_value: float,
        start_time: float,
        n_queries: int,
        global_step: int,
    ) -> None:
        if not self.print_updates:
            return
        improvement = metrics[1] / last_metric_value - 1.0 if last_metric_value != 0 else 0.0
        total_improvement = metrics[1] / first_metric_value - 1.0 if first_metric_value != 0 else 0.0
        average_time = (time.time() - start_time) / max(global_step, 1) * n_queries
        line = (
            "epoch %d: "
            "val(alpha_train_metric=%0.04f alpha_val_metric=%0.04f) "
            "epoch-time %0.04f "
            "abs-improvement %.8f "
            "improvement %.8f "
            "total-improvement %.8f "
            % (
                epoch,
                metrics[0],
                metrics[1],
                average_time,
                abs_improvement,
                improvement,
                total_improvement,
            )
        )
        if train_metrics is not None:
            line += "train(alpha_train_metric=%0.04f alpha_val_metric=%0.04f) " % (
                train_metrics[0],
                train_metrics[1],
            )
        self._log(line)

    def _wandb_log_step(self, model, loss, scores, grads, global_step: int) -> None:
        if not (self.use_wandb and wandb is not None and wandb.run is not None):
            return
        log_dict = {"train_exposure/loss_step": float(loss)}
        log_dict["train_exposure/scores"] = wandb.Histogram(np.asarray(scores))
        flat_grads, _ = jax.tree_util.tree_flatten(grads)
        grad_norms = [np.linalg.norm(np.asarray(g)) for g in flat_grads]
        log_dict["gradients_exposure/global_norm"] = np.linalg.norm(grad_norms)
        for i, g in enumerate(flat_grads):
            log_dict[f"gradients_exposure/layer_{i}_norm"] = grad_norms[i]
            if i < 10:
                log_dict[f"gradients_exposure/layer_{i}_dist"] = wandb.Histogram(np.asarray(g))
        params = nnx.state(model, nnx.Param)
        flat_params, _ = jax.tree_util.tree_flatten(params)
        for i, p in enumerate(flat_params):
            if i < 10:
                log_dict[f"weights_exposure/layer_{i}_dist"] = wandb.Histogram(np.asarray(p))
        wandb.log(log_dict, step=global_step)
