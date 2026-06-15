import logging
from typing import Any, Dict, Optional

try:
    import wandb
except ImportError:
    wandb = None


class Tracker:
    """Centralized MLOps tracking and logging utility.
    
    Wraps standard Python logging and Weights & Biases (wandb) for centralized
    experiment tracking.
    """
    
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None, project: str = "feature_based_propensities_for_ULTR", job_type: str = "train", use_wandb: bool = False):
        self.logger = logging.getLogger(name)
        self.use_wandb = use_wandb and wandb is not None
        
        if self.use_wandb:
            try:
                wandb.init(project=project, config=config, job_type=job_type)
            except Exception as e:
                self.use_wandb = False
                self.logger.warning(
                    "wandb.init failed (%s). Continuing with local logging only.",
                    e,
                )

    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None, commit: bool = True) -> None:
        """Log key-value metrics to wandb."""
        if self.use_wandb and wandb.run is not None:
            wandb.log(metrics, step=step, commit=commit)

    def log_histogram(self, key: str, values: Any, commit: bool = False) -> None:
        """Log a histogram to wandb."""
        if self.use_wandb and wandb.run is not None:
            wandb.log({key: wandb.Histogram(values)}, commit=commit)

    def info(self, msg: str, *args, **kwargs) -> None:
        self.logger.info(msg, *args, **kwargs)

    def debug(self, msg: str, *args, **kwargs) -> None:
        self.logger.debug(msg, *args, **kwargs)
        
    def warning(self, msg: str, *args, **kwargs) -> None:
        self.logger.warning(msg, *args, **kwargs)
        
    def error(self, msg: str, *args, **kwargs) -> None:
        self.logger.error(msg, *args, **kwargs)

    def finish(self) -> None:
        """Finish the tracking run (e.g. close wandb)."""
        if self.use_wandb:
            try:
                wandb.finish()
            except Exception as e:
                self.logger.warning(f"Failed to finish wandb run: {e}")
