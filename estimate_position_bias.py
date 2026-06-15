import logging

import hydra
from omegaconf import DictConfig, OmegaConf

from two_tower_confounding.experiments.position_bias_pipeline import estimate_position_bias
from two_tower_confounding.utils import set_device

logging.getLogger("jax").setLevel(logging.WARNING)


@hydra.main(version_base="1.3", config_path="config", config_name="config")
def main(config: DictConfig) -> None:
    set_device(config.device)
    print(OmegaConf.to_yaml(config))
    estimate_position_bias(config)


if __name__ == "__main__":
    main()
