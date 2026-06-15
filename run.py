import logging

import hydra
from omegaconf import DictConfig, OmegaConf

from feature_based_propensities_for_ULTR.experiments.ips_pipeline import IPSExperimentPipeline
from feature_based_propensities_for_ULTR.utils import set_device

logging.getLogger("jax").setLevel(logging.WARNING)


@hydra.main(version_base="1.3", config_path="config", config_name="config")
def main(config: DictConfig) -> None:
    set_device(config.device)
    print(OmegaConf.to_yaml(config))
    pipeline = IPSExperimentPipeline(config)
    pipeline.run()


if __name__ == "__main__":
    main()