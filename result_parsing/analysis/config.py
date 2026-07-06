from dataclasses import dataclass
from typing import Optional, Tuple

NO_PROPENSITY_IN_FOLDER = {"logging-policy", "naive-ho", "naive"}

@dataclass(frozen=True)
class ExperimentConfig:
    dataset_name: str = "mslr30k"
    # base path for the original policy_models experiments
    base_path_policy_models: str = "results/new_policy/"
    launcher_policy: str = "slurm"
    experiment_name_policy: str = "new_policy"
    # base path for the two_tower experiments
    base_path_two_tower: str = "results/two-tower/"
    launcher_two_tower: str = "slurmcpu"

    n_sessions_list: Tuple[int, ...] = (
        500, 1000, 2500, 5000, 7500, 10000,
        25000, 50000, 75000, 100000, 500000, 1000000, 5000000
    )
    n_session_percentage_list: Optional[Tuple[float, ...]] = None

    propensity_models: Tuple[str, ...] = (
        "frequency-based",
        "true_propensity",
        "MLPregression",
        "cosine",
        "knn",
        "kmeans",
    )

    ips_models: Tuple[str, ...] = (
        "dm", "dr", "ips", "max-score", "logging-policy", "naive-ho", "mul-two-tower", "add-two-tower"
    )

    temperatures: Tuple[float, ...] = (0.0, 0.5, 0.75)
    random_states: Tuple[int, ...] = (40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54)
    policy_strengths: Tuple[float, ...] = (1.0,)
    filter_single_display_pairs_values: Tuple[bool, ...] = (True, False)
