"""Variance of one document's IPS reward estimate across simulations.

This follows the repository's click simulator:

    position_bias_logit[k] = -bias_strength * log(k + 1)
    relevance_logit = label - floor(max_label / 2)
    click_probability[k] = sigmoid(position_bias_logit[k] + relevance_logit)

As in the propensity pipeline, the position examination values are
``sigmoid(position_bias_logit)``. The true document propensity averages these
values using the true position distribution. The frequency-based propensity
uses the empirical position frequencies from the simulated observations.
Ranks use the repository's e-greedy sampler: the deterministic ranking is used
with probability ``1 - policy_temperature`` and a random ranking otherwise.

Run:
    ./.venv/bin/python scripts/simple_reward_variance_experiment.py
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.special import expit

from feature_based_propensities_for_ULTR.logging_policy import EGreedySampler
from feature_based_propensities_for_ULTR.simulation import (
    Simulator,
    get_position_bias,
)


def simulate_once(
    *,
    simulator: Simulator,
    logging_policy_sampler: EGreedySampler,
    n_observations: int,
    position_probabilities: np.ndarray,
    examination_propensities: np.ndarray,
    ranking_labels: np.ndarray,
    ranking_scores: np.ndarray,
    ranking_mask: np.ndarray,
) -> tuple[float, float, float]:
    """Return true- and frequency-propensity IPS estimates for one document."""
    positions = np.empty(n_observations, dtype=np.int64)
    clicks = np.empty(n_observations, dtype=np.int8)
    for observation in range(n_observations):
        ranking = logging_policy_sampler(
            scores=ranking_scores,
            where=ranking_mask,
        )
        target_position = int(np.flatnonzero(ranking == 0)[0])
        positions[observation] = target_position
        ranking_clicks = simulator.sample_clicks(
            labels=ranking_labels,
            positions=ranking,
            where=ranking_mask,
        )
        clicks[observation] = ranking_clicks[target_position]
    observation_propensities = examination_propensities[positions]

    true_propensity = float(position_probabilities @ examination_propensities)
    frequency_propensity = float(np.mean(observation_propensities))

    # These are the actual IPS estimates: divide every click observation by
    # the propensity, then average over all observations.
    true_ips = float(np.mean(clicks) / true_propensity)
    frequency_ips = float(np.mean(clicks) / frequency_propensity)
    return true_ips, frequency_ips, frequency_propensity


def run_experiment(
    *,
    observation_counts: list[int],
    n_simulations: int,
    position_probabilities: np.ndarray,
    examination_propensities: np.ndarray,
    click_probabilities: np.ndarray,
    position_bias_logits: np.ndarray,
    relevance_logit: float,
    seed: int,
    label: float,
    max_label: int,
    bias_strength: float,
    policy_temperature: float,
    deterministic_rank: int,
    output_csv: Path,
) -> None:
    logging_policy_sampler = EGreedySampler(
        policy_temperature=policy_temperature,
        random_state=seed,
    )
    simulator = Simulator(
        logging_policy_ranker=lambda **_: None,
        logging_policy_sampler=logging_policy_sampler,
        bias_strength=bias_strength,
        max_label=max_label,
        random_state=seed,
    )
    cutoff = len(position_probabilities)
    ranking_labels = np.full(cutoff, label, dtype=np.float64)
    ranking_mask = np.ones(cutoff, dtype=bool)
    deterministic_order = np.insert(
        np.arange(1, cutoff, dtype=np.int64),
        deterministic_rank - 1,
        0,
    )
    ranking_scores = np.empty(cutoff, dtype=np.float64)
    ranking_scores[deterministic_order] = np.arange(cutoff, 0, -1)
    true_propensity = float(position_probabilities @ examination_propensities)
    expected_ctr = float(position_probabilities @ click_probabilities)
    expected_true_ips = expected_ctr / true_propensity

    print(
        "position_bias_logits="
        + np.array2string(position_bias_logits, precision=4, separator=",")
    )
    print(
        "examination_propensities="
        + np.array2string(examination_propensities, precision=4, separator=",")
    )
    print(
        "click_probabilities="
        + np.array2string(click_probabilities, precision=4, separator=",")
    )
    print(f"relevance_logit={relevance_logit:.8f}")
    print(f"policy_temperature={policy_temperature:.8f}")
    print(f"deterministic_rank={deterministic_rank}")
    print(
        "target_rank_probabilities="
        + np.array2string(position_probabilities, precision=4, separator=",")
    )
    print(f"true_document_propensity={true_propensity:.8f}")
    print(f"expected_true_ips={expected_true_ips:.8f}")
    print(
        "n_observations,"
        "var_true_ips,var_frequency_ips,"
        "mean_true_ips,mean_frequency_ips,"
        "mean_frequency_propensity"
    )
    rows: list[dict[str, int | float]] = []

    for n_observations in observation_counts:
        true_estimates = np.empty(n_simulations)
        frequency_estimates = np.empty(n_simulations)
        frequency_propensities = np.empty(n_simulations)

        for simulation in range(n_simulations):
            true_ips, frequency_ips, frequency_propensity = simulate_once(
                simulator=simulator,
                logging_policy_sampler=logging_policy_sampler,
                n_observations=n_observations,
                position_probabilities=position_probabilities,
                examination_propensities=examination_propensities,
                ranking_labels=ranking_labels,
                ranking_scores=ranking_scores,
                ranking_mask=ranking_mask,
            )
            true_estimates[simulation] = true_ips
            frequency_estimates[simulation] = frequency_ips
            frequency_propensities[simulation] = frequency_propensity

        row = {
            "n_observations": n_observations,
            "var_true_ips": float(np.var(true_estimates, ddof=1)),
            "var_frequency_ips": float(
                np.var(frequency_estimates, ddof=1)
            ),
            "mean_true_ips": float(np.mean(true_estimates)),
            "mean_frequency_ips": float(np.mean(frequency_estimates)),
            "mean_frequency_propensity": float(
                np.mean(frequency_propensities)
            ),
            "n_simulations": n_simulations,
            "cutoff": cutoff,
            "label": label,
            "max_label": max_label,
            "bias_strength": bias_strength,
            "policy_temperature": policy_temperature,
            "deterministic_rank": deterministic_rank,
            "true_document_propensity": true_propensity,
            "expected_true_ips": expected_true_ips,
        }
        rows.append(row)
        print(
            f"{row['n_observations']},"
            f"{row['var_true_ips']:.8f},"
            f"{row['var_frequency_ips']:.8f},"
            f"{row['mean_true_ips']:.8f},"
            f"{row['mean_frequency_ips']:.8f},"
            f"{row['mean_frequency_propensity']:.8f}"
        )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved_csv={output_csv}")


def repo_click_probabilities(
    *,
    cutoff: int,
    bias_strength: float,
    label: float,
    max_label: int,
    seed: int,
) -> tuple[np.ndarray, float]:
    """Get per-position click probabilities from the repository simulator."""
    simulator = Simulator(
        logging_policy_ranker=lambda **_: None,
        logging_policy_sampler=lambda **_: None,
        bias_strength=bias_strength,
        max_label=max_label,
        random_state=seed,
    )
    labels = np.full(cutoff, label, dtype=np.float64)
    positions = np.arange(cutoff, dtype=np.int64)
    where = np.ones(cutoff, dtype=bool)
    _, displayed_relevance, _, click_probabilities = (
        simulator._click_probability_components(
            labels=labels,
            positions=positions,
            where=where,
        )
    )
    return (
        np.asarray(click_probabilities, dtype=np.float64),
        float(displayed_relevance[0]),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Variance of one document's IPS reward estimate."
    )
    parser.add_argument(
        "--observation-counts",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        help="Numbers of observations of the document.",
    )
    parser.add_argument(
        "--n-simulations",
        type=int,
        default=100_000,
        help="Repeated simulations for each observation count.",
    )
    parser.add_argument(
        "--cutoff",
        type=int,
        default=25,
        help="Number of displayed positions.",
    )
    parser.add_argument(
        "--bias-strength",
        type=float,
        default=1.0,
        help="Position-bias strength passed to get_position_bias.",
    )
    parser.add_argument(
        "--policy-temperature",
        type=float,
        default=0.5,
        help="E-greedy probability of replacing the ranking with a random ranking.",
    )
    parser.add_argument(
        "--deterministic-rank",
        type=int,
        default=1,
        help="Target document's 1-based rank when e-greedy does not randomize.",
    )
    parser.add_argument(
        "--label",
        type=float,
        default=4.0,
        help="Fixed relevance label of the document.",
    )
    parser.add_argument(
        "--max-label",
        type=int,
        default=4,
        help="Maximum label used by the simulator's relevance transform.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "results"
        / "simple_reward_variance_experiment.csv",
        help="Path for the summary CSV.",
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.cutoff <= 0:
        raise ValueError("Cutoff must be positive.")
    if args.max_label <= 0:
        raise ValueError("Maximum label must be positive.")
    if not 0 <= args.policy_temperature <= 1:
        raise ValueError("Policy temperature must be in [0, 1].")
    if not 1 <= args.deterministic_rank <= args.cutoff:
        raise ValueError("Deterministic rank must be between 1 and the cutoff.")
    if any(count <= 0 for count in args.observation_counts):
        raise ValueError("Observation counts must be positive.")

    position_probabilities = np.full(
        args.cutoff,
        args.policy_temperature / args.cutoff,
    )
    position_probabilities[args.deterministic_rank - 1] += (
        1.0 - args.policy_temperature
    )

    position_bias_logits = np.asarray(
        get_position_bias(args.cutoff, strength=args.bias_strength),
        dtype=np.float64,
    )
    # This is exactly how the oracle position-bias logits are converted to
    # alpha in experiments/ips_pipeline.py.
    examination_propensities = expit(position_bias_logits)
    click_probabilities, relevance_logit = repo_click_probabilities(
        cutoff=args.cutoff,
        bias_strength=args.bias_strength,
        label=args.label,
        max_label=args.max_label,
        seed=args.seed,
    )

    run_experiment(
        observation_counts=args.observation_counts,
        n_simulations=args.n_simulations,
        position_probabilities=position_probabilities,
        examination_propensities=examination_propensities,
        click_probabilities=click_probabilities,
        position_bias_logits=position_bias_logits,
        relevance_logit=relevance_logit,
        seed=args.seed,
        label=args.label,
        max_label=args.max_label,
        bias_strength=args.bias_strength,
        policy_temperature=args.policy_temperature,
        deterministic_rank=args.deterministic_rank,
        output_csv=args.output_csv,
    )


if __name__ == "__main__":
    main()
