"""Click simulation primitives and the Simulator orchestrator."""

from __future__ import annotations

from typing import Callable, Optional, Sequence

import jax
import numpy as np
from scipy.special import expit
from tqdm import tqdm

from feature_based_propensities_for_ULTR.data.base import RatingDataset

from .aggregation import _aggregate_click_dataset
from .datasets import ClickDataset


def _parse_ratio_list(
    ratios: Optional[Sequence[float]],
    *,
    name: str,
) -> Optional[np.ndarray]:
    if ratios is None:
        return None
    if isinstance(ratios, (str, np.str_)):
        text = str(ratios).strip().strip("[](){}")
        ratios = [float(tok) for tok in text.split(",") if tok.strip()]
    arr = np.asarray(ratios, dtype=np.float64)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError(f"{name} must be a 1D sequence with at least one entry.")
    if np.any(arr < 0):
        raise ValueError(f"{name} must be non-negative.")
    total = float(arr.sum())
    if not np.isclose(total, 1.0, rtol=1e-5, atol=1e-8):
        raise ValueError(f"{name} must sum to 1.0 (got {total:.6f}).")
    return arr


def _sample_sessions_from_query_counts(
    *,
    rng: np.random.RandomState,
    n_queries: int,
    n_sessions: int,
    ratios: np.ndarray,
) -> np.ndarray:
    if n_sessions <= 0:
        return np.zeros((0,), dtype=np.int64)
    if n_queries <= 0:
        raise ValueError("n_queries must be > 0 when sampling sessions.")

    # Allocate the total session budget by ratio using largest-remainder rounding.
    per_count_target = ratios * float(n_sessions)
    per_count_budget = np.floor(per_count_target).astype(np.int64)
    remainder = n_sessions - int(per_count_budget.sum())
    if remainder > 0:
        fractional = per_count_target - per_count_budget
        order = rng.permutation(ratios.size)
        top = order[np.argsort(fractional[order])[::-1][:remainder]]
        per_count_budget[top] += 1

    # Randomly assign query ids for each count bucket (counts correspond to 1..K).
    counts = np.zeros(n_queries, dtype=np.int64)
    for count_idx, budget in enumerate(per_count_budget):
        if budget <= 0:
            continue
        per_query_count = count_idx + 1

        n_full = int(budget // per_query_count)
        if n_full > 0:
            replace = n_full > n_queries
            full_queries = rng.choice(n_queries, size=n_full, replace=replace)
            np.add.at(counts, full_queries, per_query_count)

        rem = int(budget % per_query_count)
        if rem > 0:
            rem_queries = rng.choice(n_queries, size=rem, replace=(rem > n_queries))
            np.add.at(counts, rem_queries, 1)

    sessions = np.repeat(np.arange(n_queries, dtype=np.int64), counts)
    rng.shuffle(sessions)
    return sessions


class Simulator:
    def __init__(
        self,
        logging_policy_ranker: Callable,
        logging_policy_sampler: Callable,
        bias_strength: float,
        max_label: int = 4,
        *,
        random_state: int,
    ):
        self.logging_policy_ranker = logging_policy_ranker
        self.logging_policy_sampler = logging_policy_sampler
        self.bias_strength = bias_strength
        self.rng = np.random.RandomState(random_state)
        self.max_label = max_label
        self.random_state = random_state

    def __call__(
        self,
        rating_dataset: RatingDataset,
        n_sessions: int,
        *,
        top_x: Optional[int] = None,
        query_sampling_ratios: Optional[Sequence[float]] = None,
        single_observation_per_query_doc: bool = False,
        sequential_query_sampling: bool = False,
        debug: bool = False,
        debug_label: Optional[str] = None,
        debug_max_samples: int = 3,
    ) -> ClickDataset:
        if top_x is not None:
            top_x = int(top_x)
            if top_x <= 0:
                raise ValueError(f"top_x must be > 0 when provided, got {top_x}.")

        if single_observation_per_query_doc or sequential_query_sampling:
            if query_sampling_ratios is not None:
                print(
                    "single-sample query sampling: ignoring query_sampling_ratios."
                )
            requested_sessions = int(n_sessions)
            max_sessions = min(requested_sessions, int(rating_dataset.n_queries))
            if max_sessions < requested_sessions:
                print(
                    "single-sample query sampling: capping sessions from "
                    f"{requested_sessions} to {max_sessions}."
                )
            if sequential_query_sampling:
                upper = max_sessions - 1
                print(
                    "force_single_sample=True: sampling queries sequentially "
                    f"from 0 to {upper}."
                )
                sessions = np.arange(max_sessions, dtype=np.int64)
            else:
                sessions = self.rng.choice(
                    rating_dataset.n_queries,
                    size=max_sessions,
                    replace=False,
                )
        else:
            ratios = _parse_ratio_list(query_sampling_ratios, name="query_sampling_ratios")
            if ratios is None:
                sessions = self.rng.randint(0, rating_dataset.n_queries, size=(n_sessions,))
            else:
                sessions = _sample_sessions_from_query_counts(
                    rng=self.rng,
                    n_queries=rating_dataset.n_queries,
                    n_sessions=n_sessions,
                    ratios=ratios,
                )
        sampled_clicks = []
        sampled_positions = []

        sessions_per_query = np.zeros(rating_dataset.n_queries)
        sessions_per_doc_pos = np.zeros(
            (
                rating_dataset.n_queries,
                rating_dataset.n_positions,
                rating_dataset.n_positions,
            )
        )

        scores = self.logging_policy_ranker(
            lp_query_doc_features=rating_dataset.lp_query_doc_features,
            labels=rating_dataset.labels,
            where=rating_dataset.mask,
        )
        debug_samples_printed = 0

        for session_idx in tqdm(sessions, desc="Simulating clicks.."):
            sample = rating_dataset[session_idx]

            positions = self.logging_policy_sampler(
                scores=scores[session_idx],
                where=sample["mask"],
            )

            if top_x is not None:
                positions = positions[:top_x]

            clicks = self.sample_clicks(
                labels=sample["labels"],
                positions=positions,
                where=sample["mask"],
            )
            if debug and debug_samples_printed < int(debug_max_samples):
                self._debug_print_click_sample(
                    session_idx=int(session_idx),
                    labels=sample["labels"],
                    positions=positions,
                    where=sample["mask"],
                    clicks=clicks,
                    debug_label=debug_label,
                )
                debug_samples_printed += 1

            # Keep track of the number of sessions per query and per document position:
            doc_idx = np.arange(len(positions))
            sessions_per_query[session_idx] += 1
            sessions_per_doc_pos[session_idx, doc_idx, positions] += 1

            sampled_positions.append(positions)
            sampled_clicks.append(clicks)

        return ClickDataset(
            rating_dataset=rating_dataset,
            sessions=sessions,
            clicks=np.stack(sampled_clicks),
            positions=np.stack(sampled_positions),
            sessions_per_query=sessions_per_query,
            sessions_per_doc_pos=sessions_per_doc_pos,
        )

    def aggregate(
        self,
        *,
        rating_dataset: Optional[RatingDataset] = None,
        n_sessions: Optional[int] = None,
        click_dataset: Optional[ClickDataset] = None,
        cutoff: Optional[int] = None,
        positions_are_ranks: Optional[bool] = None,
        use_query_doc_id_mapping: bool = False,
        display_histogram_max: int = 100,
        display_histogram_path: Optional[str] = None,
        query_sampling_ratios: Optional[Sequence[float]] = None,
        single_observation_per_query_doc: bool = False,
        top_x: Optional[int] = None,
        rng_key: Optional[jax.Array] = None,
        debug: bool = False,
    ):
        """
        Simulate (optional) and aggregate click data into a document-level dataset.

        - If click_dataset is provided, it will be aggregated directly.
        - Otherwise, rating_dataset + n_sessions are required to simulate clicks first.
        """
        if click_dataset is None:
            if rating_dataset is None or n_sessions is None:
                raise ValueError(
                    "Provide click_dataset or (rating_dataset and n_sessions)."
                )
            click_dataset = self(
                rating_dataset,
                n_sessions,
                top_x=top_x,
                query_sampling_ratios=query_sampling_ratios,
                single_observation_per_query_doc=single_observation_per_query_doc,
            )

        if rating_dataset is None:
            rating_dataset = RatingDataset(
                query=click_dataset.query,
                query_doc_ids=click_dataset.query_doc_ids,
                query_doc_features=click_dataset.query_doc_features,
                lp_query_doc_features=click_dataset.lp_query_doc_features,
                labels=click_dataset.labels,
                mask=click_dataset.mask,
                n=click_dataset.n,
            )

        if rng_key is None:
            rng_key = jax.random.PRNGKey(self.random_state)

        return _aggregate_click_dataset(
            rating_dataset=rating_dataset,
            click_dataset=click_dataset,
            n_sessions=n_sessions,
            cutoff=cutoff,
            positions_are_ranks=positions_are_ranks,
            use_query_doc_id_mapping=use_query_doc_id_mapping,
            display_histogram_max=display_histogram_max,
            display_histogram_path=display_histogram_path,
            rng_key=rng_key,
            debug=debug,
        )

    def sample_clicks(
        self, labels: np.ndarray, positions: np.ndarray, where: np.ndarray
    ):
        _, _, _, click_prob = self._click_probability_components(
            labels=labels,
            positions=positions,
            where=where,
        )
        return self.rng.binomial(n=1, p=click_prob)

    def _click_probability_components(
        self,
        *,
        labels: np.ndarray,
        positions: np.ndarray,
        where: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        positions = np.asarray(positions, dtype=np.int64)
        bias = get_position_bias(len(positions), strength=self.bias_strength)
        relevance = get_relevance(labels, max_label=self.max_label)
        displayed_relevance = relevance[positions]
        displayed_mask = np.asarray(where, dtype=bool)[positions]

        # Rank documents according to their position and apply position bias:
        click_logits = bias + displayed_relevance
        assert np.isfinite(click_logits).all()
        click_prob = expit(click_logits)
        # Ensure masked items cannot be clicked:
        click_prob = np.where(displayed_mask, click_prob, 0)
        return bias, displayed_relevance, click_logits, click_prob

    def _debug_print_click_sample(
        self,
        *,
        session_idx: int,
        labels: np.ndarray,
        positions: np.ndarray,
        where: np.ndarray,
        clicks: np.ndarray,
        debug_label: Optional[str],
    ) -> None:
        bias, displayed_relevance, click_logits, click_prob = (
            self._click_probability_components(
                labels=labels,
                positions=positions,
                where=where,
            )
        )
        prefix = "Click sampling debug"
        if debug_label:
            prefix += f" [{debug_label}]"
        print(f"{prefix} session={session_idx}")
        print(f"  positions:     {self._format_debug_vector(positions)}")
        print(f"  position_bias: {self._format_debug_vector(bias)}")
        print(f"  relevance:     {self._format_debug_vector(displayed_relevance)}")
        print(f"  click_logits:  {self._format_debug_vector(click_logits)}")
        print(f"  click_prob:    {self._format_debug_vector(click_prob)}")
        print(f"  sampled_click: {self._format_debug_vector(clicks)}")

        displayed_labels = np.asarray(labels)[np.asarray(positions, dtype=np.int64)]
        print(f"  labels:        {self._format_debug_vector(displayed_labels)}")

    @staticmethod
    def _format_debug_vector(values: np.ndarray, max_items: int = 25) -> str:
        arr = np.asarray(values)
        if arr.ndim == 0:
            return str(arr.item())
        if arr.size > max_items:
            head = arr[:max_items]
            return (
                np.array2string(head, precision=4, separator=", ", max_line_width=120)
                + f" ... (len={arr.size})"
            )
        return np.array2string(arr, precision=4, separator=", ", max_line_width=120)


def get_position_bias(n: int, strength: float = 1):
    return -strength * np.log(np.arange(n) + 1)


def get_relevance(labels: np.ndarray, max_label: int):
    # Normalize labels to logit range of: [-max_label/2, max_label/2]:
    return labels - (max_label // 2)
