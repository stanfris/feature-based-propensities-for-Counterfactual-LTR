"""Low-level load/save helpers for click and rating dataset NPZ artifacts."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from feature_based_propensities_for_ULTR.data.base import RatingDataset
from feature_based_propensities_for_ULTR.simulation import ClickDataset


def load_click_dataset_npz(file_path: Path):
    data = np.load(file_path, allow_pickle=True)
    lp_query_doc_features = (
        data["lp_query_doc_features"]
        if "lp_query_doc_features" in data.files
        else None
    )

    sessions_per_doc_pos = (
        data["sessions_per_doc_pos_diag"]
        if "sessions_per_doc_pos_diag" in data.files
        else data["sessions_per_doc_pos"]
    )

    if "clicks" in data and "positions" in data and "sessions" in data:
        rating_dataset = RatingDataset(
            query=data["query"],
            query_doc_ids=data["query_doc_ids"],
            query_doc_features=data["query_doc_features"],
            lp_query_doc_features=lp_query_doc_features,
            labels=data["labels"],
            mask=data["mask"],
            n=data["n"],
        )
        click_dataset = ClickDataset(
            rating_dataset=rating_dataset,
            sessions=data["sessions"],
            clicks=data["clicks"],
            positions=data["positions"],
            sessions_per_query=data["sessions_per_query"],
            sessions_per_doc_pos=sessions_per_doc_pos,
        )
        return rating_dataset, click_dataset

    rating_dataset = RatingDataset(
        query=data["queries"],
        query_doc_ids=data["query_doc_ids"],
        query_doc_features=data["query_doc_features"],
        lp_query_doc_features=lp_query_doc_features,
        labels=data["padded_clicks"],
        mask=data["mask"],
        n=data["n"],
    )
    sessions = np.arange(len(rating_dataset))
    click_dataset = ClickDataset(
        rating_dataset=rating_dataset,
        sessions=sessions,
        clicks=data["padded_clicks"],
        positions=data["padded_positions"],
        sessions_per_query=data["sessions_per_query"],
        sessions_per_doc_pos=sessions_per_doc_pos,
    )
    return rating_dataset, click_dataset


def rating_dataset_from_click_dataset(click_dataset: ClickDataset) -> RatingDataset:
    return RatingDataset(
        query=np.asarray(click_dataset.query),
        query_doc_ids=np.asarray(click_dataset.query_doc_ids),
        query_doc_features=np.asarray(click_dataset.query_doc_features),
        lp_query_doc_features=(
            np.asarray(click_dataset.lp_query_doc_features)
            if click_dataset.has_separate_lp_query_doc_features
            else None
        ),
        labels=np.asarray(click_dataset.labels),
        mask=np.asarray(click_dataset.mask),
        n=np.asarray(click_dataset.n),
    )


def save_rating_dataset_npz(dataset: RatingDataset, file_path: Path) -> None:
    payload = {
        "query": dataset.query,
        "query_doc_ids": dataset.query_doc_ids,
        "query_doc_features": dataset.query_doc_features,
        "labels": dataset.labels,
        "mask": dataset.mask,
        "n": dataset.n,
    }
    if dataset.has_separate_lp_query_doc_features:
        payload["lp_query_doc_features"] = dataset.lp_query_doc_features
    np.savez_compressed(file_path, **payload)
    print(f"RatingDataset saved to {file_path}")


def load_rating_dataset_npz(file_path: Path) -> RatingDataset:
    data = np.load(file_path, allow_pickle=True)
    return RatingDataset(
        query=data["query"],
        query_doc_ids=data["query_doc_ids"],
        query_doc_features=data["query_doc_features"],
        lp_query_doc_features=(
            data["lp_query_doc_features"]
            if "lp_query_doc_features" in data.files
            else None
        ),
        labels=data["labels"],
        mask=data["mask"],
        n=data["n"],
    )
