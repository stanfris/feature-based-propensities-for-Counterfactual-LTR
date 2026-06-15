from enum import Enum
from typing import Optional, Sequence, Union

import jax
import numpy as np
import pandas as pd
from flax import nnx

from feature_based_propensities_for_ULTR.data.base import RatingDataset
from feature_based_propensities_for_ULTR.data.utils.features import parse_feature_selection
from feature_based_propensities_for_ULTR.data.utils.tensor import log1p, pad


class Relevance(Enum):
    ORIGINAL = "original"
    LINEAR = "linear"
    DEEP = "deep"


class Preprocessor:
    FEATURE_DTYPE = np.float32

    def __init__(
        self,
        normalize_features: bool,
        generate_query_document_ids: bool,
        documents_per_chunk: Optional[int] = None,
        top_x: Optional[int] = None,
        max_documents_per_query: Optional[int] = None,
        store_separate_lp_query_doc_features: bool = True,
        disjoint_query_chunk_mode: bool = False,
        drop0rel: bool = True,
        *,
        random_state: int,
        features: str,
        drop_feature_indices: Optional[Union[str, Sequence[int]]] = None,
        relevance: Union[Relevance, str],
        relevance_noise: float,
        relevance_quantization: bool,
    ):
        self.normalize_features = normalize_features
        self.generate_query_document_ids = generate_query_document_ids
        self.documents_per_chunk = documents_per_chunk
        self.top_x = top_x
        self.max_documents_per_query = max_documents_per_query
        self.store_separate_lp_query_doc_features = bool(store_separate_lp_query_doc_features)
        self.disjoint_query_chunk_mode = bool(disjoint_query_chunk_mode)
        self.drop0rel = bool(drop0rel)
        self.random_state = random_state
        self.features = features
        self.drop_feature_indices = self._parse_drop_feature_indices(drop_feature_indices)
        self.relevance = (
            Relevance(relevance) if isinstance(relevance, str) else relevance
        )
        self.relevance_noise = relevance_noise
        self.relevance_quantization = relevance_quantization

        if self.relevance == Relevance.LINEAR:
            self.relevance_fn = LinearRelevance(
                random_state=random_state,
                noise=relevance_noise,
            )
        elif self.relevance == Relevance.DEEP:
            self.relevance_fn = DeepRelevance(
                random_state=random_state,
                noise=relevance_noise,
            )

    def __call__(self, df: pd.DataFrame, *, split: Optional[str] = None) -> RatingDataset:
        cpu = jax.devices("cpu")[0]
        with jax.default_device(cpu):
            return self._preprocess(df, split=split)

    def _preprocess(self, df: pd.DataFrame, *, split: Optional[str] = None) -> RatingDataset:
        df = df.copy()
        if self.drop_feature_indices:
            df = self.drop_features(df)

        should_filter_nonrelevant_queries = self.drop0rel

        use_disjoint_chunks = self._use_disjoint_query_chunk_mode(split)
        if use_disjoint_chunks:
            if self.documents_per_chunk is None or int(self.documents_per_chunk) <= 0:
                raise ValueError(
                    "disjoint_query_chunk_mode=True requires documents_per_chunk to be a positive integer."
                )
            if should_filter_nonrelevant_queries:
                df = self._filter_queries_by_relevance(df, query_col="query")
            df = self._chunk_queries(df)
        else:
            if should_filter_nonrelevant_queries:
                df = self._filter_queries_by_relevance(df, query_col="query")

        if self.max_documents_per_query is not None:
            df = self._truncate_documents_per_query(df)

        if self.generate_query_document_ids:
            print("Generate unique query-document ids starting from 1, 2, ...")
            df["query_doc_id"] = np.arange(1, len(df) + 1)

        if self.normalize_features:
            print("Log-transform query-document features")
            df["query_doc_features"] = df["query_doc_features"].map(lambda x: log1p(x))

        # Keep the numerically stable float64 transform path, but store features as
        # float32 from this point onward to cut dataset memory use in half.
        df["query_doc_features"] = df["query_doc_features"].map(self._store_feature_array)

        # Converting long (query-doc per row) to wide (query with all docs/row) format:
        df = (
            df.groupby(["query"])
            .agg(
                labels=("label", list),
                query_doc_ids=("query_doc_id", list),
                query_doc_features=("query_doc_features", list),
            )
            .reset_index()
        )

        # Pad all queries to the same number of documents and mask padded documents:
        df["n"] = df["query_doc_ids"].map(len)
        max_n = df.n.max()
        print(f"Pad all queries to {max_n} docs")
        df["query_doc_ids"] = df["query_doc_ids"].map(lambda x: pad(x, max_n))
        df["query_doc_features"] = df["query_doc_features"].map(lambda x: pad(x, max_n))
        df["labels"] = df["labels"].map(lambda x: pad(x, max_n))
        df["mask"] = df["n"].map(np.ones).map(lambda x: pad(x, max_n).astype(bool))
        df["n"] = df["n"].map(lambda x: min(x, max_n))

        # Optionally generate new relevance labels:
        df = self.generate_labels(df)
        # Optionally select a subset of features:
        df, lp_features_differ = self.select_features(df)
        lp_query_doc_features = (
            np.stack(df["lp_query_doc_features"]).astype(self.FEATURE_DTYPE, copy=False)
            if self.store_separate_lp_query_doc_features or lp_features_differ
            else None
        )
        # Convert to PyTorch dataset:
        return RatingDataset(
            query=df["query"].values,
            query_doc_ids=np.stack(df["query_doc_ids"]),
            query_doc_features=np.stack(df["query_doc_features"]).astype(
                self.FEATURE_DTYPE,
                copy=False,
            ),
            lp_query_doc_features=lp_query_doc_features,
            labels=np.stack(df["labels"]),
            mask=np.stack(df["mask"]),
            n=df["n"].values,
        )


    def generate_labels(self, df: pd.DataFrame):
        query_document_features = np.stack(df["query_doc_features"])
        labels = None

        if self.relevance == Relevance.LINEAR:
            print(
                f"Generating linear relevance labels with {self.relevance_noise} noise"
            )
            labels = self.relevance_fn(query_document_features)
            labels = scale_relevance(labels)
        elif self.relevance == Relevance.DEEP:
            print(
                f"Generating non-linear relevance labels with {self.relevance_noise} noise"
            )
            labels = self.relevance_fn(query_document_features)
            labels = scale_relevance(labels)
        elif self.relevance == Relevance.ORIGINAL:
            print(
                f"Using relevance labels originally provided in the dataset, no noise applied"
            )
            labels = np.stack(df["labels"])

        if self.relevance_quantization:
            print(f"Rounding to labels nearest integer for quantized relevance")
            labels = np.round(labels)

        df["labels"] = list(labels)
        return df

    def _parse_drop_feature_indices(
        self,
        drop_feature_indices: Optional[Union[str, Sequence[int]]],
    ) -> list[int]:
        if drop_feature_indices is None:
            return []
        if isinstance(drop_feature_indices, str):
            values = [v.strip() for v in drop_feature_indices.split(",") if v.strip()]
            return sorted({int(v) for v in values})
        return sorted({int(v) for v in drop_feature_indices})

    def drop_features(self, df: pd.DataFrame) -> pd.DataFrame:
        first_features = np.asarray(df["query_doc_features"].iloc[0])
        total_features = int(first_features.shape[0])
        keep_drop = [f for f in self.drop_feature_indices if 0 <= f < total_features]
        if not keep_drop:
            return df

        print(
            f"Drop query-document feature columns before normalization: {keep_drop} "
            f"(total features before drop: {total_features})"
        )
        df["query_doc_features"] = df["query_doc_features"].map(
            lambda x: np.delete(np.asarray(x), keep_drop)
        )
        return df

    def _store_feature_array(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(x, dtype=self.FEATURE_DTYPE)

    def select_features(self, df):
        query_document_features = np.stack(df["query_doc_features"])
        total_features = query_document_features.shape[2]
        features = parse_feature_selection(self.features, total_features)
        lp_features_differ = not np.array_equal(
            np.asarray(features, dtype=np.int64),
            np.arange(total_features, dtype=np.int64),
        )
        print(
            f"Select query-document features for two-tower model: {self.features}, "
            f"{len(features)}/{total_features} available features"
        )

        # Keep all features for logging policy training:
        df["lp_query_doc_features"] = df["query_doc_features"]

        # Select subset of features for all downstream models:
        df["query_doc_features"] = df["query_doc_features"].map(
            lambda x: x[:, features]
        )

        return df, lp_features_differ

    def _use_disjoint_query_chunk_mode(self, split: Optional[str]) -> bool:
        if not self.disjoint_query_chunk_mode:
            return False
        if split is None:
            return False
        split_norm = str(split).strip().lower()
        return split_norm in {"train", "val", "validation"}

    def _filter_queries_by_relevance(self, df: pd.DataFrame, *, query_col: str) -> pd.DataFrame:
        query_df = df.groupby([query_col]).agg(max_label=("label", "max")).reset_index()
        keep_df = query_df[query_df.max_label >= 1]
        df = df[df[query_col].isin(keep_df[query_col])]
        dropped_queries = len(query_df) - len(keep_df)
        print(
            f"Dropped: {dropped_queries}/{len(query_df)} queries "
            f"without any relevant document"
        )
        return df

    def _chunk_queries(self, df: pd.DataFrame) -> pd.DataFrame:
        chunk_size = int(self.documents_per_chunk)
        print(
            f"Disjoint query-chunk mode: split each query into chunks of "
            f"at most {chunk_size} documents"
        )
        df = df.sort_values(by=["query", "label"], ascending=[True, False]).copy()
        rank_within_query = df.groupby(["query"]).cumcount()
        chunk_idx = rank_within_query // chunk_size

        query_chunk_keys = pd.MultiIndex.from_arrays(
            [df["query"].to_numpy(), chunk_idx.to_numpy()]
        )
        pseudo_query_ids, _ = pd.factorize(query_chunk_keys, sort=True)
        df["query"] = pseudo_query_ids.astype(np.int64)
        return df

    def _truncate_documents_per_query(self, df: pd.DataFrame) -> pd.DataFrame:
        max_docs = int(self.max_documents_per_query)
        if max_docs <= 0:
            raise ValueError(
                "max_documents_per_query must be > 0 when provided, "
                f"got {max_docs}."
            )

        original_len = len(df)
        df = df[df.groupby("query").cumcount() < max_docs].copy()
        removed = original_len - len(df)
        print(
            f"Truncate each query to at most {max_docs} available documents "
            f"(removed {removed} rows)"
        )
        return df


def scale_relevance(x, max_label: float = 4):
    lower, upper = np.percentile(x, [5, 95])
    x = np.clip(x, lower, upper)
    return (x - lower) / (upper - lower) * max_label


class LinearRelevance:
    def __init__(self, *, random_state: int, noise: float):
        self.noise = noise
        self.weights = None
        self.rngs = np.random.default_rng(random_state)

    def __call__(self, query_document_features: np.ndarray) -> np.ndarray:
        queries, documents, features = query_document_features.shape

        if self.weights is None:
            # Ensure subsequent calls to this function use the same weights:
            self.weights = self.rngs.standard_normal(features)

        scores = query_document_features.dot(self.weights)
        noise = self.noise * self.rngs.standard_normal(scores.shape)

        return scores + noise


class DeepRelevance:
    def __init__(self, hidden_units=16, *, random_state: int, noise: float):
        self.noise = noise
        self.hidden_units = hidden_units
        self.rngs = np.random.default_rng(random_state)
        self.W1 = None
        self.b1 = None
        self.W2 = None
        self.b2 = None

    def __call__(self, query_document_features: np.ndarray) -> np.ndarray:
        queries, documents, features = query_document_features.shape

        if self.W1 is None:
            # Ensure subsequent calls to this function use the same weights:
            self.W1 = self.rngs.standard_normal((features, self.hidden_units))
            self.b1 = self.rngs.standard_normal(self.hidden_units)
            self.W2 = self.rngs.standard_normal(self.hidden_units)
            self.b2 = self.rngs.standard_normal()

        hidden = np.tanh(query_document_features.dot(self.W1) + self.b1)
        scores = hidden.dot(self.W2) + self.b2
        noise = self.noise * self.rngs.standard_normal(scores.shape)

        return scores + noise
