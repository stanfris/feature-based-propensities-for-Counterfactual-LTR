import numpy as np
from torch.utils.data import Dataset

from feature_based_propensities_for_ULTR.data.collate import stack_dict_collate


class RatingDataset(Dataset):

    def __init__(
        self,
        query: np.ndarray,
        query_doc_ids: np.ndarray,
        query_doc_features: np.ndarray,
        lp_query_doc_features: np.ndarray | None,
        labels: np.ndarray,
        mask: np.ndarray,
        n: np.ndarray,
    ):
        self.query = query
        self.query_doc_ids = query_doc_ids
        self.query_doc_features = query_doc_features
        self._lp_query_doc_features = lp_query_doc_features
        self.labels = labels
        self.mask = mask
        self.n = n

    def __getitem__(self, idx):
        return {
            "query": self.query[idx],
            "query_doc_ids": self.query_doc_ids[idx],
            "query_doc_features": self.query_doc_features[idx],
            "lp_query_doc_features": self.lp_query_doc_features[idx],
            "labels": self.labels[idx],
            "mask": self.mask[idx],
            "n": self.n[idx],
        }

    def __len__(self):
        return len(self.query)

    @staticmethod
    def collate_fn(batch):
        return stack_dict_collate(batch)

    @property
    def n_queries(self) -> int:
        return self.query_doc_features.shape[0]

    @property
    def n_positions(self) -> int:
        return self.query_doc_features.shape[1]

    @property
    def n_features(self) -> int:
        return self.query_doc_features.shape[2]

    @property
    def n_logging_policy_features(self) -> int:
        return self.lp_query_doc_features.shape[2]

    @property
    def n_documents(self) -> int:
        return self.query_doc_ids.max() + 1

    @property
    def lp_query_doc_features(self) -> np.ndarray:
        if self._lp_query_doc_features is None:
            return self.query_doc_features
        return self._lp_query_doc_features

    @property
    def has_separate_lp_query_doc_features(self) -> bool:
        return self._lp_query_doc_features is not None
