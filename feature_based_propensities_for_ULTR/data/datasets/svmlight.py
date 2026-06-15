import pickle
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.datasets import load_svmlight_file

from feature_based_propensities_for_ULTR.data.utils.file import verify_file, unarchive


class SVMLightDataSet:
    def __init__(
        self,
        name: str,
        zip_file: str | None,
        file: str,
        checksum: str | None,
        fold_split_map: Dict[int, Dict[str, str]],
        base_dir: Path,
        source_mode: str = "archive",
    ):
        self.name = name
        self.zip_file = zip_file
        self.file = file
        self.checksum = checksum
        self.fold_split_map = fold_split_map
        self.base_dir = Path(base_dir).expanduser()
        self.source_mode = str(source_mode).strip().lower()

    @property
    def dataset_directory(self):
        path = self.base_dir / "dataset"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def download_directory(self):
        path = self.base_dir / "download"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def cache_directory(self):
        path = self.base_dir / "cache"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def load(self, split: str, fold: int = 1) -> pd.DataFrame:
        """
        Parses and caches a LTR dataset in svmlight format to a pandas DataFrame
        in long format with one row query query-document pair.

        Please place any dataset in its original .ZIP in the following directory:
        ~/my/base/directory/download/
        """
        cache_path = self.cache_directory / f"{self.name}-{fold}-{split}.pckl"
        print(f"\nLoading: {self.name}, fold: {fold}, split: {split}")

        if not cache_path.exists():
            file_path = self._resolve_split_file(split=split, fold=fold)
            df = self._parse_svmlight(file_path)
            pickle.dump(df, open(cache_path, "wb"))

        return pickle.load(open(cache_path, "rb"))

    def _resolve_split_file(self, *, split: str, fold: int) -> Path:
        if fold not in self.fold_split_map:
            known_folds = sorted(self.fold_split_map.keys())
            raise ValueError(f"Unknown fold '{fold}' for dataset '{self.name}'. Known folds: {known_folds}")
        if split not in self.fold_split_map[fold]:
            known_splits = sorted(self.fold_split_map[fold].keys())
            raise ValueError(
                f"Unknown split '{split}' for dataset '{self.name}' fold '{fold}'. Known splits: {known_splits}"
            )

        rel_path = self.fold_split_map[fold][split]
        if self.source_mode == "archive":
            if not self.zip_file:
                raise ValueError(f"Dataset '{self.name}' in archive mode requires 'zip_file'.")
            if self.checksum is None:
                raise ValueError(f"Dataset '{self.name}' in archive mode requires 'checksum'.")

            zip_path = self.download_directory / self.zip_file
            verify_file(zip_path, self.checksum)
            archive_path = unarchive(zip_path, self.dataset_directory / self.file)
            split_path = archive_path / rel_path
        elif self.source_mode == "directory":
            split_path = self.dataset_directory / self.file / rel_path
        else:
            raise ValueError(
                f"Unsupported source_mode '{self.source_mode}' for dataset '{self.name}'. "
                "Expected 'archive' or 'directory'."
            )

        if not split_path.exists():
            raise FileNotFoundError(
                f"Could not find split file for dataset '{self.name}': {split_path}. "
                f"source_mode='{self.source_mode}', split='{split}', fold='{fold}'."
            )
        return split_path

    def _parse_svmlight(self, path: Path) -> pd.DataFrame:
        print(f"Parsing svmlight file: {path}")

        features, label, query = load_svmlight_file(str(path), query_id=True)
        features = np.asarray(features.todense())

        df = pd.DataFrame({"query_doc_features": list(features)})
        df["label"] = label
        df["query"] = query

        return df
