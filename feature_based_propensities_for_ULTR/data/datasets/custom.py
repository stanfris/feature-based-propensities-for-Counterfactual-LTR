
from pathlib import Path

from feature_based_propensities_for_ULTR.data.datasets.svmlight import SVMLightDataSet
import numpy as np
import pandas as pd
from sklearn.datasets import load_svmlight_file
import pickle
from pathlib import Path
from typing import Dict

from feature_based_propensities_for_ULTR.data.utils.file import verify_file, unarchive

import os
import random
import shutil
import numpy as np

class CustomDataset(SVMLightDataSet):
    name = "custom_dataset"
    zip_file = "Custom_dataset.zip"
    file = "Custom_dataset"
    checksum = "08cb7977e1d5cbdeb57a9a2537a0923dbca6d46a76db9a6afc69e043c85341ae"
    fold_split_map = {
        1: {
            "train": "Fold1/train.txt",
            "val": "Fold1/vali.txt",
            "test": "Fold1/test.txt",
        },
    }

    def __init__(self, base_dir: Path):
        super().__init__(
            name=self.name,
            zip_file=self.zip_file,
            file=self.file,
            checksum=self.checksum,
            fold_split_map=self.fold_split_map,
            base_dir=base_dir,
        )


def create_custom_dataset(initial_path, filename, 
                          num_groups=1, docs_per_group=10, 
                          D=100, s_group=0.0, s_doc=0.0, 
                          random_seed=42, num_queries=10, label_type='deep'):
    """
    Generate a synthetic LTR-style dataset using a hierarchical Gaussian model
    and balanced quantile-based relevance labels (1–5).
    """
    rng = np.random.default_rng(random_seed)
    os.makedirs(initial_path, exist_ok=True)

    path = os.path.join(initial_path, filename)

    if label_type=='linear':
        all_scores, all_data = generate_linear_score_and_features(
            num_queries=num_queries,
            num_groups=num_groups,
            docs_per_group=docs_per_group,
            D=D,
            s_group=s_group,
            s_doc=s_doc,
            rng=rng
        )
    else:
        all_scores, all_data = generate_deep_score_and_features_overlap(
            num_queries=num_queries,
            num_groups=num_groups,
            docs_per_group=docs_per_group,
            D=D,
            s_group=s_group,
            s_doc=s_doc,
            rng=rng
        )

    # Write RankLib/LibSVM format
    with open(path, 'w') as f:
        for (score, (qid, features)) in zip(all_scores, all_data):
            label = score
            feature_str = ' '.join(f"{k}:{features[k]:.4f}" for k in range(D))
            f.write(f"{label} qid:{qid} {feature_str}\n")

    print(f"✅ Dataset written: {path}")

def generate_linear_score_and_features(num_queries, num_groups, docs_per_group, D, s_group, s_doc, rng):
    """
    Generate features and scores using a linear model with hierarchical Gaussian noise.
    """
    all_scores = []
    all_data = []

    for qid in range(num_queries):
        for grp_idx in range(num_groups):
            for doc_idx in range(docs_per_group):
                # Document-level features
                a = 0
                b = rng.uniform(0, 5)
                score = a + b 
                all_scores.append(score)
                all_data.append((qid, [a, b]))  # qid starts from 0

    return all_scores, all_data

def generate_deep_score_and_features(num_queries, num_groups, docs_per_group, D, s_group, s_doc, rng):
    """
    Generate features and scores using a deep learning model with hierarchical Gaussian noise.
    """
    all_scores = []
    all_data = []

    deep_model = DeepRelevance(hidden_units=16, random_state=rng, noise=0.0)

    for qid in range(num_queries):
        for grp_idx in range(num_groups):
            for doc_idx in range(docs_per_group):

                # Document-level features
                a = 0
                b = rng.uniform(0, 5)
                features = np.array([[a, b]])
                score = deep_model(features)[0]
                all_scores.append(score)
                all_data.append((qid, [a, b]))  # qid starts from 0

    return all_scores, all_data

def generate_deep_score_and_features_overlap(num_queries, num_groups, docs_per_group, D, s_group, s_doc, rng, test_set=False):
    """
    Generate features and scores using a deep learning model with hierarchical Gaussian noise.
    """
    all_scores = []
    all_data = []

    deep_model = DeepRelevance(hidden_units=[32, 32, 32], random_state=rng, noise=0.0)
    if not test_set:
        for qid in range(num_queries):
            for _ in range(num_groups):
                for doc_idx in range(docs_per_group):
                    a = 0
                    if doc_idx == 0:
                        b = rng.uniform(doc_idx, min(doc_idx + 1 + s_doc, 10))
                    elif doc_idx == docs_per_group - 1:
                        b = rng.uniform(max(0, doc_idx - s_doc), min(10, doc_idx + 1))
                    else:
                        b = rng.uniform(max(0, doc_idx - s_doc), min(10, doc_idx + 1 + s_doc))

                    # Document-level features
                    features = np.array([[a, b]])
                    score = deep_model(features)[0]
                    all_scores.append(score)
                    all_data.append((qid, [a, b]))  # qid starts from 0
    else:
        print("writing custom test dataset")
        b = 0
        for qid in range(num_queries):
            for _ in range(num_groups):
                for doc_idx in range(docs_per_group):
                    a = 0
                    b += 10/(num_queries)
                    # Document-level features
                    features = np.array([[a, b]])
                    score = deep_model(features)[0]
                    all_scores.append(score)
                    all_data.append((qid, [a, b]))  # qid starts from 0       

    return all_scores, all_data


class DeepRelevance:
    def __init__(self, hidden_units=[16, 8], *, random_state: int, noise: float):
        """
        Parameters
        ----------
        hidden_units : list[int]
            A list specifying the number of units in each hidden layer.
            Example: [32, 16, 8] creates 3 hidden layers.
        random_state : int
            Seed for reproducibility.
        noise : float
            Standard deviation of Gaussian noise added to output.
        """
        self.hidden_units = hidden_units
        self.noise = noise
        self.rng = np.random.default_rng(random_state)
        self.layers = []  # Will hold (W, b) tuples

    def __call__(self, query_document_features: np.ndarray) -> np.ndarray:
        n_docs, n_features = query_document_features.shape

        # Initialize weights only once
        if not self.layers:
            input_size = n_features
            for units in self.hidden_units:
                W = self.rng.standard_normal((input_size, units))
                b = self.rng.standard_normal(units)
                self.layers.append((W, b))
                input_size = units

            # Output layer
            W_out = self.rng.standard_normal(input_size)
            b_out = self.rng.standard_normal()
            self.output_layer = (W_out, b_out)

        # Forward pass
        hidden = query_document_features
        for (W, b) in self.layers:
            hidden = np.tanh(hidden.dot(W) + b)

        scores = hidden.dot(self.output_layer[0]) + self.output_layer[1]

        # Add noise
        noise = self.noise * self.rng.standard_normal(scores.shape)
        return scores + noise


def write_custom_dataset(initial_path, file, data, zip_path,
                         num_groups=1, docs_per_group=10, 
                         D=100, s_group=0.5, s_doc=0.5, 
                         random_seed=42, num_queries=10, label_type='deep'):
    """
    Generate train/val/test splits and zip them into a single dataset archive.
    """
    # Create Folds directory
    fold_dir = Path(initial_path) / "Fold1"
    os.makedirs(fold_dir, exist_ok=True)

    print(f"⚙️ Creating dataset splits in: {fold_dir}")

    for split_name in ["train.txt", "vali.txt", "test.txt"]:
        create_custom_dataset(
            initial_path=fold_dir,
            filename=split_name,
            num_groups=num_groups,
            docs_per_group=docs_per_group,
            D=D,
            s_group=s_group,
            s_doc=s_doc,
            random_seed=random_seed,
            num_queries=num_queries,
            label_type=label_type
        )
    # Zip everything
    zip_path = Path(zip_path)
    os.makedirs(zip_path.parent, exist_ok=True)

    output_filename = zip_path.stem
    print(f"📦 Zipping dataset from {initial_path} → {zip_path}")
    shutil.make_archive(str(zip_path.with_suffix('')), 'zip', root_dir=initial_path)

    print(f"✅ Dataset zipped successfully at: {zip_path}")


# ==========================================================
# === Dataset Parser / Loader ===
# ==========================================================
class CustomDatasetDeep_Parser:
    def __init__(
        self,
        name: str,
        zip_file: str,
        file: str,
        checksum: str,
        fold_split_map: Dict[int, Dict[str, str]],
        base_dir: Path,
        dataset_params: Dict = None,
    ):
        self.base_dir = Path(base_dir).expanduser()
        self.dataset_params = dataset_params or {}

        # --- Generate parameter-specific suffix ---
        param_suffix = self._param_suffix(self.dataset_params)

        # --- Dynamic naming ---
        self.name = f"{name}-{param_suffix}" if param_suffix else name
        self.file = f"{file}-{param_suffix}" if param_suffix else file
        self.zip_file = f"{Path(zip_file).stem}-{param_suffix}.zip" if param_suffix else zip_file
        self.checksum = checksum
        self.fold_split_map = fold_split_map

        print(f"📁 Initialized dataset: {self.name}")

    # --- Helper for consistent param-based naming ---
    def _param_suffix(self, params: Dict) -> str:
        if not params:
            return ""
        parts = [
            f"num_groups{params.get('num_groups', 1)}",
            f"docs{params.get('docs_per_group', 10)}",
            f"D{params.get('D', 100)}",
            f"sgroup{params.get('s_group', 0.5)}",
            f"sdoc{params.get('s_doc', 0.5)}",
            f"seed{params.get('random_seed', 42)}",
            f"num_queries{params.get('num_queries', 10)}",
            f"labeltype{params.get('label_type', 'deep')}",
        ]
        print(parts)
        return "_".join(parts)

    # --- Directory helpers ---
    @property
    def dataset_directory(self):
        path = self.base_dir / "dataset" / self.file
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def download_directory(self):
        path = self.base_dir / "download"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def cache_directory(self):
        # parameter-specific cache folder
        path = self.base_dir / "cache" / self.file
        path.mkdir(parents=True, exist_ok=True)
        return path

    # --- Generation ---
    def _generate_dataset_if_needed(self):
        zip_path = self.download_directory / self.zip_file
        if zip_path.exists():
            print(f"📦 Found existing dataset zip: {zip_path}")
            return

        print("⚙️ Generating custom dataset...")
        write_custom_dataset(
            initial_path=self.dataset_directory,
            file=self.file,
            data=None,
            zip_path=zip_path,
            **self.dataset_params,
        )
        print(f"✅ Generated dataset: {zip_path}")

    # --- Loading ---
    def load(self, split: str, fold: int = 1) -> pd.DataFrame:
        cache_path = self.cache_directory / f"{self.name}-{fold}-{split}.pckl"
        print(f"\nLoading dataset: {self.name}, fold: {fold}, split: {split}")

        if not cache_path.exists():
            self._generate_dataset_if_needed()

            zip_path = self.download_directory / self.zip_file
            archive_dir = self.dataset_directory
            shutil.unpack_archive(str(zip_path), str(archive_dir))

            file_path = archive_dir / self.fold_split_map[fold][split]
            df = self._parse_svmlight(file_path)
            pickle.dump(df, open(cache_path, "wb"))

        return pickle.load(open(cache_path, "rb"))

    def _parse_svmlight(self, path: Path) -> pd.DataFrame:
        print(f"Parsing svmlight file: {path}")
        features, label, query = load_svmlight_file(str(path), query_id=True)
        features = np.asarray(features.todense())
        df = pd.DataFrame({"query_doc_features": list(features)})
        df["label"] = label
        df["query"] = query
        return df
    

class CustomDatasetDeep(CustomDatasetDeep_Parser):
    name = "custom_dataset_deep"
    zip_file = "Custom_dataset_deep.zip"
    file = "Custom_dataset_deep"
    checksum = "08cb7977e1d5cbdeb57a9a2537a0923dbca6d46a76db9a6afc69e043c85341ae"
    fold_split_map = {
        1: {
            "train": "Fold1/train.txt",
            "val": "Fold1/vali.txt",
            "test": "Fold1/test.txt",
        },
    }

    def __init__(self, base_dir: Path, **dataset_params):
        super().__init__(
            name=self.name,
            zip_file=self.zip_file,
            file=self.file,
            checksum=self.checksum,
            fold_split_map=self.fold_split_map,
            base_dir=base_dir,
            dataset_params=dataset_params,
        )
