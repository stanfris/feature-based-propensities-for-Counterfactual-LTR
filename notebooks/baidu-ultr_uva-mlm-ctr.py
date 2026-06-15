from enum import Enum
from typing import List

import datasets
import pandas as pd

from datasets import Features, Value, Array2D, Sequence, SplitGenerator, Split


_CITATION = """\
@InProceedings{huggingface:dataset,
    title = {philipphager/baidu-ultr_uva-mlm-ctr},
    author={Philipp Hager, Romain Deffayet},
    year={2023}
}
"""

_DESCRIPTION = """\
Query-document vectors and clicks for a subset of the Baidu Unbiased Learning to Rank
dataset: https://arxiv.org/abs/2207.03051
This dataset uses a Jax-based BERT cross-encoder with 12 layers pre-trained for 2 million steps
on the Baidu ULTR dataset to create query-document embeddings (768 dims).
We link the model checkpoint also under `model/`. 
"""

_HOMEPAGE = "https://huggingface.co/datasets/philipphager/baidu-ultr_baidu-mlm-ctr/"
_LICENSE = "cc-by-nc-4.0"
_VERSION = "0.1.0"


class Config(str, Enum):
    ANNOTATIONS = "annotations"
    CLICKS = "clicks"


class BaiduUltrBuilder(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version(_VERSION)
    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name=Config.CLICKS,
            version=VERSION,
            description="Load train/val/test clicks from the Baidu ULTR dataset",
        ),
        datasets.BuilderConfig(
            name=Config.ANNOTATIONS,
            version=VERSION,
            description="Load expert annotations from the Baidu ULTR dataset",
        ),
    ]

    CLICK_FEATURES = Features(
        {
            ### Query features
            "query_id": Value("string"),
            "query_md5": Value("string"),
            "query": Sequence(Value("int32")),
            "query_length": Value("int32"),
            "n": Value("int32"),
            ### Doc features
            "url_md5": Sequence(Value("string")),
            "text_md5": Sequence(Value("string")),
            "title": Sequence(Sequence(Value("int32"))),
            "abstract": Sequence(Sequence(Value("int32"))),
            "query_document_embedding": Array2D((None, 768), "float16"),
            "click": Sequence(Value("int32")),
            ### SERP features
            "position": Sequence(Value("int32")),
            "media_type": Sequence(Value("int32")),
            "displayed_time": Sequence(Value("float32")),
            "serp_height": Sequence(Value("int32")),
            "slipoff_count_after_click": Sequence(Value("int32")),
            ### LTR features
            "bm25": Sequence(Value("float32")),
            "bm25_title": Sequence(Value("float32")),
            "bm25_abstract": Sequence(Value("float32")),
            "tf_idf": Sequence(Value("float32")),
            "tf": Sequence(Value("float32")),
            "idf": Sequence(Value("float32")),
            "ql_jelinek_mercer_short": Sequence(Value("float32")),
            "ql_jelinek_mercer_long": Sequence(Value("float32")),
            "ql_dirichlet": Sequence(Value("float32")),
            "document_length": Sequence(Value("int32")),
            "title_length": Sequence(Value("int32")),
            "abstract_length": Sequence(Value("int32")),
        }
    )

    ANNOTATION_FEATURES = Features(
        {
            ### Query features
            "query_id": Value("string"),
            "query_md5": Value("string"),
            "query": Sequence(Value("int32")),
            "query_length": Value("int32"),
            "frequency_bucket": Value("int32"),
            "n": Value("int32"),
            ### Doc features
            "text_md5": Sequence(Value("string")),
            "title": Sequence(Sequence(Value("int32"))),
            "abstract": Sequence(Sequence(Value("int32"))),
            "query_document_embedding": Array2D((None, 768), "float16"),
            "label": Sequence(Value("int32")),
            ### LTR features
            "bm25": Sequence(Value("float32")),
            "bm25_title": Sequence(Value("float32")),
            "bm25_abstract": Sequence(Value("float32")),
            "tf_idf": Sequence(Value("float32")),
            "tf": Sequence(Value("float32")),
            "idf": Sequence(Value("float32")),
            "ql_jelinek_mercer_short": Sequence(Value("float32")),
            "ql_jelinek_mercer_long": Sequence(Value("float32")),
            "ql_dirichlet": Sequence(Value("float32")),
            "document_length": Sequence(Value("int32")),
            "title_length": Sequence(Value("int32")),
            "abstract_length": Sequence(Value("int32")),
        }
    )

    DEFAULT_CONFIG_NAME = Config.CLICKS

    def _info(self):
        if self.config.name == Config.CLICKS:
            features = self.CLICK_FEATURES
        elif self.config.name == Config.ANNOTATIONS:
            features = self.ANNOTATION_FEATURES
        else:
            raise ValueError(
                f"Config {self.config.name} must be in ['clicks', 'annotations']"
            )

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        if self.config.name == Config.CLICKS:
            train_files = self.download_clicks(dl_manager, parts=[1, 2, 3])
            test_files = self.download_clicks(dl_manager, parts=[0])

            query_columns = [
                "query_id",
                "query_md5",
                "query",
                "query_length",
            ]

            agg_columns = [
                "url_md5",
                "text_md5",
                "title",
                "abstract",
                "query_document_embedding",
                "click",
                "position",
                "media_type",
                "displayed_time",
                "serp_height",
                "slipoff_count_after_click",
                "bm25",
                "bm25_title",
                "bm25_abstract",
                "tf_idf",
                "tf",
                "idf",
                "ql_jelinek_mercer_short",
                "ql_jelinek_mercer_long",
                "ql_dirichlet",
                "document_length",
                "title_length",
                "abstract_length",
            ]

            return [
                SplitGenerator(
                    name=Split.TRAIN,
                    gen_kwargs={
                        "files": train_files,
                        "query_columns": query_columns,
                        "agg_columns": agg_columns,
                    },
                ),
                SplitGenerator(
                    name=Split.TEST,
                    gen_kwargs={
                        "files": test_files,
                        "query_columns": query_columns,
                        "agg_columns": agg_columns,
                    },
                ),
            ]
        elif self.config.name == Config.ANNOTATIONS:
            test_files = dl_manager.download(["parts/validation.feather"])
            query_columns = [
                "query_id",
                "query_md5",
                "query",
                "query_length",
                "frequency_bucket",
            ]
            agg_columns = [
                "text_md5",
                "title",
                "abstract",
                "query_document_embedding",
                "label",
                "bm25",
                "bm25_title",
                "bm25_abstract",
                "tf_idf",
                "tf",
                "idf",
                "ql_jelinek_mercer_short",
                "ql_jelinek_mercer_long",
                "ql_dirichlet",
                "document_length",
                "title_length",
                "abstract_length",
            ]

            return [
                SplitGenerator(
                    name=Split.TEST,
                    gen_kwargs={
                        "files": test_files,
                        "query_columns": query_columns,
                        "agg_columns": agg_columns,
                    },
                )
            ]
        else:
            raise ValueError("Config name must be in ['clicks', 'annotations']")

    def download_clicks(self, dl_manager, parts: List[int], splits_per_part: int = 10):
        urls = [
            f"parts/part-{p}_split-{s}.feather"
            for p in parts
            for s in range(splits_per_part)
        ]

        return dl_manager.download(urls)

    def _generate_examples(
        self,
        files: List[str],
        query_columns: List[str],
        agg_columns: List[str],
    ):
        """
        Reads dataset partitions and aggregates document features per query.
        :param files: List of .feather files to load from disk.
        :param query_columns: Columns with one value per query. E.g., query_id,
        frequency bucket, etc.
        :param agg_columns: Columns with one value per document that should be
        aggregated per query. E.g., click, position, query_document_embeddings, etc.
        :return:
        """
        
        for file in files:
            df = pd.read_feather(file)
            current_query = None
            sample_key = None
            sample = None

            for i in range(len(df)):
                row = df.iloc[i]

                if current_query != row["query_no"]:
                    if current_query is not None:
                        yield sample_key, sample

                    current_query = row["query_no"]
                    # Use the query number in sample_key as query_ids are not unique.
                    sample_key = f"{file}-{current_query}"
                    sample = {"n": 0}

                    for column in query_columns:
                        sample[column] = row[column]
                    for column in agg_columns:
                        sample[column] = []

                for column in agg_columns:
                    sample[column].append(row[column])

                sample["n"] += 1

            yield sample_key, sample