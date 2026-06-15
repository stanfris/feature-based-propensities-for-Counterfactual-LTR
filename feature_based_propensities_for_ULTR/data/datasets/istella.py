from pathlib import Path

from feature_based_propensities_for_ULTR.data.datasets.svmlight import SVMLightDataSet


class IstellaSample(SVMLightDataSet):
    name = "istella"
    file = "istella-s-letor/sample"
    fold_split_map = {
        1: {
            "train": "train.txt",
            "val": "vali.txt",
            "test": "test.txt",
        }
    }

    def __init__(self, base_dir: Path):
        super().__init__(
            name=self.name,
            zip_file=None,
            file=self.file,
            checksum=None,
            fold_split_map=self.fold_split_map,
            base_dir=base_dir,
            source_mode="directory",
        )
