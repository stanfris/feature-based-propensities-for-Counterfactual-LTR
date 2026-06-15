from pathlib import Path

from feature_based_propensities_for_ULTR.data.datasets.svmlight import SVMLightDataSet


class MSLR10K(SVMLightDataSet):
    name = "mslr10k"
    zip_file = "MSLR-WEB10K.zip"
    file = "MSLR-WEB10K"
    checksum = "2902142ea33f18c59414f654212de5063033b707d5c3939556124b1120d3a0ba"
    fold_split_map = {
        1: {
            "train": "Fold1/train.txt",
            "val": "Fold1/vali.txt",
            "test": "Fold1/test.txt",
        },
        2: {
            "train": "Fold2/train.txt",
            "val": "Fold2/vali.txt",
            "test": "Fold2/test.txt",
        },
        3: {
            "train": "Fold3/train.txt",
            "val": "Fold3/vali.txt",
            "test": "Fold3/test.txt",
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


class MSLR30K(SVMLightDataSet):
    name = "mslr30k"
    zip_file = "MSLR-WEB30K.zip"
    file = "MSLR-WEB30K"
    checksum = "08cb7977e1d5cbdeb57a9a2537a0923dbca6d46a76db9a6afc69e043c85341ae"
    fold_split_map = {
        1: {
            "train": "Fold1/train.txt",
            "val": "Fold1/vali.txt",
            "test": "Fold1/test.txt",
        },
        2: {
            "train": "Fold2/train.txt",
            "val": "Fold2/vali.txt",
            "test": "Fold2/test.txt",
        },
        3: {
            "train": "Fold3/train.txt",
            "val": "Fold3/vali.txt",
            "test": "Fold3/test.txt",
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
