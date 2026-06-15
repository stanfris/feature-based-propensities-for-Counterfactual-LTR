from pathlib import Path

from feature_based_propensities_for_ULTR.data.datasets.svmlight import SVMLightDataSet


class YahooC14(SVMLightDataSet):
    name = "yahoo-c14"
    zip_file = "ltrc_yahoo.tar.bz2"
    file = "ltrc_yahoo"
    checksum = "2d96b587828f5a4d43c508ba7ec3412a760b9be798f69313b38516544a33a6de"
    fold_split_map = {
        1: {
            "train": "set1.train.txt",
            "val": "set1.valid.txt",
            "test": "set1.test.txt",
        },
        2: {
            "train": "set2.train.txt",
            "val": "set2.valid.txt",
            "test": "set2.test.txt",
        },
    }

    def __init__(
        self,
        base_dir: Path,
    ):
        super().__init__(
            name=self.name,
            zip_file=self.zip_file,
            file=self.file,
            checksum=self.checksum,
            fold_split_map=self.fold_split_map,
            base_dir=base_dir,
        )
