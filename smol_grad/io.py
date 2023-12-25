class CSVDataSet(object):
    def __init__(self, fpath: str, train_split: float=.75, test_split: float=0.125):
        self.data = fpath