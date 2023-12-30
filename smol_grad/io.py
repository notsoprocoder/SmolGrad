import random
import csv

from smol_grad.tensors import Vector

class Dataset(object):
    def __init__(self, cols: list, X: list[Vector], y: list):
        self.cols = cols
        self.idx: list = list(range(len(X)))
        self.X: list[Vector] = X
        self.y: list = y
    def __getitem__(self, k: int | str): return (self.X[k], self.y[k])
    def _getcol(self, k: str) -> Vector: idx=self.cols.index(k); return Vector([v[idx] for v in self.X])
    def sample(self, n: int = 1): 
        _idx = random.sample(self.idx, k=n)
        return [self[i] for i in _idx]

class MLDataObject(object):
    def __init__(self, fpath: str, y: str="label", has_cols: bool = True, train_split: float=.75, test_split: float=0.125):
        self.has_cols: bool = has_cols
        self.y_name: str = y
        with open(fpath, newline='') as csvfile:
            csv_file = list(csv.reader(csvfile))
            self._cols= list(csv_file[0]) if self.has_cols else None
            self.y_idx: str = self._cols.index(self.y_name) if self.has_cols else None
            self.y = [int(V.pop(self.y_idx)) for V in csv_file[1:]]
            self.X = [Vector([float(v) for v in V]) for V in csv_file[1:]]
            self.shape = (len(self.X[0]), len(self.X))
            csvfile.close()
        # training splits
        idx = set(range(self.shape[1]))
        val_idx = set(random.sample(list(idx), k=int((1-train_split-test_split)*self.shape[1])))
        test_idx = set(random.sample(list(idx-val_idx), k=int(test_split*self.shape[1])))
        train_idx = set(idx - val_idx - test_idx)
        self.train = Dataset(self._cols, [self.X[i] for i in train_idx], [self.y[i] for i in train_idx])
        self.test = Dataset(self._cols, [self.X[i] for i in test_idx], [self.y[i] for i in test_idx])
        self.val = Dataset(self._cols, [self.X[i] for i in val_idx], [self.y[i] for i in val_idx])

        