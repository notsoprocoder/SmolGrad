import unittest

from smol_grad.io import MLDataObject
from smol_grad.tensors import Vector

class TestMLDataObject(unittest.TestCase):
    def setUp(self):self.dataset = MLDataObject("mnist.csv") 
    def test_train_sample_y(self): self.assertEqual(type(self.dataset.train.sample()[0][1]), int)
    def test_train_sample_X(self): self.assertEqual(type(self.dataset.train.sample()[0][0]), Vector)
    def test_test_sample_y(self): self.assertEqual(type(self.dataset.train.sample()[0][1]), int)
    def test_test_sample_X(self): self.assertEqual(type(self.dataset.train.sample()[0][0]), Vector)
    def test_val_sample_y(self): self.assertEqual(type(self.dataset.train.sample()[0][1]), int)
    def test_val_sample_X(self): self.assertEqual(type(self.dataset.train.sample()[0][0]), Vector)
        
class TestDataset(unittest.TestCase):
    def setUp(self): self.dataset = MLDataObject("mnist.csv") 
    def test_sample_y(self): self.assertEqual(type(self.dataset.train.sample()[0][1]), int)
    def test_sample_X(self): self.assertEqual(type(self.dataset.train.sample()[0][0]), Vector)
    def test_getcol_type(self): self.assertEqual(type(self.dataset.train._getcol("7x7")), Vector)
    def test_getcol_len(self): self.assertEqual(len(self.dataset.train._getcol("7x7")), 7500)
    def test_getitem_int_y(self): self.assertEqual(type(self.dataset.train[0][0]), Vector)
    def test_getitem_int_X(self): self.assertEqual(type(self.dataset.train[0][1]), int)
    def tets_getitem_X_len(self): self.assertEqual(len(self.dataset.train[0][1]), 784)
    
    
