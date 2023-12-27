import unittest

from smol_grad.io import MLDataObject

class TestMLDataObject(unittest.TestCase):
    def setUp(self):
        self.dataset = MLDataObject("mnist.csv") 
        
    def test_train_sample(self): self.assertEqual(type(self.dataset.train.sample()[0][1]), int)
        