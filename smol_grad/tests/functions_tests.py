import unittest
import math

from smol_grad.tensors import Element, Vector, Matrix
from smol_grad.functions import relu, sigmoid, softmax


class TestRelu(unittest.TestCase):
    def test_element(self): self.assertEqual(relu(Element(5)), Element(5))
    def test_element(self): self.assertEqual(relu(Element(-5)), Element(0))
    def test_vector(self): self.assertEqual(relu(Vector([-1,6,5])), Vector([0,6,5]))

class TestSigmoid(unittest.TestCase):
    def test_element(self): self.assertAlmostEqual(sigmoid(Element(10)).value, 0.9999546021312976)
    def test_element_zero(self): self.assertEqual(sigmoid(Element(0)), Element(0.5))
    def test_element_negative(self): self.assertAlmostEqual(sigmoid(Element(-2)).value, 0.11920292202211755)
    def test_vector(self): self.assertTrue(all([math.isclose(v1.value, v2, rel_tol=1e-05) for v1, v2 in zip(sigmoid(Vector([10, 0, -2])), [0.9999546021312976, 0.5, 0.11920292202211755])]))

class TestSoftmax(unittest.TestCase):
    def test_vector(self): self.assertTrue(all([math.isclose(v1.value, v2, rel_tol=1e-05) for v1, v2 in zip(softmax(Vector([1, 0, -2])), [0.70538451, 0.25949646, 0.03511903])]))

        