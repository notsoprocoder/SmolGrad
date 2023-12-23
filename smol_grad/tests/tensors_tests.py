import unittest

from smol_grad.tensors import Element, Vector, Matrix

class TestValue(unittest.TestCase):
    def test_value_int_add(self):
        self.assertEqual(Element(60)+Element(40), Element(100))
    def test_value_float_add(self):
        self.assertEqual(Element(6.6)+Element(2.9), Element(9.5))
    def test_value_int_radd(self):
        self.assertEqual(Element(40)+Element(60), Element(100))
    def test_value_float_radd(self):
        self.assertEqual(Element(2.9) + Element(6.6), Element(9.5))
    def test_int_add(self):
        self.assertEqual(Element(60)+40, Element(100))
    def test_float_add(self):
        self.assertEqual(6.6+Element(2.9), Element(9.5))
    def test_int_radd(self):
        self.assertEqual(40+Element(60), Element(100))
    def test_float_radd(self):
        self.assertEqual(Element(2.9) + 6.6, Element(9.5))
    def test_vector_add(self):
        self.assertEqual(Element(5)+Vector([1, 10, 29]), Vector([6, 15, 34]))
    def test_vector_radd(self):
        self.assertEqual(Vector([1, 10, 29])+Element(5), Vector([6, 15, 34]))
    def test_matrix_add(self):
        self.assertEqual(Element(5)+Matrix([[1, 10, 29], [1, 10, 29]]), Matrix([[6, 15, 34],[6, 15, 34]]))
    def test_value_int_mul(self):
        self.assertEqual(Element(5)*Element(5), Element(25))
    def test_value_float_mul(self):
        self.assertEqual(Element(5.4)*Element(.5), Element(2.7))
    def test_int_mul(self):
        self.assertEqual(Element(5)*5, Element(25))
    def test_int_rmul(self):
        self.assertEqual(5*Element(5), Element(25))
    def test_float_mul(self):
        self.assertEqual(Element(5.4)*.5, Element(2.7))
    def test_float_rmul(self):
        self.assertEqual(.5*Element(5.4), Element(2.7))
    def test_vector_mul(self):
        self.assertEqual(Element(2)*Vector([1,2,3]), Vector([2, 4, 6]))
    def test_matrix_mul(self):
        self.assertEqual(Element(2)*Matrix([[1,2,3], [1,2,3]]), Matrix([[2,4,6], [2,4,6]]))
    

class TestVector(unittest.TestCase):
    def test_get_item(self): self.assertEqual(Vector([1,2,3])[1], Element(2))
    def test_add(self):
        self.assertEqual(Vector([1, 3, 4])+Vector([1, 3, 4]), Vector([2, 6, 8]))
    def test_int_add(self):
        self.assertEqual(Vector([9, 10, 11])+5, Vector([14, 15, 16]))
    def test_value_add(self):
        self.assertEqual(Vector([9, 10, 11])+Element(5), Vector([14, 15, 16]))
    def test_mul(self):
        self.assertEqual(Vector([4,5,6])*Vector([4, 5, 6]), Vector([16, 25, 36]))
    def test_value_mul(self):
        self.assertEqual(Vector([4,5,6])*Element(2), Vector([8, 10, 12]))
    def test_matrix_mul(self):
        self.assertEqual(Vector([1,1])*Matrix([[1, 2, 3], [4, 5, 6]]), Vector([6, 15]))



class TestMatrix(unittest.TestCase):
    def test_matrix_get_item(self):
        self.assertEqual(Matrix([[7,8],[9,10], [11, 12]])[2], Vector([11,12]))
    def test_add(self):
        self.assertEqual(Matrix([[1, 10, 29], [1, 10, 29]])+Element(5), Matrix([[6, 15, 34],[6, 15, 34]]))
    def test_value_add(self):
        self.assertEqual(Matrix([[1, 2, 3], [4, 5, 6]])+Element(10), Matrix([[11, 12, 13], [14, 15, 16]]))
    def test_vector_mul(self):
        self.assertEqual(Matrix([[1, 2, 3], [4, 5, 6]]) * Vector([1, 1, 1]), Vector([5, 7, 9]))
    def test_matrix_mul(self):
        self.assertEqual(Matrix([[1, 2, 3], [4, 5, 6]]) * Matrix([[7,8],[9,10], [11, 12]]), Matrix([[58, 64], [139, 154]]))
    def test_matrix_mul_commutative(self):
        self.assertEqual(Matrix([[7,8],[9,10], [11, 12]])*Matrix([[1, 2, 3], [4, 5, 6]]), Matrix([[39, 54, 69], [49, 68, 87], [59, 82, 105]]))
