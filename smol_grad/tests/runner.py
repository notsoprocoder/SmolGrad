import unittest

from smol_grad.tests.tensors_tests import TestValue, TestVector, TestMatrix
from smol_grad.tests.io_tests import TestMLDataObject
from smol_grad.tests.functions_tests import TestRelu, TestSigmoid, TestSoftmax

if __name__ == '__main__':
    unittest.main()