import random
import math
from typing import Callable

from smol_grad.tensors import Element, Vector, Matrix
from smol_grad.io import MLDataObject
from smol_grad.functions import relu, sigmoid, softmax  

class Layer(object):
    def __init__(self, input_dims: int, output_dims: int, activation_fn: Callable, activation_derivative: Callable, learning_rate: float=0.01):
        self.W: Matrix = Matrix([[random.random() for j in range(intput_dims)] for i in range(output_dims)])
        self.b: Vector = Vector([1 for i in range(output_dims)])
        self.learning_rate: float = learning_rate
        self.activation_fn: Callable = activation_fn
        self.activation_derivative: Callable = activation_derivative
        self.a: Vector = None
        self.z: Vector = None

    def forward(self, X: Vector):
        self.z=X*self.W + self.b
        self.a=self.activation(self.z)
        return self.a
    
    def update_grad(self, y: Vector, y_hat: Vector):
        for idx, r in enumerate(self.W): for e in r: e.grad=self.activation_derivative(e)
        for idx, b in enumerate(self.b): b.grad = self.activation_derivative(b)

    def step(self):
        for r in self.W: for e in r: e.value -= self.learning_rate * (e.grad+e.value)
        for b in self.b: b.value -= self.learning_rate * (b.grad+b.value)


 