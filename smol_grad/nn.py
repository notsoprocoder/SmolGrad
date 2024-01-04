import random
import math
from typing import Callable
from functools import reduce

from smol_grad.tensors import Element, Vector, Matrix
from smol_grad.io import MLDataObject
from smol_grad.functions import relu, sigmoid, softmax  

class Layer(object):
    def __init__(self, input_dims: int, output_dims: int, activation_fn: Callable, activation_derivative: Callable, learning_rate: float=0.01):
        self.W: Matrix = Matrix([[random.random() for j in range(input_dims)] for i in range(output_dims)])
        self.b: Vector = Vector([1 for i in range(output_dims)])
        self.learning_rate: float = learning_rate
        self.activation_fn: Callable = activation_fn
        self.activation_derivative: Callable = activation_derivative
        self.prev_a: Vector = None
        self.z: Vector = None
        self.final_layer: bool = False

    def forward(self, X: Vector):
        self.prev_a = X
        self.z=X*self.W + self.b
        return self.activation(self.z)
    
    def backward(self, error: Vector):
        if not self.final_layer:
            error = (self.W._transpose() * error) * self.activation_derivative(self.z)
        weight_grad = self.prev_a*error
        for ridx, r in enumerate(self.W):
            for eidx, e in enumerate(r):
                e.grad = weight_grad[ridx][eidx]
        for idx, b in enumerate(self.b): b.grad = error[idx]
        return error

    def step(self):
        for r in self.W: for e in r: e.value -= self.learning_rate * e.grad
        for b in self.b: b.value -= self.learning_rate * b.grad


class Model(object):
    def __init__(self, layers: list[Layer], cost_fn: Callable, cost_derivative:
                  Callable, learning_rate: float, batch_size: int=16):
        self.layers: list[Layer] = layers
        self.learning_rate: float = learning_rate
        self.true_learning_rate: float = learning_rate/batch_size
        self.batch_size: int = batch_size
        self.cost_fn: Callable = cost_fn
        self.cost_derivative: Callable = cost_derivative
        self.layers[-1].final_layer = True
    def inference(self, X): for l in self.layers[1:]: X = l.forward(X); return X

    def batch_update(self, X, y):
        for X, y in zip(X,y):
            y_hat = self.inference(X)
            error = self.cost_derivative(y, y_hat)

            for idx, l in enumerate(reversed(self.layers)):
                error = l.backward(error)
                l.step()


    def train(self, data, epochs=25, val_obs=250, val_at_n=5): 
        batch_indxs = list(zip(list(
            range(0, len(data.train.idx), self.batch_size)),
            range(self.batch_size, len(data.train.idx), self.batch_size)))
        batch_indxs.append((batch_indxs[-1][1], None))
        for epoch in epochs:
            for t in batch_idx:
                self.batch_update(data.train.X[t[0]: t[1]], data.train.y[t[0]: t[1]])

                if val_at_n:
                    loss = 0
                    if epoch%val_at_n:
                        for i in range(val_obs):
                            X, y = data.val.sample()
                            y_hat = self.inference(X)
                            loss += self.cost_fn(y, y_hat)
                        loss /= val_obs
                        print(f"Loss at epoch {epoch}:   {round(loss, 5)}")









    


 