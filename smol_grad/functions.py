import math

from smol_grad.tensors import Element, Vector, Matrix

def cross_entropy_loss(y: Vector, y_hat: Vector) -> Element:
    return Element(-sum([y_i.value * math.log(y_hat_i.value) for y_i, y_hat_i in zip(y, y_hat)]))

def relu(x: Element | Vector) -> Element | Vector: 
    if isinstance(x, Element): return x if x > 0 else Element(0.0)
    elif isinstance(x, Vector): return Vector([relu(v) for v in x])
def relu_derivative(a: Vector, y: Vector) -> Vector: return Vector([1 if x > 0 else 0.0 for x in a])


def sigmoid(x: Element): 
    if isinstance(x, Element): return Element(1 / (1 + (math.e ** -x.value)))
    elif isinstance(x, Vector): return Vector([sigmoid(v) for v in x])
def sigmoid_derivative(a: Vector, y: Vector) -> Vector: return a * (1-a)

def softmax(V: Vector) -> float: _sum=sum([math.e**v.value for v in V]); return Vector([(math.e**v.value)/_sum for v in V])
def softmax_derivative(a: Vector, y: Vector) -> Vector: return a - y