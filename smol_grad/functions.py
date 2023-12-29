import math

from smol_grad.tensors import Element, Vector

def relu(x: Element | Vector) -> float: 
    if isinstance(x, Element): return x if x > 0 else Element(0.0)
    elif isinstance(x, Vector): return Vector([relu(v) for v in x])

def sigmoid(x: Element): 
    if isinstance(x, Element): return Element(1 / (1 + (math.e ** -x.value)))
    elif isinstance(x, Vector): return Vector([sigmoid(v) for v in x])

def softmax(V: Vector) -> float: _sum=sum([math.e**v.value for v in V]); return Vector([(math.e**v.value)/_sum for v in V])
