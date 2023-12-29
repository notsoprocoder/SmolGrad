from __future__ import annotations
from typing import Iterator

class Element(object):
    def __init__(self, value: float | int):
        if isinstance(value, Element): self.value = value.value
        elif isinstance(value, bool): self.value = int(bool)
        else: self.value = value
    def __repr__(self) -> str: return str(self.value)
    def __sub__(self, other: Element) -> Element: return self.__add__(Element(-1)*other)
    def __radd__(self, other: Element) -> Element: return self.__add__(other)
    def __rmul__(self, other: Element | Vector | Matrix) -> Element | Vector | Matrix: return self.__mul__(other)

    def __add__(self, other: Element) -> Element | Vector | Matrix:
        if isinstance(other, (int, float)): return Element(self.value + other)
        elif isinstance(other, Element): return Element(self.value + other.value)        
        elif isinstance(other, Vector): return Vector([self.value+v for v in other])
        elif isinstance(other, Matrix): return Matrix([self+r for r in other])
        raise TypeError(f"{type(other)} cannot use addition operator with Element.") 
    
    def __mul__(self, other: Element) -> Element | Vector | Matrix:
        if isinstance(other, (int, float)): return Element(self.value*other)
        elif isinstance(other, Element): return Element(self.value*other.value)
        elif isinstance(other, Vector): return Vector([self.value*v for v in other])
        elif isinstance(other, Matrix): return Matrix([self*r for r in other])
        raise TypeError(f"{type(other)} cannot use mul operator with Element.") 
    
    def __eq__(self, other: Element | float | int) -> bool:
        if isinstance(other, Element): return Element(other).value == self.value 
        return False
    
    def __gt__(self, other: Element) -> Element | Vector | Matrix:
        if isinstance(other, (int, float)): return self.value > other
        elif isinstance(other, Element): return self.value > other.value 
        elif isinstance(other, Vector): return Vector([self.value > v.value for v in other])
        elif isinstance(other, Matrix): return Matrix([self>r for r in other])
        raise TypeError(f"{type(other)} cannot use addition operator with Element.")
    
    def __lt__(self, other: Element) -> Element | Vector | Matrix:
        if isinstance(other, (int, float)): return self.value < other
        elif isinstance(other, Element): return self.value < other.value
        elif isinstance(other, Vector): return Vector([self.value < v.value for v in other])
        elif isinstance(other, Matrix): return Matrix([self<r for r in other])
        raise TypeError(f"{type(other)} cannot use addition operator with Element.")

    @staticmethod
    def parse_bool(b: bool) -> Element: return Element(1) if b else Element(0)


class Vector(object):
    def __init__(self, values: list[int | float | Element] | Vector):
        self.idx = 0
        if isinstance(values, Vector): self.values = [v for v in values]
        else: self.values = [Element(v) for v in values]
    def __repr__(self) -> str: return str([v.value for v in self.values])
    def __radd__(self, other: Element | Vector) -> Element | Vector: return self.__add__(other)
    def __len__(self) -> int: return len(self.values)
    def __getitem__(self, k) -> Element: return self.values[k]
    def __next__(self) -> Element: self.idx += 1; return self.values[self.idx]
    def __iter__(self) -> Iterator[Element]:
        for v in self.values:
            yield v

    def __add__(self, other: int | float | Element | Vector) -> Vector:
        if isinstance(other, (int, float, Element)): return Vector([other+v for v in self])
        elif isinstance(other, Vector): return Vector([v1+v2 for v1, v2 in zip(self, other)])
        raise TypeError(f"{type(other)} cannot use addition operator with Vector.") 
    
    def __mul__(self, other: Element | Vector | Matrix) -> Vector:
        if isinstance(other, Element): return Vector([v*other for v in self])
        elif isinstance(other, Vector): return Vector([v1*v2 for v1, v2 in zip(self, other)])
        elif isinstance(other, Matrix): return Vector([sum(v * r) for v, r in zip(self, other)])
        raise TypeError(f"{type(other)} cannot use addition operator with Vector.") 

    def __eq__(self, other: Vector) -> bool:
        if isinstance(other, Vector): return all([v==x for v, x in zip(self.values, other)]) 
        return False

        
class Matrix(object):
    def __init__(self, rows: list[list | Vector] | Vector):
        self.idx = 0
        self.rows = [r if isinstance(r, Vector) else Vector(r)  for r in rows]
        self.shape = (len(self.rows[0]), len(self))

    def __repr__(self) -> str: return "\n".join([r.__repr__() for r in self])
    def __len__(self) -> int: return len(self.rows)
    def __getitem__(self, k) -> Element: return self.rows[k]
    def __next__(self) -> Element: self.idx += 1; return self.rows[self.idx]
    def _get_col(self, c_idx: int) -> Vector: return Vector([r[c_idx] for r in self])
    def _transpose(self) -> Matrix: return Matrix([self._get_col(i) for i in range(self.shape[0])])
    def __iter__(self) -> Iterator[Vector]:
        for r in self.rows:
            yield r 

    def __mul__(self, other: Element) -> Element | Vector | Matrix:
        if isinstance(other, Element): return Matrix([r*other for r in self])
        elif isinstance(other, Vector): return Vector([sum(c*v) for c, v in zip(self._transpose(),other)])
        elif isinstance(other, Matrix): return Matrix([[sum(r*c) for c in other._transpose()] for r in self])
        raise TypeError(f"{type(other)} cannot use mul operator with Matrix.") 
    
    def __add__(self, other: Element | Matrix) -> Element | Matrix:
        if isinstance(other, Element): return Matrix([r+other for r in self])
        elif isinstance(other, Matrix): return Matrix([r1+r2 for r1, r2 in zip(self, other)])
        raise TypeError(f"{type(other)} cannot use addition operator with Matrix.") 
    
    def __eq__(self, other: Matrix) -> bool:
        if isinstance(other, Matrix): return all([r==x for r, x in zip(self, other)]) 
        return False
    
