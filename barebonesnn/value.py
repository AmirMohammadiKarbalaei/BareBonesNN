import math
from typing import Union, Tuple, Callable, Set




class Value:
    """
    Class to represent a value in a computational graph, supporting operations like addition,
    multiplication, exponentiation, division, negation, and the hyperbolic tangent function.
    It also supports automatic differentiation via the backward method.
    """

    def __init__(self, data: float, _children: Tuple['Value', ...] = (), _op: str = '', label: str = ''):
        """
        Initialize a Value object.

        :param data: The numerical data associated with this value.
        :param _children: The parent nodes contributing to this value in the computational graph.
        :param _op: The operation that produced this value.
        :param label: An optional label for this value.
        """
        self.data = data
        self.grad = 0.0
        self._backward: Callable[[], None] = lambda: None
        self._prev: Set['Value'] = set(_children)
        self._op = _op
        self.label = label


    def __repr__(self) -> str:
        """
        Return a string representation of the Value object.

        :return: A string representation of the Value object.
        """
        return f"Value(data={self.data})"


    def __add__(self, other: Union['Value', float]) -> 'Value':
        """
        Add another value or float to this value.

        :param other: Another value or a float to add.
        :return: The result of the addition as a new Value object.
        """
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad

        out._backward = _backward

        return out
    

    def __radd__(self, other: Union['Value', float]) -> 'Value':
        """
        Add another value or float to this value (reverse addition).

        :param other: Another value or a float to add.
        :return: The result of the addition as a new Value object.
        """
        return self + other


    def __mul__(self, other: Union['Value', float]) -> 'Value':
        """
        Multiply this value by another value or float.

        :param other: Another value or a float to multiply by.
        :return: The result of the multiplication as a new Value object.
        """
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward

        return out


    def __rmul__(self, other: Union['Value', float]) -> 'Value':
        """
        Multiply this value by another value or float (reverse multiplication).

        :param other: Another value or a float to multiply by.
        :return: The result of the multiplication as a new Value object.
        """
        return self * other


    def __pow__(self, other: Union[int, float]) -> 'Value':
        """
        Raise this value to the power of another value or float.

        :param other: The exponent, which must be an int or float.
        :return: The result of the exponentiation as a new Value object.
        """
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Value(self.data ** other, (self,), f'**{other}')

        def _backward():
            self.grad += other * (self.data ** (other - 1)) * out.grad

        out._backward = _backward

        return out


    def __truediv__(self, other: Union['Value', float]) -> 'Value':
        """
        Divide this value by another value or float.

        :param other: Another value or a float to divide by.
        :return: The result of the division as a new Value object.
        """
        return self * other**-1


    def __neg__(self) -> 'Value':
        """
        Negate this value.

        :return: The negated value as a new Value object.
        """
        return self * -1


    def __sub__(self, other: Union['Value', float]) -> 'Value':
        """
        Subtract another value or float from this value.

        :param other: Another value or a float to subtract.
        :return: The result of the subtraction as a new Value object.
        """
        return self + (-other)


    def tanh(self) -> 'Value':
        """
        Compute the hyperbolic tangent of this value.

        :return: The result of the tanh function as a new Value object.
        """
        x = self.data
        t = (math.exp(2 * x) - 1) / (math.exp(2 * x) + 1)
        out = Value(t, (self,), 'tanh')

        def _backward():
            self.grad += (1 - t ** 2) * out.grad

        out._backward = _backward

        return out



    def exp(self) -> 'Value':
        """
        Compute the exponential of this value.

        :return: The result of the exp function as a new Value object.
        """
        x = self.data
        out = Value(math.exp(x), (self,), 'exp')

        def _backward():
            self.grad += out.data * out.grad

        out._backward = _backward

        return out


    def backward(self) -> None:
        """
        Perform backpropagation to compute the gradients of this value with respect to all
        preceding values in the computational graph.
        """
        topo = []
        visited = set()

        def build_topo(v: 'Value') -> None:
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        self.grad = 1.0
        for node in reversed(topo):
            node._backward()