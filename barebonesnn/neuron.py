import random
from typing import List
from .value import Value



class Neuron:
    """
    A single neuron in a neural network, with a set of weights and a bias.
    """

    def __init__(self, nin: int):
        """
        Initialize a Neuron with random weights and a bias.

        :param nin: Number of input connections to the neuron.
        """
        self.w: List[Value] = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b: Value = Value(random.uniform(-1, 1))

    def __call__(self, x: List[Value]) -> Value:
        """
        Compute the output of the neuron given an input.

        :param x: List of input values.
        :return: The output value after applying the tanh activation function.
        """
        # Weighted sum of inputs plus bias
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        out = act.tanh()
        return out

    def parameters(self) -> List[Value]:
        """
        Get all parameters of the neuron (weights and bias).

        :return: A list containing the weights and bias.
        """
        return self.w + [self.b]