from typing import Union, List
from .layer import Layer
from .value import Value



class MLP:
    """
    A multi-layer perceptron (MLP) neural network, consisting of multiple layers.
    """

    def __init__(self, nin: int, nouts: List[int]):
        """
        Initialize an MLP with a specified number of input connections and a list of output connections per layer.

        :param nin: Number of input connections to the MLP.
        :param nouts: List specifying the number of output connections for each layer.
        """
        sz = [nin] + nouts
        self.layers: List[Layer] = [Layer(sz[i], sz[i + 1]) for i in range(len(nouts))]

    def __call__(self, x: List[Value]) -> Union[Value, List[Value]]:
        """
        Compute the output of the MLP given an input.

        :param x: List of input values.
        :return: The output values from the MLP.
        """
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self) -> List[Value]:
        """
        Get all parameters of the MLP.

        :return: A list containing all the parameters (weights and biases) of the neurons in all layers of the MLP.
        """
        return [p for layer in self.layers for p in layer.parameters()]