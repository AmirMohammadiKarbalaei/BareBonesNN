from typing import Union, List
from .neuron import Neuron
from .value import Value



class Layer:
    """
    A layer in a neural network, consisting of multiple neurons.
    """

    def __init__(self, nin: int, nout: int):
        """
        Initialize a Layer with a specified number of input and output connections.

        :param nin: Number of input connections to each neuron.
        :param nout: Number of neurons in this layer.
        """
        self.neurons: List[Neuron] = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x: List[Value]) -> Union[Value, List[Value]]:
        """
        Compute the output of the layer given an input.

        :param x: List of input values.
        :return: The output values from the layer.
        """
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self) -> List[Value]:
        """
        Get all parameters of the layer.

        :return: A list containing all the parameters (weights and biases) of the neurons in the layer.
        """
        return [p for neuron in self.neurons for p in neuron.parameters()]