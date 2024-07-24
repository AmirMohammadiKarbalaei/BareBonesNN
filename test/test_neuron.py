import unittest
from barebonesnn.value import Value
from barebonesnn.neuron import Neuron

class TestNeuron(unittest.TestCase):
    def test_neuron_forward(self):
        n = Neuron(3)
        x = [Value(1.0), Value(2.0), Value(3.0)]
        y = n(x)
        self.assertIsInstance(y, Value)

    def test_parameters(self):
        n = Neuron(3)
        params = n.parameters()
        self.assertEqual(len(params), 4)  # 3 weights + 1 bias

if __name__ == '__main__':
    unittest.main()
