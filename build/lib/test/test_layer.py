import unittest
from barebonesnn.value import Value
from barebonesnn.layer import Layer

class TestLayer(unittest.TestCase):
    def test_layer_forward(self):
        l = Layer(3, 2)
        x = [Value(1.0), Value(2.0), Value(3.0)]
        y = l(x)
        self.assertEqual(len(y), 2)
        self.assertIsInstance(y[0], Value)

    def test_parameters(self):
        l = Layer(3, 2)
        params = l.parameters()
        self.assertEqual(len(params), 8)  # 2 neurons * (3 weights + 1 bias)

if __name__ == '__main__':
    unittest.main()
