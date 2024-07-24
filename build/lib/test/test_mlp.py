import unittest
from barebonesnn.value import Value
from barebonesnn.mlp import MLP

class TestMLP(unittest.TestCase):
    def test_mlp_forward(self):
        mlp = MLP(3, [4, 4, 1])
        x = [Value(1.0), Value(2.0), Value(3.0)]
        y = mlp(x)
        self.assertIsInstance(y, Value)

    def test_parameters(self):
        mlp = MLP(3, [4, 4, 1])
        params = mlp.parameters()
        expected_params = (3 * 4) + 4 + (4 * 4) + 4 + (4 * 1) + 1  # (nin * nout + bias) for each layer
        self.assertEqual(len(params), expected_params)

if __name__ == '__main__':
    unittest.main()
