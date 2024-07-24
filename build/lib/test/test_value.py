import unittest
from barebonesnn.value import Value

class TestValue(unittest.TestCase):
    def test_addition(self):
        a = Value(2.0)
        b = Value(3.0)
        c = a + b
        self.assertEqual(c.data, 5.0)

    def test_multiplication(self):
        a = Value(2.0)
        b = Value(3.0)
        c = a * b
        self.assertEqual(c.data, 6.0)

    def test_exponentiation(self):
        a = Value(3.0)
        b = a ** 2
        self.assertEqual(b.data, 9.0)

    def test_tanh(self):
        import math
        a = Value(0.5)
        b = a.tanh()
        expected = (math.exp(1.0) - math.exp(-1.0)) / (math.exp(1.0) + math.exp(-1.0))
        self.assertAlmostEqual(b.data, expected, places=6)

if __name__ == '__main__':
    unittest.main()