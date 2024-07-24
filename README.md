**BareBonesNN**
A neural network library built from scratch.

**BareBonesNN** is a lightweight neural network library implemented from scratch in Python, without relying on high-level machine learning libraries. This project aims to provide a clear and fundamental understanding of neural network concepts and operations, making it an excellent educational resource for learning and experimentation.

## Features

- **Basic Neural Network Components**: Core classes including `Value`, `Neuron`, `Layer`, and `MLP` (Multi-Layer Perceptron).
- **Automatic Differentiation**: Built-in backpropagation for gradient computation.
- **Customisable and Extensible**: Easily modify and extend the code to experiment with various neural network architectures.
- **Lightweight**: Minimal dependencies, focusing on core principles.

## Installation

To install BareBonesNN, use pip:

```bash
pip install barebonesnn==0.1.0
```

## Usage

Here is a basic example demonstrating how to create and use a simple neural network using BareBonesNN:

### Example: Creating and Using an MLP

```python
from barebonesnn import Value, MLP

# Create a simple MLP with 3 input neurons and layers with [4, 4, 1] neurons
mlp = MLP(3, [4, 4, 1])

# Example input
x = [Value(1.0), Value(2.0), Value(3.0)]

# Forward pass
output = mlp(x)
print("Output:", output)

# Backward pass
output.backward()

# Inspect gradients
for param in mlp.parameters():
    print(param.label, param.grad)
```

### Structure

- **`barebonesnn/value.py`**: Contains the `Value` class, which represents a value in a computational graph.
- **`barebonesnn/neuron.py`**: Contains the `Neuron` class, representing a single neuron.
- **`barebonesnn/layer.py`**: Contains the `Layer` class, representing a layer of neurons.
- **`barebonesnn/mlp.py`**: Contains the `MLP` class, representing a multi-layer perceptron.

### Documentation

#### `Value` Class

Represents a value in a computational graph, supporting basic operations and automatic differentiation.

#### `Neuron` Class

Represents a single neuron in a neural network, with a set of weights and a bias.

#### `Layer` Class

Represents a layer in a neural network, consisting of multiple neurons.

#### `MLP` Class

Represents a multi-layer perceptron, consisting of multiple layers.

## Examples

The `examples/` directory contains scripts demonstrating various usages of the library. To run an example:

```bash
python examples/example.py
```

## Tests

Unit tests are provided in the `tests/` directory. To run the tests, use the following command from the root of the project:

```bash
python -m unittest discover tests
```

## Contributing

Contributions are welcome! If you have suggestions, bug reports, or feature requests, please open an issue or submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
