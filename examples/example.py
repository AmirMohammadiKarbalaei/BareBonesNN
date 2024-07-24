from barebonesnn.value import Value
from barebonesnn.mlp import MLP

# Create a simple MLP with 3 input neurons and layers with [6, 6, 1] neurons
mlp = MLP(3, [6, 6, 1])

# Example input
x = [Value(1.0), Value(2.0), Value(3.0)]

# Forward pass
output = mlp(x)
print("Output:", output)

# Backward pass
output.backward()