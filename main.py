import random
from libs.layer_dense import LayerDense
from libs.activation_functions import sigmoid, leaky_relu
import numpy as np
from libs.neural_network import NeuralNetwork, load_nn


nn = NeuralNetwork([LayerDense(2, 16, leaky_relu), 
                    LayerDense(16, 8,leaky_relu), 
                    LayerDense(8, 4,leaky_relu),
                    LayerDense(4, 1,sigmoid)])

nn.set_learning_rate(0.01)

for i in range(1_000):
    inputs = [random.randint(0, 1), random.randint(0, 1)]
    expected_output = [inputs[0] ^ inputs[1]]
    nn.train(inputs, expected_output)

nn.save("model.npy")
nn = load_nn("model.npy")

print(f"real answer is {0 ^ 0}")
print(f"ai prediction is {np.float64(nn.forward([0, 0])[0])}")
print(f"real answer is {0 ^ 1}")
print(f"ai prediction is {np.float64(nn.forward([0, 1])[0])}")
print(f"real answer is {1 ^ 0}")
print(f"ai prediction is {np.float64(nn.forward([1, 0])[0])}")
print(f"real answer is {1 ^ 1}")
print(f"ai prediction is {np.float64(nn.forward([1, 1])[0])}")

