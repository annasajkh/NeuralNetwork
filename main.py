import random
from libs.neural_network import *

nn = NeuralNetwok(input_size=2, hidden_layer_size=10, output_size=1,hidden_layer_count=3)
nn.set_activation_functions(hidden_activation=leaky_relu, output_activation=sigmoid)

for i in range(5_000):
    inputs = [random.randint(0, 1), random.randint(0, 1)]
    output = [inputs[0] ^ inputs[1]]
    nn.train(inputs, output)


nn.save("model.npy")
nn = load_nn("model.npy")

print(f"real answer is {0 ^ 0}")
print(f"ai prediction is {nn.forward([0, 0])[0]}")
print(f"real answer is {0 ^ 1}")
print(f"ai prediction is {nn.forward([0, 1])[0]}")
print(f"real answer is {1 ^ 0}")
print(f"ai prediction is {nn.forward([1, 0])[0]}")
print(f"real answer is {1 ^ 1}")
print(f"ai prediction is {nn.forward([1, 1])[0]}")
