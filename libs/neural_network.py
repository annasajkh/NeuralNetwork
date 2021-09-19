from libs.activation_functions import get_function
from typing import List
from numpy import ndarray
from libs.layer_dense import LayerDense
import numpy as np


class NeuralNetwork:
    def __init__(self, layers : List[LayerDense]) -> None:
        self.layers : LayerDense = layers
        self.learning_rate : float = 0.01
        self.network : List[ndarray] = None
    
    def set_learning_rate(self, learning_rate : float) -> None:
        """set the learning rate default is 0.01"""
        self.learning_rate : float = learning_rate
    
    def forward(self, input : ndarray) -> ndarray:
        """feed forward through entire network"""
        input = np.array(input, dtype=np.float64).reshape(-1, 1)

        assert len(input.flatten()) == self.layers[0].num_input, "input size is not the same as input layer size"
        
        self.network : List[ndarray] = []
        self.network.append(input)

        for i, layer in enumerate(self.layers):
            self.network.append(layer.forward(self.network[i]))

        return self.network[-1]
    
    def get_all_errors(self, output_error : np.ndarray) -> List[np.ndarray]:
        """get all layers errors"""

        errors : List[np.ndarray] = [0] * len(self.layers)
        errors[len(errors) - 1] = output_error

        for i in range(len(errors) - 1, 0,-1):
            errors[i - 1] = self.layers[i].get_errors(errors[i])
        
        return errors

    
    def train(self, input : ndarray, target : ndarray) -> ndarray:
        """train the network"""
        target = np.array(target, dtype=np.float64).reshape(-1,1)

        assert len(target) == self.layers[-1].num_output, "target size is not the same as output layer size"
        
        output : ndarray = self.forward(input)

        errors = self.get_all_errors(target - output)

        for i in range(len(errors) - 1, -1, -1):
            self.layers[i].step(errors[i], self.network[i + 1], self.network[i], self.learning_rate)
        
        return output
    
    def save(self, filename : str) -> None:
        np.save(filename, np.array([[[layer.weights, layer.biases, layer.activation_function_id, layer.num_input, layer.num_output]for layer in self.layers], self.learning_rate], dtype=object))
        print(f"saved to {filename}")


def load_nn(filename : str) -> NeuralNetwork:
    data : ndarray = np.load(filename, allow_pickle=True)
    layers = []

    for layer_data in data[0]:
        layer : LayerDense = LayerDense(layer_data[3],layer_data[4],get_function(layer_data[2]))
        layer.weights = layer_data[0]
        layer.biases = layer_data[1]
        layers.append(layer)
    
    nn = NeuralNetwork(layers)
    nn.set_learning_rate(data[1])

    print(f"loaded from {filename}")

    return nn