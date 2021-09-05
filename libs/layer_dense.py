import numpy as np
from numpy import ndarray
from libs.activation_functions import *

class LayerDense:
    def __init__(self, num_input : ndarray, num_output : ndarray, activation_function : ActivationFunction) -> None:
        self.weights : ndarray = np.random.rand(num_output, num_input) * 4 - 2
        self.biases : ndarray = np.zeros((num_output, 1))
        self.activation_function_id : ActivationFunction = activation_function.id
        self.activation_function = get_function(self.activation_function_id)
        self.num_input : int = num_input
        self.num_output : int = num_output

    def forward(self, input : ndarray) -> ndarray:
        """feedforward the input and returns the output of this layer"""
        return self.activation_function.function(np.dot(self.weights, np.array(input).reshape(-1,1)) + self.biases)
    
    def get_errors(self, errors : ndarray) -> ndarray:
        """get the errors of this layer using other errors (usually it's the layer before this layer in feedforward network)"""
        return np.dot(self.weights.T, errors)
    
    def step(self, errors: np.ndarray, layer : ndarray, after_layer : np.ndarray, learning_rate : float) -> None:
        """change weights and biases by errors"""

        if self.activation_function_id  != softmax.id:
            gradient : ndarray = self.activation_function.derivative(layer) * errors * learning_rate
        else:
            gradient : ndarray = (layer - errors) * learning_rate

        delta_weights : np.ndarray = np.dot(gradient, after_layer.T)

        self.weights += delta_weights
        self.biases += gradient
