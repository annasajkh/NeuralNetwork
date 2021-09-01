import numpy as np
from numpy import ndarray
from libs.activation_function import *

class LayerDense:
    def __init__(self, num_input : ndarray, num_output : ndarray, activation_function : ActivationFunction) -> None:
        self.weights : ndarray = np.random.rand(num_output, num_input) * 4 - 2
        self.biases : ndarray = np.zeros((num_output, 1))
        self.activation_function_id : ActivationFunction = activation_function.id

    def forward(self, input : ndarray) -> ndarray:
        """feedforward the input and returns the output of this layer"""
        return get_function(self.activation_function_id).function(np.dot(self.weights, np.array(input).reshape(-1,1)) + self.biases)
    
    def get_errors(self, errors : ndarray) -> ndarray:
        """get the errors of this layer using other errors (usually it's the layer before this layer in feedforward network)"""
        return np.dot(self.weights.T, errors)
    
    def step(self, errors: np.ndarray, layer : ndarray, after_layer : np.ndarray, learning_rate : float) -> None:
        """change weights and biases by errors"""
        gradient : ndarray = get_function(self.activation_function_id).derivative(layer)
        gradient *= errors
        gradient *= learning_rate
        
        delta_weights : np.ndarray = np.dot(gradient, after_layer.T)

        self.weights += delta_weights
        self.biases += gradient
