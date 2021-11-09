import numpy as np
from numpy import ndarray
from libs.activation_functions import *

class LayerDense:
    def __init__(self, num_input : ndarray, num_output : ndarray, activation_function : ActivationFunction) -> None:
        self.weights : ndarray = np.random.randn(num_output, num_input)
        self.biases : ndarray = np.zeros((num_output, 1))
        self.activation_function : ActivationFunction = activation_function
        self.num_input : int = num_input
        self.num_output : int = num_output

    def forward(self, input : ndarray) -> ndarray:
        """feedforward the input and returns the output of this layer"""
        return self.activation_function.function(np.dot(self.weights, np.array(input).reshape(-1,1)) + self.biases)
    
    def get_error(self, error : ndarray) -> ndarray:
        """get the error of this layer using other error (usually it's the layer before this layer in feedforward network)"""
        return np.dot(self.weights.T, error)
    
    def step(self, error: np.ndarray, layer : ndarray, after_layer : np.ndarray, learning_rate : float) -> None:
        """change weights and biases by error"""
        gradient : ndarray = self.activation_function.derivative(layer) * error * learning_rate

        delta_weights : np.ndarray = np.dot(gradient, after_layer.T)

        self.weights += delta_weights
        self.biases += gradient