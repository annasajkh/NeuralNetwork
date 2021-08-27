from typing import Callable
import numpy as np

class ActivationFunction:
    def __init__(self, activation_function : Callable[[np.ndarray], np.ndarray], derivative : Callable[[np.ndarray],  np.ndarray], id : int) -> None:
        self.activation_function = activation_function
        self.derivative = derivative
        self.id = id


sigmoid = ActivationFunction(lambda arr : 1 / (1 + (np.exp(-arr))) , lambda arr : arr * (1 - arr), 0)
leaky_relu = ActivationFunction(lambda arr : ((arr > 0) * arr) + ((arr <= 0) * arr * 0.01), lambda arr : ((arr > 0) * 1) + ((arr <= 0) * 0.01), 1)

def get_function(id: int) -> ActivationFunction:
    if id == sigmoid.id:
        return sigmoid
    elif id == leaky_relu.id:
        return leaky_relu
    else:
        raise Exception("invalid id")