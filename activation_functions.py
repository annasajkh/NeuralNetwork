import math
from typing import Callable
import numpy as np

class ActivationFunction:
    def __init__(self, activation_function : Callable[[np.ndarray], np.ndarray], derivative : Callable[[np.ndarray],  np.ndarray]) -> None:
        self.activation_function = activation_function
        self.derivative = derivative


sigmoid = ActivationFunction(lambda arr : 1 / (1 + (np.exp(-arr))) , lambda arr : arr * (1 - arr))
leaky_relu = ActivationFunction(lambda arr : ((arr > 0) * arr) + ((arr <= 0) * arr * 0.01), lambda arr : ((arr > 0) * 1) + ((arr <= 0) * 0.01))
