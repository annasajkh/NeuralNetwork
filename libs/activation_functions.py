from typing import Callable
import numpy as np

class ActivationFunction:
    def __init__(self, function : Callable[[np.ndarray], np.ndarray], derivative : Callable[[np.ndarray],  np.ndarray], id : int) -> None:
        self.function = function
        self.derivative = derivative
        self.id = id


def softmax(x):
    x = x - max(x)
    e_x = np.exp(x) 

    return e_x/ e_x.sum(axis=0, keepdims=True)

def d_softmax(s):
    return np.diagflat(s) - np.dot(s, s.T)


sigmoid = ActivationFunction(lambda arr : 1 / (1 + (np.exp(-arr))) , lambda arr : arr * (1 - arr), 0)
leaky_relu = ActivationFunction(lambda arr : ((arr > 0) * arr) + ((arr <= 0) * arr * 0.01), lambda arr : ((arr > 0) * 1) + ((arr <= 0) * 0.01), 1)
softmax = ActivationFunction(softmax,d_softmax, 2)

def get_function(id: int) -> ActivationFunction:
    if id == sigmoid.id:
        return sigmoid
    elif id == leaky_relu.id:
        return leaky_relu
    elif id == softmax.id:
        return softmax
    else:
        raise Exception("invalid id")