from typing import Callable
import numpy as np

class ActivationFunction:
    def __init__(self, function : Callable[[np.ndarray], np.ndarray], derivative : Callable[[np.ndarray],  np.ndarray], id : int) -> None:
        self.function = function
        self.derivative = derivative
        self.id = id


def softmax(arr):
    f = np.exp(arr - np.max(arr))
    return f / np.sum(f)

def d_softmax(arr):
    jacobian_m = np.array([np.diag(arr)])


    for i in range(len(jacobian_m)):
        for j in range(len(jacobian_m)):
            if i == j:
                jacobian_m[i][j] = arr[i] * (1-arr[i])
            else:
                jacobian_m[i][j] = -arr[i]*arr[j]

    return jacobian_m


sigmoid = ActivationFunction(lambda arr : np.exp(np.fmin(arr, 0)) / (1 + np.exp(-np.abs(arr))), lambda arr : arr * (1 - arr), 0)
leaky_relu = ActivationFunction(lambda arr : ((arr > 0) * arr) + ((arr <= 0) * arr * 0.01), lambda arr : ((arr > 0) * 1) + ((arr <= 0) * 0.01), 1)
tanh = ActivationFunction(lambda arr: np.tanh(arr), lambda arr : 1 - arr ** 2, 2)
relu = ActivationFunction(lambda arr : (arr > 0) * arr, lambda arr : (arr > 0) * 1, 3)
softmax = ActivationFunction(softmax, d_softmax, 4)

def get_function(id: int) -> ActivationFunction:
    if id == sigmoid.id:
        return sigmoid
    elif id == leaky_relu.id:
        return leaky_relu
    elif id == tanh.id:
        return tanh
    elif id == relu.id:
        return relu
    elif id == softmax.id:
        return softmax
    else:
        raise Exception("invalid id")