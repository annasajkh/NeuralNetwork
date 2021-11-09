import numpy as np
from typing import Callable
from numpy import ndarray

def MAE(prediction : ndarray, target : ndarray) -> ndarray:
    return target - prediction

loss_functions_arr = [MAE]

def get_function(id : int) -> Callable[[ndarray, ndarray], ndarray]:
    return loss_functions_arr[id]

def get_function_id(function : Callable[[ndarray, ndarray], ndarray]):
    return loss_functions_arr.index(function)