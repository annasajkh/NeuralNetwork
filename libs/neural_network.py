from typing import List

from numpy import float64, ndarray
from libs.activation_functions import *

class NeuralNetwok:

    def __init__(self, input_size: int, hidden_layer_size: int, output_size: int, hidden_layer_count : int) -> None:
        # make weights and biases
        self.weights : List[np.ndarray] = [0] * (hidden_layer_count + 1)
        self.biases : List[np.ndarray] = [0] * (hidden_layer_count + 1)

        self.input_size : int = input_size
        self.hidden_layer_size : int = hidden_layer_size
        self.output_size : int = output_size
        self.hidden_layer_count : int = hidden_layer_count
        self.learning_rate : float = 0.01
        self.expected_output : np.ndarray = None
        self.weights_arr : List[float] = [0] * len(self.weights)
        self.hidden_activation : ActivationFunction = leaky_relu
        self.output_activation : ActivationFunction = sigmoid

        # make the network with size of input + hidden_layer_count + output
        self.network : List[np.ndarray]= [np.empty(shape=(input_size,1)),*[np.empty(shape=(hidden_layer_size,1)) for i in range(hidden_layer_count)], np.empty(shape=(output_size,1))]
        

        for i in range(1, len(self.network)):
            bias : np.ndarray = None
            weight : np.ndarray = None

            # make this if it's a hidden layer
            if i > 1 and i != len(self.network) - 1:
                # weight from hidden to hidden
                weight = np.random.rand(hidden_layer_size, hidden_layer_size) * 4 - 2
            else:
                # make this is it's a output layer
                if i == len(self.network) - 1:
                    # weight from hidden to output
                    weight = np.random.rand(output_size, hidden_layer_size) * 4 - 2
                
                # make this is it's a input layer
                else:
                    # weight from input to hidden
                    weight = np.random.rand(hidden_layer_size, input_size) * 4 - 2
            
            self.weights[i - 1] = weight

            # if is not output layer make new array size of hidden layer else make new array size of output layer
            if i != len(self.network) - 1:
                bias = np.zeros((hidden_layer_size,1))
            else:
                bias = np.zeros((output_size,1))

            self.biases[i - 1] = bias

    def set_activation_functions(self, hidden_activation : ActivationFunction, output_activation : ActivationFunction) -> None:
        """set hidden layer and output layer activation function"""
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
    
    def set_learning_rate(self, learning_rate : float) -> None:
        """set the learning rate default is 0.01"""
        self.learning_rate = learning_rate
    
    def preprocess(self, i: int, train: int) -> None:
        # dot product between weight and before layer and add biasses
        results : np.ndarray = np.dot(self.weights[i - 1], self.network[i - 1])
        results += self.biases[i - 1]


        # apply activation function
        if i == len(self.network) - 1:
            results = self.output_activation.activation_function(results)
        else:
            results = self.hidden_activation.activation_function(results)

        # set current layer to the result
        self.network[i] = results

        # if the current layer is the output and it's training then do the backpropagation
        if i == len(self.network) - 1 and train:
            self.backpropagation(results)
    
    def forward(self, input: List[float]) -> np.ndarray:
        # checking if input length is greater than input size
        if len(input) > self.input_size:
            raise Exception("Error input is bigger than the input size")
        
        # pass input to input layer
        self.network[0] = np.array(input).reshape(-1,1)

        # feed forward the input from layer 1 so it can get layer 0
        for i in range(1, len(self.network)):
            self.preprocess(i, False)

        # return the last layer aka the output
        return self.network[-1].flatten()
    
    def get_all_errors(self, output_error : np.ndarray) -> List[np.ndarray]:
        # make array and set last index = error
        errors : List[np.ndarray] = self.weights_arr
        errors[len(errors) - 1] = output_error

        # calculate error on index i and pass it on index before it
        for i in range(len(errors) - 1, 0,-1):
            errors[i - 1] = np.dot(self.weights[i].T, errors[i])
        
        return errors
    
    def changing_weights_and_biases(self, i : int, errors: np.ndarray, layer : np.ndarray, after_layer : np.ndarray) -> None: 
        if i == len(self.network) - 2:
            layer_result = self.output_activation.derivative(layer)
        else:
            layer_result = self.hidden_activation.derivative(layer)

        # multiply it by errors and learning rate
        layer_result *= errors
        layer_result *= self.learning_rate
        
        # deltaWeight = gradient multiply after_layer transposted
        deltaWeight : np.ndarray = np.dot(layer_result, after_layer.T)

        # adjust the weight by deltaWeight
        self.weights[i] += deltaWeight

        # adjust the bias by it's delta (it's just the gradient)
        self.biases[i] += layer_result

        
    def backpropagation(self, output: np.ndarray) -> None:
        self.expected_output -= output


        # get all errors
        errors : List[np.ndarray] = self.get_all_errors(self.expected_output)

        # backpropagation
        for i in range(len(errors) - 1, -1, -1):
            self.changing_weights_and_biases(i, errors[i], self.network[i + 1], self.network[i])
    
    def train(self, input : List[float], expected_output : List[float]) -> None:
        # checking if expected output length is greater than output size
        if len(expected_output) > self.output_size:
            raise Exception("Error expected output is bigger than the output size")
        
        # checking if input length is greater than input size
        if len(input) > self.input_size:
            raise Exception("Error input is bigger than the input size")

        self.expected_output = np.array(expected_output, dtype=float64).reshape(-1,1)

        # pass input to input layer
        self.network[0] = np.array(input).reshape(-1,1)

        # feedforward
        for i in range(1, len(self.network)):
            self.preprocess(i, True)
    
    def save(self, file : str) -> None:
        data : ndarray = np.array([ self.weights, 
                                    self.biases, 
                                    [self.input_size, self.hidden_layer_size, self.output_size, self.hidden_layer_count, self.learning_rate],
                                    [self.hidden_activation.id,self.output_activation.id]],dtype=object)
        
        np.save(file, data)

        print("model saved to " + file)


def load_nn(file : str) -> NeuralNetwok:
    print("model loaded from " + file)
    
    data : ndarray = np.load(file, allow_pickle=True)
    nn = NeuralNetwok(data[2][0], data[2][1], data[2][2], data[2][3])
    nn.set_learning_rate(data[2][4])
    nn.set_activation_functions(get_function(data[3][0]), get_function(data[3][1]))
    nn.weights = data[0]
    nn.biases = data[1]

    return nn