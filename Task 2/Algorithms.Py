import numpy as np
import pandas as pd

# Activation functions

##values to be between -1 and 1
##df(a)=1-a²
##you want the tanh use tanh_and_derivative(x)[0]
##you want derivative use tanh_and_derivative(x)[1]
def tanh_and_derivative(x):
    t = (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
    dt = 1 - t ** 2
    return t, dt

##f(x)=1/(1+exp(-x) the function range between (0,1)
## df(x)=f(x)*(1-f(x))
##you want the sigmoid use sigmoid_and_derivative(x)[0]
##you want derivative use sigmoid_and_derivative(x)[1]
def sigmoid_and_derivative(x):
    s = 1 / (1 + np.exp(-x))
    ds = s * (1 - s)
    return s, ds


# Model helper functions

# Initializes random weight matrices according to MLP architecture
def initialize_weights(n_features, number_of_hidden_layers, number_of_nodes_in_each_layer,  use_bias,  n_classes = 3):
    # Number of inputs to the first hidden layer
    input_layer_size = n_features

    # List of weight matrices
    W = []
    B = []
    for i in range (number_of_hidden_layers):
        # Weight matrix shape -> number of nodes in current layer * number of inputs from previous layer
        # The weights of the first perceptron in the first hidden layer would be w[0][0] -> vector that is the size of input of previous layer
        w_current_layer = np.random.rand(number_of_nodes_in_each_layer[i], input_layer_size)

        # 1 bias for each perceptron
        if (use_bias):
            b_current_layer = np.random.rand(number_of_nodes_in_each_layer[i])
        else:
            b_current_layer = np.zeros(number_of_nodes_in_each_layer[i])
        W.append(w_current_layer)
        B.append(b_current_layer)

        # Next layer's input size is current layer's number of nodes.
        input_layer_size = number_of_nodes_in_each_layer[i]

    # Output layer weights and bias
    # Three perceptrons that have input size of last hidden layer.
    w_output_layer = np.random.rand(n_classes, input_layer_size)
    if (use_bias):
        b_output_layer = np.random.rand(n_classes)
    else:
        b_output_layer = np.zeros(n_classes)

    W.append(w_output_layer)
    B.append(b_output_layer)

    return W, B

# Propagates all neurons' outputs and returns a matrix that has the activation of every neuron.
def forward_propagate(x, W, B, isSigmoid):
    # Initialize the input to the first layer as the sample input x
    a = x

    # Store all activations along the way (for backward propagation)
    activations = []
    gradients = []

    # Iterate over the layers of the network
    for w, b in zip(W, B):
        # Calculate net output of this layer
        z = np.dot(w, a) + b
        # Apply the selected activation function to get the input of the next layer
        if isSigmoid:
            a, g = sigmoid_and_derivative(z)
        else:
            a, g = tanh_and_derivative(z)
        activations.append(a)
        gradients.append(g)
    
    # return the activations of all layers including the final output layer
    return activations, gradients

# Used to predict bean class (1 sample only)
def predict(x,W,B, isSigmoid, classes = ["BOMBAY", "CALI", "SIRA"]):
    # Get output layer activations
    activations = forward_propagate(x,W,B,isSigmoid)[0]

    # Last layer output is three activations (or probabilities), one for each class
    output = activations[-1]

    # Getting the index of the max value (max probability is my predicted class)
    max_index = np.argmax(output)

    # Getting final class label
    return classes[max_index]


# Propagates error signal and calculates local gradients.
def backward_propagate(y, W, activations, gradients):
    # One hot encoding for each class
    classes_encoding = {"BOMBAY": [1, 0, 0], "CALI": [0, 1, 0], "SIRA": [0, 0, 1]}

    # Calculating output layer gradients
    output_layer_errors = (classes_encoding[y] - activations[-1]) * gradients[-1]

    layers_errors = [output_layer_errors]

    # Iterate backwards through the layers to calculate gradients
    for i in range(len(W) - 2, -1, -1):
        current_layer_errors = np.dot(W[i+1].T, layers_errors[0]) * gradients[i]
        layers_errors.insert(0, current_layer_errors)  # Insert at beginning to maintain order

    return layers_errors


# Updates weights using calculated gradients
def update_weights(x, W, B, layers_errors, activations, learning_rate, use_bias):
    # All layers inputs (starting with x, then a series of hidden layers activations) in one matrix (Excluding last layer activation)
    inputs = [x] + activations[:-1]  # Exclude output layer activation

    # Update weights of each layer
    for i in range(len(W)):
        W[i] += learning_rate * np.outer(layers_errors[i], inputs[i])
        if use_bias:
            B[i] += learning_rate * layers_errors[i]

    return W, B


def compute_MSE():
    print("To be implemented")



# Main algorithm function
def multi_layer_perceptron(X,y, MSE_threshold = 0.02, number_of_hidden_layers = 2, number_of_nodes_in_each_layer = [4,5] , learning_rate = 0.01, epochs= 100, use_bias = True, isSigmoid = True):

    # Initialization
    n_samples, n_features = X.shape
    W, B = initialize_weights(n_features, number_of_hidden_layers, number_of_nodes_in_each_layer, use_bias= use_bias)

    # Training the network
    for j in range(epochs):
        for i in range(n_samples):
            activations, gradients = forward_propagate(X[i], W, B, isSigmoid)
            layers_errors = backward_propagate(y[i], W, activations, gradients)
            W, B = update_weights(X[i], W, B, layers_errors, activations, learning_rate, use_bias=use_bias)


    return W, B

    
