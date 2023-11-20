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
    df = s * (1 - s)
    return s, df



# Model helper functions

def forward_propagate():
    print("To be implemented")

def backward_propagate():
    print("To be implemented")

def compute_MSE():
    print("To be implemented")




# Main algorithm function
def multi_layer_perceptron():
    print("To be implemented")
  
