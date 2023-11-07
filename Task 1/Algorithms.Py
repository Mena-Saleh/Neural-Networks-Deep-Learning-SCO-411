import numpy as np
import pandas as pd

# Activation function
def signum(x):
    if x > 0:
        return 1
    elif x < 0:
        return -1
    else:
        return 0

# Model helper functions
def predict(X,w,b):
    y_pred_continuous = np.dot(X, w) + b
    y_pred = [signum(x) for x in y_pred_continuous]
    return np.array(y_pred)

def compute_MSE(X,y,w,b):
    y_pred = predict(X,w,b)
    MSE = np.mean((y - y_pred)**2)
    return MSE

# Model training function (using stochastic gradient descent)
def single_layer_perceptron(X,y, MSE_threshold = 0.02 , learning_rate = 0.01, epochs= 100, isAdaline = False, use_bias = True):
  
    # Initialization

    n_samples, n_features = X.shape
    w = np.random.rand(n_features)
    if use_bias:
        b = np.random.rand()
    else:
        b = 0

    # Training the model
    for j in range(epochs):
        for i in range(n_samples):
            # Calculate prediction
            if (isAdaline):
                 y_pred = np.dot(w, X[i]) + b 
            else:
                y_pred_continuous = np.dot(w, X[i]) + b
                y_pred = signum(y_pred_continuous)

            # Update weights and bias
            if y[i] != y_pred:
                loss = y[i] - y_pred
                w += learning_rate * loss * X[i]
                if use_bias:
                    b += learning_rate * loss
            
            MSE = compute_MSE(X,y,w,b)
            print('Epoch', j + 1, 'Sample:', i + 1, 'and the MSE is:', MSE)

        # compute MSE
        MSE = compute_MSE(X,y,w,b)
        print('Epoch', j + 1, 'Done with MSE:', MSE)
        if(MSE <= MSE_threshold):
            print('\nThreshold reached, terminating the training process. \n')
            break
        
    print(f"Algorithm: {'Adaline' if isAdaline else 'Perceptron'}")
    print(f"Learning rate: {learning_rate}")
    print(f"Max epochs: {epochs}")
    print(f"MSE threshold: {MSE_threshold}")
    print(f"Bias: {'Yes' if use_bias else 'NO'}")
    print('Final parameters: ')
    print('Weights:', w)
    print('Bias:', b)
    return w,b

