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
    return y_pred

def compute_MSE(X,y,w,b):
    y_pred = predict(X,w,b)
    MSE = np.mean((y - y_pred)**2)
    return MSE

# Model training function
def single_layer_perceptron(X,y, MSE_threshold = 0.02 , learning_rate = 0.01, epochs= 100, isAdaline = False):
  
    # Initialization

    n_samples, n_features = X.shape
    w = np.random.rand(n_features)
    b = np.random.rand()

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
                b += learning_rate * loss
            
            MSE = compute_MSE(X,y,w,b)
            print('Epoch', j + 1, 'Sample:', i+ 1, 'and the MSE is:', MSE)

        # compute MSE
        MSE = compute_MSE(X,y,w,b)
        print('Epoch', j + 1, 'Done with MSE:', MSE)
        if(MSE <= MSE_threshold):
            print('\nThreshold reached, terminating the training process. \n')
            break
    
    print('Final parameters: ')
    print('Weight:', w)
    print('Bias:', b)
    return w,b

