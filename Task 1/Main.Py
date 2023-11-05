import tkinter as tk
import numpy as np
import pandas as pd
import Preprocess as pp
import Algorithms as ag
import Metrics as mt
import tkinter as tk



# TEMPORARILY untill GUI is implemented:

# Read and preprocess
X, y = pp.preprocess()

# Filter features and classes
features = ['Roundness', 'Perimeter']
classes = ['CALI', 'BOMBAY']
X,y = pp.filter_data(X,y, features, classes)

# Train/Test split
X_train, X_test, y_train, y_test = pp.train_test_split(X,y)

# Train model
w, b = ag.single_layer_perceptron(X_train, y_train, isAdaline=True, use_bias = True)

# Make predictions
y_pred = ag.predict(X_test,w,b)

# Evaluate model 

# Confusion matrix and other metrics
mt.compute_confusion_matrix(y_test, y_pred)

# Model visualization
mt.visualize_model_boundary(X_test,y_test, w, b, features, classes)