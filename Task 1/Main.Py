import tkinter as tk
import numpy as np
import pandas as pd
import Preprocess as pp
import Algorithms as ag


# Should make our own
from sklearn.metrics import confusion_matrix, accuracy_score


# GUI should be Implemented somewhere here in this file :)





# TEMPORARILY untill GUI is implemented:

# Read and preprocess
X, y = pp.preprocess()

# Filter features and classes
features = ['Area', 'Perimeter']
classes = ['CALI', 'BOMBAY']
X,y = pp.filter_data(X,y, features, classes)

# Train/Test split
X_train, X_test, y_train, y_test = pp.train_test_split(X,y)

# Train model
w, b = ag.single_layer_perceptron(X_train, y_train, isAdaline=True, use_bias = True)

# Make predictions
y_pred = ag.predict(X_test,w,b)

# Evaluate model (use from scratch code later rather than the built in one here)

cm = confusion_matrix(y_test, y_pred)
print(cm)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

