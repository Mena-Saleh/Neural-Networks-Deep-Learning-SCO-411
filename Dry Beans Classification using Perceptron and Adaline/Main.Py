import tkinter as tk
import numpy as np
import pandas as pd
import Preprocess as pp
import Algorithms as ag
import Metrics as mt
import tkinter as tk
from tkinter import messagebox


def run(metrics_label_object: tk.Label, features, classes , MSE_threshold , learning_rate , epochs, isAdaline, use_bias):

    # Validation
    if features[0] == features[1]:
        messagebox.showerror('Validation error', 'Selected features must be different.')
        return

    if classes[0] == classes[1]:
        messagebox.showerror('Validation error', 'Selected classes must be different.')
        return

    # Read and preprocess
    X, y = pp.preprocess()

    # Filter features and classes
    X,y = pp.filter_data(X,y, features, classes)

    # Train/Test split
    X_train, X_test, y_train, y_test = pp.train_test_split(X,y)

    # Train model
    w, b = ag.single_layer_perceptron(X_train, y_train, MSE_threshold = MSE_threshold, learning_rate = learning_rate, epochs = epochs, isAdaline=isAdaline, use_bias = use_bias)

    # Make predictions
    y_pred = ag.predict(X_test,w,b)

    # Evaluate model 

    # Confusion matrix and other metrics
    tp,tn,fp,fn,accuracy,precision,recall = mt.compute_confusion_matrix(y_test, y_pred)

    # Displaying metrics on the metrics_label object

    metrics = f"""Confusion Matrix:

    TP: {tp}     FP: {fp}
    FN: {fn}     TN: {tn}

    Accuracy: {accuracy*100 :.2f}%
    Precision: {precision}
    Recall: {recall}

    Algorithm: {'Adaline' if isAdaline else 'Perceptron'}
    Learning rate: {learning_rate}
    Max epochs: {epochs}
    MSE threshold: {MSE_threshold}
    Bias: {'Yes' if use_bias else 'NO'}

    Final parameters: 
    Weights: {w}
    Bias: {b}"""

    
    metrics_label_object.config(text=metrics)

    # Model visualization
    mt.visualize_model_boundary(X_test,y_test, w, b, features, classes)



# GUI

# Define colors
colors = {
    "white": "#F2F2F2",
    "sunset": "#EFC88B",
    "burgundy": "#A5243D",
}



# Initialize main window
root = tk.Tk()
root.title("Perceptron and Adaline NN Trainer")
root.config(bg=colors["white"])
root.geometry("800x520") 
root.resizable(False, False)
#root.iconbitmap("icon.ico")


# Features and Classes
features = ["Area", "Perimeter", "MajorAxisLength", "MinorAxisLength", "Roundness"]
classes = ["BOMBAY", "CALI", "SIRA"]


# Widgets

# Feature 1 dropdown
feature1_label = tk.Label(root, text="Feature 1:", bg=colors["white"])
feature1_label.grid(row=0, column=0, padx=60, pady=(40,5), sticky="w")

feature1_var = tk.StringVar(root)
feature1_var.set(features[0])  # set default value
feature1 = tk.OptionMenu(root, feature1_var, *features)
feature1.config(width=20, height=1)
feature1.grid(row=0, column=1, padx=30, pady=(40,5), sticky="w")

# Feature 2 dropdown
feature2_label = tk.Label(root, text="Feature 2:", bg=colors["white"])
feature2_label.grid(row=1, column=0, padx=60, pady=5, sticky="w")

feature2_var = tk.StringVar(root)
feature2_var.set(features[1])  # set default value
feature2 = tk.OptionMenu(root, feature2_var, *features)
feature2.config(width=20, height=1)
feature2.grid(row=1, column=1, padx=30, pady=5, sticky="w")

# Class 1 dropdown
class1_label = tk.Label(root, text="Class 1:", bg=colors["white"])
class1_label.grid(row=2, column=0, padx=60, pady=5, sticky="w")

class1_var = tk.StringVar(root)
class1_var.set(classes[0])  # set default value
class1 = tk.OptionMenu(root, class1_var, *classes)
class1.config(width=20, height=1)
class1.grid(row=2, column=1, padx=30, pady=5, sticky="w")

# Class 2 dropdown
class2_label = tk.Label(root, text="Class 2:", bg=colors["white"])
class2_label.grid(row=3, column=0, padx=60, pady=(5,50), sticky="w")

class2_var = tk.StringVar(root)
class2_var.set(classes[1])  # set default value
class2 = tk.OptionMenu(root, class2_var, *classes)
class2.config(width=20, height=1)
class2.grid(row=3, column=1, padx=30, pady=(5,50), sticky="w")

# Learning rate entry
learning_rate_label = tk.Label(root, text="Learning rate:", bg=colors["white"])
learning_rate_label.grid(row=4, column=0, padx=60, pady=10, sticky="w")

learning_rate_entry = tk.Entry(root, width=26)
learning_rate_entry.insert(0, 0.01)
learning_rate_entry.grid(row=4, column=1, padx=30, pady=10, sticky="w")

# Epochs entry
epochs_label = tk.Label(root, text="Number of epochs:", bg=colors["white"])
epochs_label.grid(row=5, column=0, padx=60, pady=10, sticky="w")

epochs_entry = tk.Entry(root, width=26)
epochs_entry.insert(0, 100)
epochs_entry.grid(row=5, column=1, padx=30, pady=10, sticky="w")

# MSE entry
MSE_label = tk.Label(root, text="MSE threshold:", bg=colors["white"])
MSE_label.grid(row=6, column=0, padx=60, pady=10, sticky="w")

MSE_entry = tk.Entry(root, width=26)
MSE_entry.insert(0, 0.02)
MSE_entry.grid(row=6, column=1, padx=30, pady=10, sticky="w")

# Bias checkbox
use_bias_var = tk.BooleanVar(value=True)
bias_check = tk.Checkbutton(root, text="Use Bias", variable= use_bias_var, bg=colors["white"],)
bias_check.grid(row=7, column=0, columnspan=2, sticky = 'w',  padx=60, pady=10,)

# Algorithm radiobutton
algorithm_label = tk.Label(root, text="Algorithm:", bg=colors["white"])
algorithm_label.grid(row=8, column=0, padx=(60,35), pady=10, sticky="w")

is_adaline_var = tk.BooleanVar(value=False) 
perceptron_radio = tk.Radiobutton(root, text="Perceptron", variable=is_adaline_var, value= False, bg=colors["white"])
perceptron_radio.grid(row=8, column=1, padx=20, pady=(10,0), sticky="w")

adaline_radio = tk.Radiobutton(root, text="Adaline", variable=is_adaline_var, value= True, bg=colors["white"])
adaline_radio.grid(row=9, column=1, padx=20, pady=(0,10), sticky="w")

# Metrics label
metrics_label = tk.Label(root, bg=colors["white"])
metrics_label.place(x = 520, y = 40)

# Run button
run_btn = tk.Button(root, text="Run", bg=colors["burgundy"], fg=colors["white"], width=20, height=2, relief="flat")
run_btn.place(x = 550, y = 450 )



run_btn.config(command = lambda: run(metrics_label, [feature1_var.get(), feature2_var.get()], 
                             [class1_var.get(), class2_var.get()], 
                             float(MSE_entry.get()), 
                             float(learning_rate_entry.get()),
                             int(epochs_entry.get()), 
                             is_adaline_var.get(),use_bias_var.get()),
                             )

# Run the main window loop
root.mainloop()




