import tkinter as tk
import numpy as np
import pandas as pd
import Preprocess as pp
import Algorithms as ag
import Metrics as mt
import tkinter as tk
from tkinter import messagebox


def run(MSE_threshold, number_of_hidden_layers, number_of_nodes_in_each_layer, learning_rate , epochs, use_bias, isSigmoid):
    # Type casting the list of strings to a list of integers.
    number_of_nodes_in_each_layer = [int(x) for x in number_of_nodes_in_each_layer]

    # Read and preprocess
    X, y = pp.preprocess()

    # Train/Test split
    X_train, X_test, y_train, y_test = pp.train_test_split(X,y)

    # Train model
    W, B = ag.multi_layer_perceptron(X_train, y_train, MSE_threshold = MSE_threshold,
                                     number_of_hidden_layers = number_of_hidden_layers, 
                                     number_of_nodes_in_each_layer = number_of_nodes_in_each_layer, 
                                     learning_rate = learning_rate, 
                                     epochs = epochs, 
                                     use_bias = use_bias, 
                                     isSigmoid = isSigmoid)

    # Make predictions
    y_pred = [ag.predict(x,W,B, isSigmoid = isSigmoid) for x in X_test]

    #print(W)
    #print(B)
    #print(y_pred)
    
    # Evaluate model (temporarily built in code)

    from sklearn.metrics import accuracy_score

    # Calculate accuracy using sklearn's accuracy_score function for now
    accuracy = accuracy_score(y_test, y_pred)

    print("Accuracy:", accuracy)

    # You should use from scratch custom made confusion matrix and other metrics like accuracy

# GUI

# Define colors
colors = {
    "white": "#F2F2F2",
    "sunset": "#EFC88B",
    "burgundy": "#A5243D",
}



# Initialize main window
root = tk.Tk()
root.title("Multi Layer Perceptron NN Trainer")
root.config(bg=colors["white"])
root.geometry("800x520") 
root.resizable(False, False)
root.iconbitmap("icon.ico")



# Widgets

# Number of hidden layers entry
hidden_layers_count_label = tk.Label(root, text="Number of hidden layers:", bg=colors["white"])
hidden_layers_count_label.grid(row=2, column=0, padx=60, pady=(50, 10), sticky="w")

hidden_layers_count_entry = tk.Entry(root, width=26)
hidden_layers_count_entry.insert(0, 1)
hidden_layers_count_entry.grid(row=2, column=1, padx=30, pady=(50, 10), sticky="w")


# Number of nodes in each layer entry
hidden_nodes_count_label = tk.Label(root, text="Number of nodes in each layer (seperated by space)", bg=colors["white"])
hidden_nodes_count_label.grid(row=3, column=0, padx=60, pady=10, sticky="w")

hidden_nodes_count_entry = tk.Entry(root, width=26)
hidden_nodes_count_entry.insert(0, 5)
hidden_nodes_count_entry.grid(row=3, column=1, padx=30, pady=10, sticky="w")

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

# Activation radio button
activation_label = tk.Label(root, text="Activation Function:", bg=colors["white"])
activation_label.grid(row=8, column=0, padx=(60,35), pady=10, sticky="w")

is_sigmoid_var = tk.BooleanVar(value=False) 
sigmoid_radio = tk.Radiobutton(root, text="Sigmoid", variable=is_sigmoid_var, value= True, bg=colors["white"])
sigmoid_radio.grid(row=8, column=1, padx=20, pady=(10,0), sticky="w")

tanh_radion = tk.Radiobutton(root, text="Hyperbolic Tangent", variable=is_sigmoid_var, value= False, bg=colors["white"])
tanh_radion.grid(row=9, column=1, padx=20, pady=(0,10), sticky="w")

# Metrics label
metrics_label = tk.Label(root, bg=colors["white"])
metrics_label.place(x = 520, y = 40)

# Run button
run_btn = tk.Button(root, text="Run", bg=colors["burgundy"], fg=colors["white"], width=20, height=2, relief="flat")
run_btn.place(x = 550, y = 450 )



run_btn.config(command = lambda: run(
                             float(MSE_entry.get()), 
                             int(hidden_layers_count_entry.get()),
                             hidden_nodes_count_entry.get().split(" "),
                             float(learning_rate_entry.get()),
                             int(epochs_entry.get()), 
                             use_bias_var.get(),
                             is_sigmoid_var.get(),
                             ))

# Run the main window loop
root.mainloop()




