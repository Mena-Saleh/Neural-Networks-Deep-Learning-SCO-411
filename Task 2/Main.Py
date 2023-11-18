import tkinter as tk
import numpy as np
import pandas as pd
import Preprocess as pp
import Algorithms as ag
import Metrics as mt
import tkinter as tk
from tkinter import messagebox


def run():
    print("To be implemented")
   

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
sigmoid_radio = tk.Radiobutton(root, text="Sigmoid", variable=is_sigmoid_var, value= False, bg=colors["white"])
sigmoid_radio.grid(row=8, column=1, padx=20, pady=(10,0), sticky="w")

tanh_radion = tk.Radiobutton(root, text="Hyperbolic Tangent", variable=is_sigmoid_var, value= True, bg=colors["white"])
tanh_radion.grid(row=9, column=1, padx=20, pady=(0,10), sticky="w")

# Metrics label
metrics_label = tk.Label(root, bg=colors["white"])
metrics_label.place(x = 520, y = 40)

# Run button
run_btn = tk.Button(root, text="Run", bg=colors["burgundy"], fg=colors["white"], width=20, height=2, relief="flat")
run_btn.place(x = 550, y = 450 )



run_btn.config(command = lambda: run())

# Run the main window loop
root.mainloop()




