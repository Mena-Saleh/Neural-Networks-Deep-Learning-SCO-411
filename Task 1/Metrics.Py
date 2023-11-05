import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tkinter import messagebox


# Draws the different class points and the decision boundary seperating them
def visualize_model_boundary(X, y, w, b, features, classes):
    # Scatter the points and color each class
    plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], c='navy', label=f'{classes[0]}', s = 5)
    plt.scatter(X[y == -1][:, 0], X[y == -1][:, 1], c='crimson', label=f'{classes[1]}', s = 5)

    # Plotting the seperator line according to the equation: w.X + b = 0, w1.x1 + w2.x2 + b = 0, x2 = (-b - w1.x1) / w2
    plt.plot(X[:,0], (-b - w[0]*X[:,0]) / w[1], label='Separator', c ='seagreen')

    plt.xlabel(f'{features[0]}')
    plt.ylabel(f'{features[1]}')
    plt.legend()
    plt.title('Classes and Decision Boundary')

    plt.show()

# Computes confusion matrix in addition to accuracy, recall and precision
def compute_confusion_matrix(y, y_pred):
    # Supposing class 1 is true and class 2 is false , class1 = 1 and class2 = -1
    # tp -> actually class 1 and predicted class 1, tn -> actually class 2 and predicted class 2
    # fp -> actually class 2 and predicted class 1, fn -> actually class 1 and predicted class 2

    tp = np.sum((y == 1) & (y_pred == 1))
    tn = np.sum((y == -1) & (y_pred == -1))
    fp = np.sum((y == -1) & (y_pred == 1))
    fn = np.sum((y == 1) & (y_pred == -1))

    # Printing the matrix
    print(f'TP:{tp}     FP:{fp}\nFN:{fn}     TN:{tn}')
    #messagebox.showinfo('CM', f'TP:{tp}     FP:{fp}\nFN:{fn}     TN:{tn}')

    # Other metrics
    accuracy = (tp + tn)/(tp + tn + fp + fn)
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)

    print(f'Accuracy: {accuracy * 100:.2f}%')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')

    return tp,tn,fp,fn,accuracy,precision,recall




