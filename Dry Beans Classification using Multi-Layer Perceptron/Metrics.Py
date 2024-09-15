import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tkinter import messagebox


# Draws the different class points and the regions of each class according to the model's predictions.
def visualize_model():
    print("To be implemented")

# Computes confusion matrix in addition to accuracy, recall and precision (on 3 classes rather than 2)
import numpy as np

def compute_confusion_matrix(y, y_pred, num_classes=3):
    # Initialize the confusion matrix with zeros
    matrix = np.zeros((num_classes, num_classes), dtype=int)

    # Populate the confusion matrix
    for true_label, pred_label in zip(y, y_pred):
        matrix[true_label, pred_label] += 1

    # Calculate true positives, false positives, and false negatives
    tp = np.diag(matrix)
    fp = np.sum(matrix, axis=0) - tp
    fn = np.sum(matrix, axis=1) - tp

    # Calculate precision, recall, and accuracy
    precision = np.divide(tp, tp + fp, where=(tp + fp) != 0)
    recall = np.divide(tp, tp + fn, where=(tp + fn) != 0)
    accuracy = np.sum(tp) / np.sum(matrix)

    # Calculate true negatives for each class
    tn = np.sum(matrix) - (fp + fn + tp)

    return {
        'confusion_matrix': matrix,
        'true_positives': tp,
        'true_negatives': tn,
        'false_positives': fp,
        'false_negatives': fn,
        'precision': precision,
        'recall': recall,
        'accuracy': accuracy
    }


def compute_accuracy(y,y_pred):
    return np.sum(y == y_pred)/len(y)







