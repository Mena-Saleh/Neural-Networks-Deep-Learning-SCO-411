import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tkinter import messagebox


# Draws the different class points and the regions of each class according to the model's predictions.
def visualize_model():
    print("To be implemented")

# Computes confusion matrix in addition to accuracy, recall and precision (on 3 classes rather than 2)
def compute_confusion_matrix(y, y_pred,num_classes=3):
    # Initialize the confusion matrix with zeros
    matrix = np.zeros((num_classes, num_classes), dtype=int)

    # Populate the confusion matrix
    for true_label, pred_label in zip(y, y_pred):
        matrix[true_label, pred_label] += 1

    # Calculate true positives, true negatives, false positives, and false negatives
    tp = np.diag(matrix)
    fp = np.sum(matrix, axis=0) - tp
    fn = np.sum(matrix, axis=1) - tp
    tn = np.sum(np.sum(matrix)) - (tp + fp + fn)

    # Calculate precision, recall, and accuracy
    precision = tp / (tp + fp) if np.any(tp + fp != 0) else np.zeros(num_classes)
    recall = tp / (tp + fn) if np.any(tp + fn != 0) else np.zeros(num_classes)
    accuracy = np.sum(tp) / np.sum(matrix)

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



