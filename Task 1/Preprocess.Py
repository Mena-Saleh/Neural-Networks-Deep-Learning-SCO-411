import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler



def preprocess(): 
    # Reading data
    df = pd.read_excel('Dry_Bean_Dataset.xlsx')

    y = df['Class']
    X = df.drop('Class', axis=1)

    # Normalizing
    scaler = StandardScaler()

    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)


    # Filling missing values

    X = X.fillna(X.mean())

    return X,y


def filter_data(X,y, features, classes):
    # Filtering out user selected data
    X = X[features]
    # To filter out unwanted rows that has the third class that is not chosen by the user
    mask = np.isin(y, classes)
    y = y[mask]
    X = X[mask]
    X = X.to_numpy()

    # Finally, encoding the chosen classes to be 1 and -1
    y = np.where(y == classes[0], 1, -1)
    return X, y


# Use it after filtering the data
def train_test_split(X,y, train_per_class=30, test_per_class=20):
    # Shuffling the two classes' indices randomly
    indices_class1 = np.random.permutation(np.arange(50)) 
    indices_class2 = np.random.permutation(np.arange(50, 100)) 

    # Selecting the train and test indices for each class
    train_indices_class1 = indices_class1[:train_per_class]
    test_indices_class1 = indices_class1[train_per_class:train_per_class + test_per_class]
    train_indices_class2 = indices_class2[:train_per_class]
    test_indices_class2 = indices_class2[train_per_class:train_per_class + test_per_class]

    # Concatenating results
    train_indices = np.concatenate((train_indices_class1, train_indices_class2))
    test_indices = np.concatenate((test_indices_class1, test_indices_class2))
    
    # Splitting according to indices
    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]

    return X_train, X_test, y_train, y_test
