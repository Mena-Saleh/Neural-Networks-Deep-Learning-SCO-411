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

    # Encoding the classes

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