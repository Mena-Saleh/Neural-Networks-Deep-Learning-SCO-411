import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from joblib import dump, load


def TF_IDF_vectorize(x, is_test = False):
    if (is_test):
        vectorizer = load('TF_IDF_vectorizer.joblib')
        x = vectorizer.transform(x)
    else:
        vectorizer = TfidfVectorizer()
        x = vectorizer.fit_transform(x)
        dump(vectorizer,  'TF_IDF_vectorizer.joblib')
    return x
    