import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle


def TF_IDF_vectorize(x, is_test = False, vectorizer = TfidfVectorizer()):
    if (is_test):
        x = vectorizer.transform(x)
    else:
        x = vectorizer.fit_transform(x)
    return x
    