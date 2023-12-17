import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
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


def CountVectorize(x, is_test=False):
    if is_test:
        vectorizer = load('Count_vectorizer.joblib')
        x = vectorizer.transform(x)
    else:
        vectorizer = CountVectorizer()
        x = vectorizer.fit_transform(x)
        dump(vectorizer, 'Count_vectorizer.joblib')
    return x



# Tokenizes text, encodes it and then pads it all to a similar length to prepare for the embedding layer.
def prepare_for_embedding(texts, max_length=None, is_test=False):
    if is_test:
        # Load the tokenizer used during training
        tokenizer = load("Tokenizer.joblib")
    else:
        # Tokenization
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(texts)
        # Save the tokenizer for later use during testing
        dump(tokenizer, "Tokenizer.joblib")

    # Integer Encoding
    sequences = tokenizer.texts_to_sequences(texts)

    # Vocabulary size (+1 for padding token)
    vocab_size = len(tokenizer.word_index) + 1

    # Padding Sequences
    if max_length is None:
        max_length = max(len(sequence) for sequence in sequences)
    padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post')

    return padded_sequences, tokenizer, vocab_size
