import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import Preprocess as pp
import Feature_Extraction as fe
import Models as md

# [ONLY NEED TO RUN ONCE AND THEN IT SAVES PREPROCESSED DATA]

# Read data
# df = pd.read_excel("train.xlsx")

# # Split into training and (validation + test)
# train_df, val_test_df = train_test_split(df, test_size = 2000, random_state=42)

# # Split the (validation + test) into validation and test
# val_df, test_df = train_test_split(val_test_df, test_size = 1000, random_state=42)


# # Preprocess each df
# train_df = pp.preprocess_df(train_df, "train")
# val_df = pp.preprocess_df(val_df, "val")
# test_df = pp.preprocess_df(test_df, "test")




# Read preprocssed data
train_df = pd.read_excel("preprocessed_train.xlsx").fillna("")
val_df = pd.read_excel("preprocessed_val.xlsx").fillna("")
test_df = pd.read_excel("preprocessed_test.xlsx").fillna("")

# Extract features

# TF-IDF on each set
tf_idf_train = fe.TF_IDF_vectorize(train_df['review_description'])
tf_idf_val = fe.TF_IDF_vectorize(val_df['review_description'], is_test= True)
tf_idf_test = fe.TF_IDF_vectorize(test_df['review_description'], is_test= True)

# Save original shape for model first layer input shape specification.
model_shape = tf_idf_train.shape

# Convert to arrays and reshape for the sequential models.
tf_idf_train= tf_idf_train.toarray()
tf_idf_train = np.reshape(tf_idf_train, newshape=(tf_idf_train.shape[0],1, tf_idf_train.shape[1]))
tf_idf_val= tf_idf_val.toarray()
tf_idf_val = np.reshape(tf_idf_val, newshape=(tf_idf_val.shape[0],1, tf_idf_val.shape[1]))
tf_idf_test= tf_idf_test.toarray()
tf_idf_test = np.reshape(tf_idf_test, newshape=(tf_idf_test.shape[0],1, tf_idf_test.shape[1]))


# Print all shapes
print("Neural Network shape:", model_shape)
print("Train set shape:", tf_idf_train.shape)
print("Validation set shape:", tf_idf_val.shape)
print("Test set shape:", tf_idf_test.shape)



# Target value of each set
y_train = train_df['rating']
y_val = val_df['rating']
y_test = test_df['rating']


# Train and evaluate models

#1 LSTM
LSTM_model = md.build_LSTM(input_shape=(1, model_shape[1]))
md.train_evaluate_model(LSTM_model, tf_idf_train, y_train, tf_idf_val, y_val, tf_idf_test, y_test, 'LSTM.keras')