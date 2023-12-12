import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import Preprocess as pp
import Feature_Extraction as fe
import Models as md

# [ONLY NEED TO RUN ONCE AND THEN IT SAVES PREPROCESSED DATA]

# # Read data
# df = pd.read_excel("train.xlsx")

# # Split into training and (validation + test)
# train_df, val_test_df = train_test_split(df, test_size=0.3, random_state=42)

# # Split the (validation + test) into validation and test
# val_df, test_df = train_test_split(val_test_df, test_size=0.5, random_state=42)


# # Preprocess each df
# train_df = pp.preprocess_df(train_df, "train", num_samples= 3000)
# val_df = pp.preprocess_df(val_df, "val", num_samples= 300)
# test_df = pp.preprocess_df(test_df, "test", num_samples= 300)





# Read preprocssed data
train_df = pd.read_excel("preprocessed_train.xlsx").fillna("")
val_df = pd.read_excel("preprocessed_val.xlsx").fillna("")
test_df = pd.read_excel("preprocessed_test.xlsx").fillna("")

# Extract features

# TF-IDF on each set
tf_idf_train = fe.TF_IDF_vectorize(train_df['review_description'])
tf_idf_val = fe.TF_IDF_vectorize(val_df['review_description'], is_test= True)
tf_idf_test = fe.TF_IDF_vectorize(test_df['review_description'], is_test= True)

# Convert the sparse TF-IDF matrices to dense format
tf_idf_train_dense = tf_idf_train.todense()
tf_idf_val_dense = tf_idf_val.todense()
tf_idf_test_dense = tf_idf_test.todense()

# Expand dimensions
tf_idf_train_expanded = np.expand_dims(tf_idf_train_dense, axis=-1)
tf_idf_val_expanded = np.expand_dims(tf_idf_val_dense, axis=-1)
tf_idf_test_expanded = np.expand_dims(tf_idf_test_dense, axis=-1)

# Target value of each set
y_train = train_df['rating']
y_val = val_df['rating']
y_test = test_df['rating']


#print(tf_idf_train)



# Train and evaluate models

#1 LSTM
LSTM_model = md.build_LSTM(input_shape=(tf_idf_train_expanded.shape[1], tf_idf_train_expanded.shape[2]))
md.train_evaluate_predict_model(LSTM_model, tf_idf_train_expanded, y_train, tf_idf_val_expanded, y_val, tf_idf_test_expanded, y_test, 'LSTM.h5')