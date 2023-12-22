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
# train_df, val_df = train_test_split(df, test_size = 1000, random_state=42)

# # Preprocess each df (and save to a file)
# train_df = pp.preprocess_df(train_df, "train")
# val_df = pp.preprocess_df(val_df, "val")


# Read preprocssed data
train_df = pd.read_excel("preprocessed_train.xlsx").fillna("")
val_df = pd.read_excel("preprocessed_val.xlsx").fillna("")

# Extract features

# #1 TF-IDF
# # TF-IDF on each set
# X_train_tf_idf = fe.TF_IDF_vectorize(train_df['review_description'])
# X_val_tf_idf = fe.TF_IDF_vectorize(val_df['review_description'], is_test= True)

# # Save original shape for model first layer input shape specification.
# model_shape = X_train_tf_idf.shape

# # Convert to arrays and reshape for the sequential models.
# X_train_tf_idf= X_train_tf_idf.toarray()
# X_train_tf_idf = np.reshape(X_train_tf_idf, newshape=(X_train_tf_idf.shape[0],1, X_train_tf_idf.shape[1]))
# X_val_tf_idf= X_val_tf_idf.toarray()
# X_val_tf_idf = np.reshape(X_val_tf_idf, newshape=(X_val_tf_idf.shape[0],1, X_val_tf_idf.shape[1]))


# # Print all shapes
# print("Neural Network shape:", model_shape)
# print("Train set shape:", X_train_tf_idf.shape)
# print("Validation set shape:", X_val_tf_idf.shape)


#2 Embedding using embedding layers

X_train_embedding, tokenizer, vocab_size = fe.prepare_for_embedding(train_df['review_description'].tolist(), max_length= 120)
X_val_embedding, _, _= fe.prepare_for_embedding(val_df['review_description'].tolist(),max_length= X_train_embedding.shape[1], is_test= True)

# Print all shapes
print("Train set shape:", X_train_embedding.shape)
print("Validation set shape:", X_val_embedding.shape)


# Target value of each set
y_train = train_df['rating']
y_val = val_df['rating']


# Train and evaluate models

#1 LSTM
# print("#1 LSTM model \n")
# lstm_model = md.build_lstm(input_shape=(1, model_shape[1]))
# md.train_evaluate_model(lstm_model, X_train_tf_idf, y_train, X_val_tf_idf, y_val, 'Saved Models/LSTM.h5')
# print("\n")

# #2 Bidirectional LSTM
# print("#2 Bidirectional LSTM model \n")
# bidirectional_lstm_model = md.build_lstm_bidirectional(input_shape=(1, model_shape[1]))
# md.train_evaluate_model(bidirectional_lstm_model, X_train_tf_idf, y_train, X_val_tf_idf, y_val, 'Saved Models/Bidirectional LSTM.h5')
# print("\n")

# #3 Simple RNN
# print("#3 Simple RNN model \n")
# simple_rnn_model = md.build_simple_rnn(input_shape=(1, model_shape[1]))
# md.train_evaluate_model(simple_rnn_model, X_train_tf_idf, y_train, X_val_tf_idf, y_val, 'Saved Models/Simple RNN.h5')
# print("\n")

# #4 GRU
# print("#4 GRU model \n")
# gru_model = md.build_gru(input_shape=(1, model_shape[1]))
# md.train_evaluate_model(gru_model, X_train_tf_idf, y_train, X_val_tf_idf, y_val, 'Saved Models/GRU.h5')
# print("\n")


#5 Embedding LSTM
# print("#5 Embedding LSTM \n")
# embedding_lstm_model = md.build_embedding_lstm(vocab_size=vocab_size, embedding_dim= 25, input_length= X_train_embedding.shape[1], learning_rate= 0.0001)
# md.train_evaluate_model(embedding_lstm_model, X_train_embedding, y_train, X_val_embedding, y_val, 'Saved Models/Embedding LSTM.h5', batch_size=256, use_early_stopping= True)

#6 Transformer
print("#6 Transformer \n")
transformer_model = md.build_transformer(vocab_size=vocab_size, embedding_dim=10, num_heads=5, num_transformer_layers=1, input_length= X_train_embedding.shape[1])
md.train_evaluate_model(transformer_model, X_train_embedding, y_train, X_val_embedding, y_val, 'Saved Models/Transformer.h5', batch_size=32, use_early_stopping= True, epochs= 20)

