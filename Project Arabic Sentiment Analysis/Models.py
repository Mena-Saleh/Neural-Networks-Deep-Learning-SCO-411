from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Bidirectional, SimpleRNN, GRU, Embedding
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
import numpy as np
import pandas as pd
from keras.optimizers import Adam



# Models architecture 
def build_lstm(input_shape, output_units=3, learning_rate=0.0001):
    # Model architecture
    model = Sequential()
    model.add(LSTM(units=32, return_sequences=True, input_shape=input_shape, activation='tanh'))
    model.add(Dropout(0.3))
    model.add(LSTM(units=32,  return_sequences=False, activation='tanh'))
    model.add(Dropout(0.3))
    model.add(Dense(units=output_units, activation='softmax'))
    
    # Compile the model with a specified learning rate
    adam_optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=adam_optimizer,
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])
    
    return model


def build_lstm_bidirectional(input_shape, output_units=3, learning_rate=0.0001):
    model = Sequential()
    model.add(Bidirectional(LSTM(units=32, activation='tanh'), input_shape=input_shape))
    model.add(Dense(units=output_units, activation='softmax'))
    
    adam_optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=adam_optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model


def build_simple_rnn(input_shape, output_units=3, learning_rate=0.0001):
    model = Sequential()
    model.add(SimpleRNN(units=32, activation='tanh', input_shape=input_shape))
    model.add(Dense(units=output_units, activation='softmax'))
    
    adam_optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=adam_optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model


def build_gru(input_shape, output_units=3, learning_rate=0.0001):
    model = Sequential()
    model.add(GRU(units=32, activation='tanh', input_shape=input_shape))
    model.add(Dense(units=output_units, activation='softmax'))
    
    adam_optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=adam_optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model


def build_embedding_lstm(vocab_size, embedding_dim, input_length, output_units=3, learning_rate=0.0001):
    # Model architecture
    model = Sequential()

    # Embedding layer
    model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=input_length))

    # LSTM layers
    model.add(LSTM(units=32, return_sequences=True, activation='tanh'))
    model.add(Dropout(0.2))
    model.add(LSTM(units=32, return_sequences=False, activation='tanh'))
    model.add(Dropout(0.2))

    # Output layer
    model.add(Dense(units=output_units, activation='softmax'))
    
    # Compile the model with a specified learning rate
    adam_optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=adam_optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model

# Train and evaluate model
def train_evaluate_model(model, x_train, y_train, x_val, y_val, model_path='Saved Models/model.h5', epochs = 20, batch_size = 32):
    # Define early stopping callback
    early_stopping = EarlyStopping(monitor='val_loss', patience=5)
    
    # Define the checkpoint callback to save the model with the highest validation accuracy 
    checkpoint = ModelCheckpoint(
        model_path,            
        monitor='val_loss',     
        verbose=1,                  
        save_best_only=True,        
        mode='min',                 
        save_freq='epoch')

    
    
    # Train the model
    history = model.fit(x_train, y_train, 
                        epochs=epochs, 
                        batch_size=batch_size,
                        validation_data=(x_val, y_val),
                        callbacks=[early_stopping, checkpoint])
    
    print("Best model saved")

    
  


