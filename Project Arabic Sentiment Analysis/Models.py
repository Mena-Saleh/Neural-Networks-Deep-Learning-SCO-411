import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from keras.callbacks import ModelCheckpoint
import numpy as np
from keras.models import load_model
from keras.optimizers import Adam
from keras.layers import Bidirectional
from keras.layers import SimpleRNN
from keras.layers import GRU


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


# Train and evaluate model
def train_evaluate_model(model, x_train, y_train, x_val, y_val, x_test, y_test, model_path='Saved Models/model.h5', epochs = 20, batch_size = 32):
    # Define early stopping callback
    early_stopping = EarlyStopping(monitor='val_loss', patience=3)
    
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
    
   
    
    # Evaluate the best model on test data
    
    best_model = load_model(model_path)
    test_loss, val_accuracy = best_model.evaluate(x_test, y_test, verbose=0)
    
    print("best model")
    print(f"Test Loss: {test_loss}")
    print(f"Test Accuracy: {val_accuracy}")
    
    
  


