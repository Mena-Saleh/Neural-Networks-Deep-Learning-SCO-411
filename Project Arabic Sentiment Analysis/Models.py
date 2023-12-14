import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from keras.callbacks import ModelCheckpoint
import numpy as np
from keras.models import load_model
    
def build_LSTM(input_shape, output_units=3):
    # Model architecture
    model = Sequential()
    model.add(LSTM(units=32, return_sequences=True, input_shape=input_shape, activation='tanh'))
    model.add(Dropout(0.3))
    model.add(LSTM(units=16, activation='tanh'))
    model.add(Dropout(0.3))
    model.add(Dense(units=output_units, activation='softmax'))
    
    # Compile the model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])
    
    return model


def train_evaluate_model(model, x_train, y_train, x_val, y_val, x_test, y_test, model_path='model.keras', epochs = 10, batch_size = 32):
    # Define early stopping callback
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=3)
    
    # Define the checkpoint callback to save the model with the highest validation accuracy 
    checkpoint = ModelCheckpoint(
        model_path,            
        monitor='val_accuracy',     
        verbose=1,                  
        save_best_only=True,        
        mode='max',                 
        save_freq='epoch')

    
    
    # Train the model
    history = model.fit(x_train, y_train, 
                        epochs=epochs, 
                        batch_size=batch_size,
                        validation_data=(x_val, y_val),
                        callbacks=[early_stopping, checkpoint])
    
    # Save final model (not necessarily the best)
    
    model.save('Last_' + model_path)
    
    # Evaluate final model on test data
    test_loss, val_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print("last model")
    print(f"Test Loss: {test_loss}")
    print(f"Test Accuracy: {val_accuracy}")
    
    
    # Evaluate the best model on test data
    
    best_model = load_model(model_path)
    test_loss, val_accuracy = best_model.evaluate(x_test, y_test, verbose=0)
    
    print("best model")
    print(f"Test Loss: {test_loss}")
    print(f"Test Accuracy: {val_accuracy}")
    
    
  


