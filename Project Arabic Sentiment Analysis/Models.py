import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import numpy as np


# Check for CUDA GPU availability
device_name = tf.test.gpu_device_name()
if device_name != '':
    print(f'Device GPU: {device_name}')
else:
    print('Device CPU: No GPU found, using CPU instead.')
    
    
def build_LSTM(input_shape, output_units=3):
    # Model architecture
    model = Sequential()
    model.add(LSTM(units=32, return_sequences=True, input_shape=input_shape, activation='tanh'))
    model.add(Dropout(0.2))
    # model.add(LSTM(units=128, activation='tanh'))
    # model.add(Dropout(0.2))
    model.add(Dense(units=output_units, activation='softmax'))
    
    # Compile the model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])
    
    return model


def train_evaluate_predict_model(model, x_train, y_train, x_val, y_val, x_test, y_test, model_path='model.h5'):
    # Define early stopping callback
    early_stopping = EarlyStopping(monitor='val_loss', patience=3)
    
    # Train the model
    history = model.fit(x_train, y_train, 
                        epochs=10, 
                        batch_size=32,
                        validation_data=(x_val, y_val),
                        callbacks=[early_stopping])
    
    # Evaluate the model on validation data
    val_loss, val_accuracy = model.evaluate(x_val, y_val, verbose=0)
    print(f"Validation Loss: {val_loss}")
    print(f"Validation Accuracy: {val_accuracy}")
    
    # Predict the classes on the test set
    y_pred_probabilities = model.predict(x_test)
    y_pred = np.argmax(y_pred_probabilities, axis=1)  # Convert probabilities to class predictions

    # Evaluate model
    model.evaluate(x_test, y_test)
    

    # Save the model
    model.save(model_path)
    print(f"Model saved to {model_path}")

