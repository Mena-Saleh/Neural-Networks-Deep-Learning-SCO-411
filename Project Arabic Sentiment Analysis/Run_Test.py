from keras.models import load_model
import pandas as pd
import Preprocess as pp
import Feature_Extraction as fe
import numpy as np

# Read and preprocess test data
df = pd.read_csv("test_no_label.csv")
preprocessed_df = pp.preprocess_df(df, "train", isPredict=True)


# Extact features

# TF-IDF
tf_idf = fe.TF_IDF_vectorize(preprocessed_df['review_description'], is_test=True)

# Convert to array and reshape
tf_idf = tf_idf.toarray()
tf_idf = np.reshape(tf_idf, newshape=(tf_idf.shape[0],1, tf_idf.shape[1]))

# Load model and predict
best_model = load_model('Saved Models/Bidirectional LSTM.h5')

# Gets probabilities of all classes using softmax function
predictions = best_model.predict(tf_idf)

# Gets max probability (result is 0,1,2, need to map back to -1, 0, 1, can do that by subtracting 1)
predicted_classes = np.argmax(predictions, axis=1)
predicted_classes = predicted_classes - 1


# Save as CSV in the correct format (ID, rating)

new_df = pd.DataFrame(df.iloc[:, 0])
new_df['rating'] = predicted_classes

new_df.to_csv('95.csv', index= False)
print("Saved to csv file successfully.")

