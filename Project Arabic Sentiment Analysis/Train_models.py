import tensorflow as tf
import Preprocess as pp
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# Read data
df = pd.read_excel("train.xlsx")

# Split into training and (validation + test)
train_df, val_test_df = train_test_split(df, test_size=0.3, random_state=42)

# Split the (validation + test) into validation and test
val_df, test_df = train_test_split(val_test_df, test_size=0.5, random_state=42)


# Preprocess each df
train_df = pp.preprocess_df(df, "train", num_samples=10)
val_df = pp.preprocess_df(df, "val", num_samples=10)
test_df = pp.preprocess_df(df, "test", num_samples=10)

# Extract features


# Train and evaluate models



