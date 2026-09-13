import pandas as pd
import os
import random
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (mean_absolute_error,mean_squared_error,r2_score)
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

df = pd.read_csv("MLP_NSGA_Dataset.csv")

print("Original shape:", df.shape)
display(df.head())

required_columns = [
    "t_r",
    "alpha_r",
    "t_c",
    "N",
    "L_s",
    "position",
    "orientation",
    "delta_x",
    "delta_y",
    "delta_z",
    "F_res"
]

df = df[required_columns].copy()

print("Columns retained:")
print(df.columns.tolist())

df = df.dropna(
    subset=required_columns
).reset_index(drop=True)

print("Cleaned dataset shape:", df.shape)

print("Positions:")
print(sorted(df["position"].unique()))

print("\nOrientations:")
print(sorted(df["orientation"].unique()))

geometry_columns = [
    "t_r",
    "alpha_r",
    "t_c",
    "N",
    "L_s"
]

geometry_counts = (
    df.groupby(geometry_columns)
      .size()
      .reset_index(name="rows_per_geometry")
)

print(
    "Number of geometries:",
    len(geometry_counts)
)
print(
    geometry_counts[
        "rows_per_geometry"
    ].value_counts()
)

display(geometry_counts.head())

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

input_columns = [
    "t_r",
    "alpha_r",
    "t_c",
    "N",
    "L_s",
    "position",
    "orientation"
]

output_columns = [
    "delta_x",
    "delta_y",
    "delta_z",
    "F_res"
]

X = df[input_columns].copy()
Y = df[output_columns].copy()

print("Input shape :", X.shape)
print("Output shape:", Y.shape)

for column in input_columns:

    print(
        f"{column:15s}"
        f" min = {df[column].min():8.4f}"
        f"   max = {df[column].max():8.4f}"
    )

X_train, X_temp, Y_train, Y_temp = train_test_split(
    X,
    Y,
    test_size=0.20,
    random_state=SEED,
    shuffle=True
)

X_val, X_test, Y_val, Y_test = train_test_split(
    X_temp,
    Y_temp,
    test_size=0.50,
    random_state=SEED,
    shuffle=True
)

print("Training samples  :", len(X_train))
print("Validation samples:", len(X_val))
print("Testing samples   :", len(X_test))

X_scaler = MinMaxScaler()

X_train_scaled = X_scaler.fit_transform(
    X_train
)

X_val_scaled = X_scaler.transform(
    X_val
)

X_test_scaled = X_scaler.transform(
    X_test
)
print("Input scaling completed.")
Y_scaler = MinMaxScaler()

Y_train_scaled = Y_scaler.fit_transform(
    Y_train
)

Y_val_scaled = Y_scaler.transform(
    Y_val
)

Y_test_scaled = Y_scaler.transform(
    Y_test
)

print("Output scaling completed.")

model = keras.Sequential([

    layers.Input(
        shape=(7,)
    ),
    layers.Dense(
        256,
        activation="relu",
        name="hidden_1"
    ),
    layers.Dense(
        128,
        activation="relu",
        name="hidden_2"
    ),
    layers.Dense(
        128,
        activation="relu",
        name="hidden_3"
    ),

    layers.Dense(
        64,
        activation="relu",
        name="hidden_4"
    ),

    layers.Dense(
        32,
        activation="relu",
        name="hidden_5"
    ),

    layers.Dense(
        4,
        activation="linear",
        name="outputs"
    )

])

model.summary()

optimizer = keras.optimizers.Adam(
    learning_rate=0.0005
)

model.compile(
    optimizer=optimizer,
    loss="mse",
    metrics=["mae"]
)

print("Model compiled.")