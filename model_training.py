'''
Model training script
Author: molingc1
'''
# ================================================
# -*- coding: utf-8 -*-
# This script depends on an external module: scaler.py
# Make sure scaler.py (with PiecewiseScaler class) is in the same directory.
# ================================================

import h5py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
import pandas_ta as ta
import joblib
from sklearn.preprocessing import PowerTransformer, FunctionTransformer, MinMaxScaler
from scaler import PiecewiseScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

filename = ["grating012umpitch05dutycycle15um", "grating012umpitch05dutycycle20um", "grating12_11pitch2_8","grating12pitch100"]
link = ["grating012umpitch05dutycycle15um.h5", "grating012umpitch05dutycycle20um.h5"]

file = "grating012umpitch05dutycycle60um"

# Load h5 file
with h5py.File(file+".h5", 'r') as f:
    # Load dataset
    dset = f[file]
    # Convert to numpy array
    arr_3d_loaded = dset[()]

# Downsample
arr_3d_loaded = arr_3d_loaded[::1,::1,100::10]

print(arr_3d_loaded.shape)

df_data_main_main = pd.DataFrame()

for y in range(arr_3d_loaded.shape[1]):
    df = arr_3d_loaded[:, y, :]
    df = df.transpose()

    df_data_main = pd.DataFrame()
    for z in range(arr_3d_loaded.shape[2]):
        e = df[z, :]
        df_data = pd.DataFrame()
        df_data['e'] = e
        # df_data['after10'] = df_data['e'].shift(-10)
        # df_data['after20'] = df_data['e'].shift(-20)
        # df_data['after3'] = df_data['e'].shift(-30)
        df_data['e_diff10'] = df_data['e'] - df_data['e'].shift(-10)  # Difference over 10 steps
        df_data['e_diff20'] = df_data['e'] - df_data['e'].shift(-20)  # Difference over 20 steps
        # df_data['e_diff30'] = df_data['e'] - df_data['e'].shift(-30)
        df_data['x'] = range(arr_3d_loaded.shape[0])
        df_data['RSI'] = ta.rsi(df_data['e'], length = 30)
        df_data['EMAS'] = ta.ema(df_data['e'], length = 10)
        df_data['EMAM'] = ta.ema(df_data['e'], length = 20)
        df_data['EMAL'] = ta.ema(df_data['e'], length = 30)
        for i in range(1, 30):
            df_data[f'next{i}'] = df_data['e'].shift(-i)

        # Clean data
        df_data.dropna(inplace = True)
        df_data.reset_index(inplace = True, drop = True)

        df_data_main = pd.concat([df_data_main, df_data], axis = 0)
        print(y, z)
    df_data_main_main = pd.concat([df_data_main_main, df_data_main], axis = 0)

'''--------------------------- YJ Transformation -------------------------------'''
df = df_data_main_main.copy()

# Feature columns
feature_cols = ['e','e_diff10','e_diff20','x','RSI','EMAS','EMAM','EMAL']
# Label columns
X_df = df[feature_cols]
y_df = df.iloc[:, 10:]

scalers_x = {}
scalers_y = {}
X_trans = pd.DataFrame(index=X_df.index, columns=feature_cols)
y_trans = pd.DataFrame()

# Feature scaling
for col in feature_cols:
    data = X_df[col].values.astype(float)
    if col == 'e':
        scaler = PiecewiseScaler()
        scaler.fit(data)
        transformed = scaler.transform(data)
        scalers_x[col] = scaler
    else:
        scaler = MinMaxScaler()
        transformed = scaler.fit_transform(data.reshape(-1, 1)).flatten()
        scalers_x[col] = scaler
    X_trans[col] = transformed
    print(f"Column {col} processed.")

joblib.dump(scalers_x, 'scalers_features.pkl')
print("Feature scalers saved to scalers_features.pkl")

# Label scaling
for col in range(29):
    data = y_df.iloc[:, col].values.astype(float)
    scaler = PiecewiseScaler()
    scaler.fit(data)
    transformed = scaler.transform(data)
    scalers_y[col] = scaler
    y_trans[col] = transformed

joblib.dump(scalers_y, 'scalers_labels.pkl')
print("Label scalers saved to scalers_labels.pkl")

# Reshape for model input
X_train = X_trans.values.reshape(X_trans.shape[0], X_trans.shape[1], 1)
y_train = y_trans.values

'''--------------------------- Training -------------------------------'''
# GPU config
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("✅ Enabled GPU memory growth.")
    except RuntimeError as e:
        print(e)

# Transformer model
num_transformer_blocks = 6
num_heads = 4
ff_dim = 8
head_size = 128

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = layers.MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(x, x)
    x = layers.Dropout(dropout)(x)
    res = x + inputs

    x = layers.LayerNormalization(epsilon=1e-6)(res)
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    return x + res

def build_model(input_shape, head_size, num_heads, ff_dim, num_transformer_blocks, mlp_units, dropout=0, mlp_dropout=0):
    inputs = keras.Input(shape=input_shape)
    x = inputs
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)
    x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
    for dim in mlp_units:
        x = layers.Dense(dim, activation="relu")(x)
        x = layers.Dropout(mlp_dropout)(x)
    outputs = layers.Dense(y_train.shape[1])(x)
    return keras.Model(inputs, outputs)

# Inverse transform for labels
def inv_transform_labels(y_transformed):
    y_original = np.empty_like(y_transformed)
    n_labels = y_transformed.shape[1]
    for i in range(n_labels):
        col_data = y_transformed[:, i].reshape(-1, 1)
        y_original[:, i] = scalers_y[i].inverse_transform(col_data).flatten()
    return y_original

# Custom Callback
class SortedRelativeErrorCallback(tf.keras.callbacks.Callback):
    def __init__(self, X, y, inv_transform_func):
        super().__init__()
        self.X = X
        self.y = y
        self.inv_transform_func = inv_transform_func

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % 20 == 0:
            predictions = self.model.predict(self.X, verbose=0)
            y_pred_original = self.inv_transform_func(predictions)
            y_train_original = self.inv_transform_func(self.y)
            relative_mse = np.mean((y_pred_original - y_train_original) ** 2, axis=1) / (np.mean(y_train_original ** 2, axis=1) + 1e-8)
            sample_means = np.mean(y_train_original, axis=1)
            sorted_indices = np.argsort(sample_means)
            sorted_y_train = sample_means[sorted_indices]
            sorted_relative_mse = relative_mse[sorted_indices]

            plt.figure(figsize=(8, 6))
            sns.set(style="whitegrid", context="talk")
            colors = sns.color_palette("viridis", as_cmap=True)
            scatter = plt.scatter(sorted_y_train, sorted_relative_mse * 100, c=sorted_y_train,
                                  cmap=colors, edgecolors='k', alpha=0.75)
            plt.xlabel("Value", fontsize=16, fontweight='bold')
            plt.ylabel("Relative MSE (%)", fontsize=16, fontweight='bold')
            plt.title(f"Relative MSE vs. Value (Epoch {epoch+1})", fontsize=18, fontweight='bold')
            plt.xticks(fontsize=14, fontweight='bold')
            plt.yticks(fontsize=14, fontweight='bold')
            plt.gca().spines['top'].set_linewidth(1.5)
            plt.gca().spines['right'].set_linewidth(1.5)
            plt.gca().spines['bottom'].set_linewidth(1.5)
            plt.gca().spines['left'].set_linewidth(1.5)
            cbar = plt.colorbar(scatter)
            cbar.set_label("Value Intensity", fontsize=14, fontweight='bold')
            plt.grid(True, linestyle="--", alpha=0.6)
            plt.tight_layout()
            plt.show()

# Sample data
sample_indices = np.random.choice(len(y_train), size=100, replace=False)
sample_X = X_train[sample_indices]
sample_y = y_train[sample_indices]
print("x_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)

loss_yj_callback = SortedRelativeErrorCallback(sample_X, sample_y, inv_transform_func=inv_transform_labels)

# Build model
input_shape = X_train.shape[1:]
model = build_model(
    input_shape,
    head_size=head_size,
    num_heads=num_heads,
    ff_dim=ff_dim,
    num_transformer_blocks=num_transformer_blocks,
    mlp_units=[1024, 512, 256, 128],
    mlp_dropout=0.1,
    dropout=0.1,
)

model.compile(
    loss='MSE',
    optimizer=keras.optimizers.Adam(learning_rate=1e-4)
)

# Data generator
def data_generator(X, y, subset_size, batch_size):
    while True:
        indices = np.random.choice(len(X), subset_size, replace=False)
        X_subset = X[indices]
        y_subset = y[indices]
        for i in range(0, subset_size, batch_size):
            yield X_subset[i:i+batch_size], y_subset[i:i+batch_size]

def val_data_generator(X, y, val_subset_size, batch_size):
    indices = np.random.choice(len(X), val_subset_size, replace=False)
    X_val_subset = X[indices]
    y_val_subset = y[indices]
    while True:
        for i in range(0, val_subset_size, batch_size):
            yield X_val_subset[i:i+batch_size], y_val_subset[i:i+batch_size]

subset_size = 4000000
val_subset_size = 500000
batch_size = 1000
steps_per_epoch = subset_size // batch_size
validation_steps = val_subset_size // batch_size

history = model.fit(
    data_generator(X_train, y_train, subset_size, batch_size),
    steps_per_epoch=steps_per_epoch,
    validation_data=val_data_generator(X_train, y_train, val_subset_size, batch_size),
    validation_steps=validation_steps,
    epochs=100,
    callbacks=[loss_yj_callback]
)

# Save and plot loss
loss_df = pd.DataFrame({
    'epoch': range(1, len(history.history['loss']) + 1),
    'loss': history.history['loss'],
    'val_loss': history.history['val_loss']
})
loss_df.to_csv("loss_data_epoch20.csv", index=False)

plt.figure(figsize=(8, 6))
plt.plot(loss_df['epoch'], loss_df['loss'], label='Training Loss')
plt.plot(loss_df['epoch'], loss_df['val_loss'], label='Validation Loss', linestyle='dashed')
plt.xlabel("Epoch", fontweight='bold')
plt.ylabel("Loss", fontweight='bold')
plt.title("Training & Validation Loss Curve", fontweight='bold')
plt.legend()
plt.tight_layout()
plt.show()

model.save('model_YJ.keras') 