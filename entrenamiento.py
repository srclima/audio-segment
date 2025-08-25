import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.saving import save_model
from sklearn.preprocessing import StandardScaler

# 🔹 Cargar el dataset con MFCCs (Asegúrate de que este archivo existe)
dataset_path = "audio_features.csv"
df = pd.read_csv(dataset_path)

# 🔹 Separar características (MFCCs) y etiquetas (label)
X = df.iloc[:, 1:].values  # MFCCs (sin la última columna)

# 🔹 Normalizar los MFCCs
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # Normalizar las características

# 🔹 Definir un Autoencoder
input_dim = X_scaled.shape[1]

# 🔹 Definir la Red Neuronal en TensorFlow
model = keras.Sequential([
    keras.Input(shape=(input_dim,)),  # Entrada
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),  # Capa comprimida (representación del audio)
    layers.Dense(64, activation='relu'),
    layers.Dense(input_dim, activation='linear')  # Reconstrucción
])

# 🔹 Compilar el modelo
model.compile(optimizer='adam', loss='mse')

# 🔹 Entrenar la red neuronal
history = model.fit(X_scaled, X_scaled, epochs=100, batch_size=32, validation_split=0.2)

# 🔹 Guardar el modelo entrenado
save_model(model, "modelo_match.keras")
