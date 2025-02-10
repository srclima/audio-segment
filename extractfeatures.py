import librosa
import numpy as np
import os
import pandas as pd

def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=16000)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return np.mean(mfccs, axis=1)

data = []
file_names = []

folder = "processed/"
for file in os.listdir(folder):
    if file.endswith(".wav"):
        file_path = os.path.join(folder, file)
        features = extract_features(file_path)
        data.append(features)
        file_names.append(file)

# Crear DataFrame sin etiqueta de clasificación
df = pd.DataFrame(data)
df.insert(0, "file_name", file_names)  # Agregar nombre del archivo como primera columna
df.to_csv("audio_features.csv", index=False)

print("Extracción de características completada. Archivo guardado como 'audio_features.csv'.")