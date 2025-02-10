import tensorflow as tf
from pydub import AudioSegment
import numpy as np
import os
import librosa
import pandas as pd
import soundfile as sf
from sklearn.metrics.pairwise import cosine_similarity

# Cargar modelo entrenado
MODEL_PATH = "modelo_match.keras"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"No se encontró el modelo en {MODEL_PATH}")
modelo = tf.keras.models.load_model(MODEL_PATH)

# Cargar features de audios previos
features_df = pd.read_csv("audio_features.csv")

# Función para extraer MFCCs
def extract_mfcc(wav_file, n_mfcc=13):
    try:
        audio, sr = librosa.load(wav_file, sr=16000)
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
        return np.mean(mfccs, axis=1)  # Promedio sobre el tiempo
    except Exception as e:
        print(f"Error procesando {wav_file}: {e}")
        return None

# Función para comparar fragmento con base de datos usando Similitud del Coseno
def find_similar_audio(fragment_features, threshold=0.85):
    similarities = []
    for i, row in features_df.iterrows():
        stored_features = row.iloc[1:].values  # Omitimos la columna del nombre del archivo
        sim = cosine_similarity([fragment_features], [stored_features])[0][0]
        similarities.append((row["file_name"], sim))
    return [file for file, sim in similarities if sim >= threshold]

# Función para identificar coincidencias en un audio completo
def detect_matches_in_audio(input_mp3):
    if not os.path.exists(input_mp3):
        raise FileNotFoundError(f"El archivo {input_mp3} no existe.")

    input_wav = input_mp3.replace(".mp3", ".wav")
    AudioSegment.from_mp3(input_mp3).export(input_wav, format="wav")
    
    audio = AudioSegment.from_wav(input_wav)
    chunks = [audio[i:i + 1000] for i in range(0, len(audio), 1000)]

    matched_intervals = []
    
    for i, chunk in enumerate(chunks):
        start_time = i  # En segundos
        chunk_wav = "chunk_temp.wav"
        chunk.export(chunk_wav, format="wav")

        features = extract_mfcc(chunk_wav)
        if features is None:
            os.remove(chunk_wav)
            continue

        samples = np.expand_dims(features, axis=0)
        prediction = modelo.predict(samples)[0][0]

        matched_files = find_similar_audio(features)

        if prediction >= 0.5 or matched_files:
            matched_intervals.append((start_time, start_time + 1, matched_files))

        os.remove(chunk_wav)

    intervals_df = pd.DataFrame(matched_intervals, columns=["start_time", "end_time", "matched_files"])
    intervals_csv = input_wav.replace(".wav", "_matches.csv")
    intervals_df.to_csv(intervals_csv, index=False)

    print(f"Intervalos de coincidencia guardados en: {intervals_csv}")
    return intervals_csv

# Ejecutar el proceso con un archivo específico
if __name__ == "__main__":
    input_audio = "audioscompletos/vital6FEB.mp3"  # Asegúrate de que el archivo exista
    intervals_csv = detect_matches_in_audio(input_audio)
