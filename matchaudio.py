from pydub import AudioSegment
import numpy as np
from scipy.signal import correlate
import os

def convert_to_wav(input_path):
    """Convierte un archivo de audio a formato WAV y lo guarda en la misma carpeta."""
    output_path = os.path.splitext(input_path)[0] + ".wav"
    if not os.path.exists(output_path):  # Evitar conversiones innecesarias
        audio = AudioSegment.from_file(input_path)
        audio.export(output_path, format="wav")
    return output_path

def audio_to_array(audio_path):
    """Carga un archivo de audio en WAV y lo convierte en un array de numpy"""
    audio = AudioSegment.from_file(audio_path)
    samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
    return samples, audio.frame_rate

def find_filtered_occurrences(long_audio_path, short_audio_path, min_gap=10):
    """Encuentra ocurrencias del fragmento corto en el audio largo con un mínimo de separación entre coincidencias."""
    # Convertir a WAV
    long_audio_wav = convert_to_wav(long_audio_path)
    short_audio_wav = convert_to_wav(short_audio_path)

    # Cargar audios en formato NumPy
    long_audio, long_rate = audio_to_array(long_audio_wav)
    short_audio, short_rate = audio_to_array(short_audio_wav)

    if long_rate != short_rate:
        raise ValueError("Las tasas de muestreo de los audios no coinciden.")

    # Correlación cruzada para detectar similitudes
    correlation = correlate(long_audio, short_audio, mode='valid')

    # Ajustar el umbral dinámicamente basado en el valor máximo de correlación
    threshold = np.max(correlation) * 0.3  # 70% del máximo valor de correlación

    # Encontrar picos que superen el umbral
    raw_occurrences = np.where(correlation > threshold)[0] / long_rate  # Convertir a segundos

    # Filtrar coincidencias que estén demasiado cerca
    filtered_occurrences = []
    last_time = -min_gap  # Para que el primero siempre se guarde

    for time in raw_occurrences:
        if time - last_time >= min_gap:
            filtered_occurrences.append(time)
            last_time = time

    return filtered_occurrences

# Rutas de los archivos de audio
long_audio_path = "audioscompletos/vital6FEB.mp3"
short_audio_path = "data/espaciovital.mp3"

# Buscar ocurrencias con mínimo 1 minuto (60s) de separación
positions = find_filtered_occurrences(long_audio_path, short_audio_path, min_gap=60)

print("El fragmento apareció en los segundos:", positions)
