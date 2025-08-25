import numpy as np
import librosa
from pydub import AudioSegment
from sklearn.metrics.pairwise import cosine_similarity
from typing import List

def find_audio_matches_spectrogram(long_audio_path: str, short_audio_path: str, threshold: float = 0.8) -> List[int]:
    # Cargar y pre-procesar los audios
    long_audio = AudioSegment.from_file(long_audio_path).set_frame_rate(22050).set_channels(1)
    short_audio = AudioSegment.from_file(short_audio_path).set_frame_rate(22050).set_channels(1)
    
    long_y, long_sr = np.array(long_audio.get_array_of_samples()).astype(np.float32), long_audio.frame_rate
    short_y, short_sr = np.array(short_audio.get_array_of_samples()).astype(np.float32), short_audio.frame_rate

    # Generar espectrogramas de Mel
    long_mel = librosa.feature.melspectrogram(y=long_y, sr=long_sr)
    short_mel = librosa.feature.melspectrogram(y=short_y, sr=short_sr)
    
    # Aplanar el espectrograma corto para la comparación
    short_mel_flat = short_mel.flatten().reshape(1, -1)
    
    long_mel_len = long_mel.shape[1]
    short_mel_len = short_mel.shape[1]

    if long_mel_len < short_mel_len:
        return []

    match_seconds = []
    
    step_size = int(short_mel_len * 0.5)
    
    for i in range(0, long_mel_len - short_mel_len + 1, step_size):
        segment_mel = long_mel[:, i:i + short_mel_len]
        segment_mel_flat = segment_mel.flatten().reshape(1, -1)
        
        # Calcular la similitud de coseno
        similarity = cosine_similarity(segment_mel_flat, short_mel_flat)[0][0]

        if similarity > threshold:
            start_time_ms = (i * 512 / long_sr) * 1000
            match_seconds.append(round(start_time_ms / 1000))

    return match_seconds

# --- Uso del script ---

if __name__ == "__main__":
    long_audio_file = "audioscompletos/vital6FEB.wav"
    short_audio_file = "data/espaciovital_mid.mp3"

    similarity_threshold = 0.5
    matches = find_audio_matches_spectrogram(long_audio_file, short_audio_file, similarity_threshold)

    unique_matches = sorted(list(set(matches)))
    
    print(f"Número de veces que el audio corto aparece: {len(unique_matches)}")
    print("Segundos en los que se encontraron coincidencias:")
    for second in unique_matches:
        minutes = second // 60
        remaining_seconds = second % 60
        print(f" - Minuto: {minutes} | Segundos: {remaining_seconds}")