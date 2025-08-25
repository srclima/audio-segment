from pydub import AudioSegment
import os

def convert_mp3_to_wav(input_folder, output_folder, segment_length=3):
    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.endswith(".mp3"):
                # Construir rutas de entrada y salida manteniendo la estructura de subcarpetas
                os.makedirs(output_folder, exist_ok=True)

                input_path = os.path.join(root, file)
                output_path = os.path.join(output_folder, file.replace(".mp3", ".wav"))

                # Convertir a WAV
                audio = AudioSegment.from_mp3(input_path)
                audio = audio.set_frame_rate(16000).set_channels(1)  # 16kHz, Mono
                # audio.export(output_path, format="wav")
                print(f"Convertido: {file} → {output_path}")

                # Dividir el archivo WAV en segmentos
                segment_samples = segment_length * 1000  # en milisegundos
                num_segments = len(audio) // segment_samples  

                for i in range(num_segments):
                    segment = audio[i * segment_samples:(i + 1) * segment_samples]
                    segment_name = file.replace(".mp3", f"_segment_{i + 1}.wav")
                    segment_path = os.path.join(output_folder, segment_name)
                    segment.export(segment_path, format="wav")
                    print(f"Segmento {i + 1} guardado como: {segment_name}")

convert_mp3_to_wav("data/", "processed/")