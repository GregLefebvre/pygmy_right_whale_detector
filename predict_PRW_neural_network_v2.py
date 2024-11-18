import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import os
import csv
import pandas as pd

nfft = 2048
window_type_fft = "hamming"
overlap_fft = 0.9
min_frequency = 29  # Hz
max_frequency = 100  # Hz
spectrogram_window_width = 3  # seconds
spectrogram_window_height = 70  # Hz
spectrogram_window_overlap = 0.9  # 0.9 or 0.95
audio_files_extension = ".flac"

def load_filenames_from_previous_detections(csv_filepath):
    bdd_data = pd.read_csv(csv_filepath)
    distinct_files = bdd_data["File"].unique()
    return distinct_files

# Fonction pour lisser les prédictions en prenant la moyenne des 'smooth_number' valeurs avant et après
def smooth_predictions(predictions, smooth_number):
    smoothed_predictions = np.zeros_like(predictions)
    half_window = smooth_number // 2  # Diviser le nombre par 2 pour obtenir le lissage avant et après
    
    for i in range(predictions.shape[1]):
        for j in range(predictions.shape[0]):
            # Calculer les indices de début et de fin pour lisser avant et après
            start_idx = max(0, j - half_window)  # Empêche les indices négatifs
            end_idx = min(predictions.shape[0], j + half_window + 1)  # Empêche de dépasser la fin des indices
            
            # Prendre la moyenne des valeurs dans cette fenêtre
            smoothed_predictions[j, i] = np.mean(predictions[start_idx:end_idx, i])
    
    return smoothed_predictions

# Fonction pour enregistrer les détections dans un fichier CSV
def save_detections_to_csv(detections, csv_filename):
    with open(csv_filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        for detection in detections:
            writer.writerow(detection)

# Fonction pour vérifier le recouvrement entre deux rectangles
def check_overlap(portion1, portion2):
    start_time_1, end_time_1, start_freq_1, end_freq_1 = portion1
    start_time_2, end_time_2, start_freq_2, end_freq_2 = portion2
    
    # Chevauchement temporel
    overlap_time = max(0, min(end_time_1, end_time_2) - max(start_time_1, start_time_2))
    # Chevauchement fréquentiel
    overlap_freq = max(0, min(end_freq_1, end_freq_2) - max(start_freq_1, start_freq_2))
    
    return overlap_time > 0 and overlap_freq > 0

def normalize_matrix(matrix):
    # Normalize each matrix individually based on its min and max values
    min_val = np.min(matrix)
    max_val = np.max(matrix)
    if max_val > min_val:  # Prevent division by zero if min and max are equal
        return (matrix - min_val) / (max_val - min_val)
    else:
        return matrix  # Return the matrix as-is if all values are the same

# Function to validate detections based on context
def validate_detections(predictions, context_size, seuil_PRW, seuil_non_1, seuil_non_2):
    # Create a binary array for detections
    detection_flags = np.zeros(predictions.shape[0], dtype=bool)
    
    for i in range(context_size, predictions.shape[0] - context_size):
        # Check if detection meets the criteria for the current segment and context
        if (
            all(predictions[i-context_size:i+context_size+1, 1] >= seuil_PRW)
            and all(np.sum(predictions[i-context_size:i+context_size+1, 2:], axis=1) <= seuil_non_1)
            and all(np.all(predictions[i-context_size:i+context_size+1, class_idx] < seuil_non_2) for class_idx in range(2, 7))
        ):
            detection_flags[i] = True

    return detection_flags

# Fonction pour afficher un spectrogramme, le découper, faire des prédictions avec le réseau de neurones, et afficher les labels
def neural_net_predict_on_audio(model_path, sound_folder, csv_filename, context_size, seuil_PRW, seuil_non_1, seuil_non_2, files_considered):
    model = tf.keras.models.load_model(model_path)

    if files_considered is None: #on prend tout le dossier
        files_considered = os.listdir(sound_folder)
    
    for f in tqdm(files_considered, desc="fichiers"):
        if not f.endswith(audio_files_extension):
            continue
        
        audio_filepath = os.path.join(sound_folder, f)

        try:
            # Attempt to load using librosa
            audio, sr = librosa.load(audio_filepath, sr=None)
            print("Loaded using librosa.")
        except Exception as e:
            print(f"Librosa failed with error: {e}")
            print("Attempting to load with soundfile instead.")
            continue

        # Charger l'audio avec librosa
        audio_duration = librosa.get_duration(y=audio, sr=sr)
        
        # Calculer le spectrogramme
        hop_length = int(nfft * (1 - overlap_fft))
        S = librosa.stft(audio, n_fft=nfft, hop_length=hop_length, window=window_type_fft)
        S_magnitude = np.abs(S)

        # Assurez-vous que toutes les valeurs sont positives (ajoutez une petite constante si nécessaire)
        S_magnitude[S_magnitude == 0] = 1e-6

        # Convertir en dB
        S_db = 20*np.log10(S_magnitude)
        
        freqs = librosa.fft_frequencies(sr=sr, n_fft=nfft)
        min_idx = np.argmax(freqs >= min_frequency)
        max_idx = np.argmax(freqs > max_frequency)
        
        S_db = S_db[min_idx:max_idx, :]
        freqs = freqs[min_idx:max_idx]
        
        time_per_pixel = hop_length / sr
        freq_per_pixel = sr / nfft

        window_width_px = int(spectrogram_window_width / time_per_pixel)
        window_height_px = int(spectrogram_window_height / freq_per_pixel)

        step_time = spectrogram_window_width * (1 - spectrogram_window_overlap)
        step_px = step_time / time_per_pixel

        num_cols = int((audio_duration - spectrogram_window_width) // step_time)

        portions_dates = []
        portions_start_times = []
        portions = []
        all_coordinates_px = []

        for x in range(0, num_cols):
            coordinates_px = [int(x * step_px), int(x * step_px)+window_width_px,0, window_height_px]
            portion = S_db[0:window_height_px, int(x * step_px):int(x * step_px + window_width_px)]
            portions.append(normalize_matrix(portion))
            all_coordinates_px.append(coordinates_px)

            portion_date = f
            portion_start_time = x * step_time
            portions_dates.append(portion_date)
            portions_start_times.append(portion_start_time)

        portions_start_times = np.array(portions_start_times)
        portions_dates = np.array(portions_dates)
        all_coordinates_px = np.array(all_coordinates_px)
        portions = np.array(portions)
        portions_expanded = np.expand_dims(portions, -1)  # Ajouter une dimension pour correspondre à (hauteur, largeur, 1)

        # Prédictions du modèle
        predictions = model.predict(portions_expanded)
        
        # Filtrer les détections par contexte
        detection_flags = validate_detections(predictions, context_size, seuil_PRW, seuil_non_1, seuil_non_2)
    


        # Appliquer le filtre de chevauchement pour n'afficher que la détection la plus forte
        final_rectangles = []
        for i, portion in enumerate(all_coordinates_px):
            if detection_flags[i]:  # Prendre en compte seulement les détections validées
                start_px, end_px, _, _ = portion
                start_time = start_px*step_time/step_px

                prediction_value = predictions[i][1]
                
                overlap_found = False
                for j, (existing_portion, existing_prediction, _, _, _) in enumerate(final_rectangles):
                    if check_overlap(portion, existing_portion):
                        # Remplacer la détection si elle est plus forte
                        if prediction_value > existing_prediction:
                            final_rectangles[j] = (portion, prediction_value, 1, f, start_time)
                        overlap_found = True
                        break
                
                if not overlap_found:
                    final_rectangles.append((portion, prediction_value, 1, f, start_time))


        # Enregistrer les nouvelles détections dans le fichier CSV sans supprimer les anciennes
        if len(final_rectangles) > 0:
            save_detections_to_csv(final_rectangles, csv_filename)
