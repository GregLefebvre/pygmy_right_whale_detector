import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
from tqdm import tqdm
import cv2
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import seaborn as sns
import librosa.display
from tensorflow.keras.callbacks import ModelCheckpoint


nfft = 2048
window_type_fft = "hamming"
overlap_fft = 0.9

min_frequency = 29#Hz (30-1)
max_frequency = 100#Hz

spectrogram_window_width = 3#seconds
spectrogram_window_height = 70#Hz
spectrogram_window_overlap = 0.9#0.9 ou 0.95

sr=0

def normalize_matrix(matrix):
    # Normalize each matrix individually based on its min and max values
    min_val = np.min(matrix)
    max_val = np.max(matrix)
    if max_val > min_val:  # Prevent division by zero if min and max are equal
        return (matrix - min_val) / (max_val - min_val)
    else:
        return matrix  # Return the matrix as-is if all values are the same

def save_spectrograms_as_matrices(spectrogram_window_width, spectrogram_window_height, spectrogram_window_overlap, audio_filepath):
    """
    Sauvegarde les spectrogrammes découpés en portions sous forme de fichiers NumPy.
    Retourne les coordonnées et les chemins des fichiers sauvegardés.
    """
    audio, sr = librosa.load(audio_filepath, sr=None)
    audio_duration = librosa.get_duration(y=audio, sr=sr)
    
    hop_length = int(nfft * (1 - overlap_fft))
    S = librosa.stft(audio, n_fft=nfft, hop_length=hop_length, window=window_type_fft)
    S_magnitude = np.abs(S)

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

    portions_start_times = []
    portions_end_times = []
    portions = []
    all_coordinates_px = []

    for x in range(0, num_cols):
        coordinates_px = [int(x * step_px), int(x * step_px)+window_width_px,0, window_height_px]
        portion = S_db[0:window_height_px, int(x * step_px):int(x * step_px + window_width_px)]
        portions.append(portion)
        all_coordinates_px.append(coordinates_px)

        portion_start_time = x * step_time
        portions_start_times.append(portion_start_time)
        portion_end_time = portion_start_time + spectrogram_window_width
        portions_end_times.append(portion_end_time)

    portions_start_times = np.array(portions_start_times)
    all_coordinates_px = np.array(all_coordinates_px)
    portions = np.array(portions)
    portions_expanded = np.expand_dims(portions, -1)  # Ajouter une dimension pour correspondre à (hauteur, largeur, 1)
    return portions_expanded, np.array(portions_start_times), np.array(portions_end_times)

# Charger les fichiers CSV
def load_data(bdd_file):
    """Charger les fichiers CSV BDD.csv"""
    bdd_data = pd.read_csv(bdd_file)
    return bdd_data

# Fonction pour charger le fichier son correspondant à une date
def load_audio_for_date(audio_parent_dir, date_str, mode=0):
    if mode==0:
        # Construire le préfixe de la date en supprimant les deux points
        date_prefix = date_str[:8] + 'T' + date_str[9:11] + date_str[12:14]  # Format 'YYYYMMDDTHHmm'
    elif mode==1:
        date_prefix = date_str
    # Rechercher le sous-dossier correspondant dans le répertoire des annotations
    matching_files = [f for f in os.listdir(audio_parent_dir) if f.startswith(date_prefix)]
    if matching_files:
        return matching_files[0]
    
    return None

def time_overlap_percentage(A1, B1, A2, B2):
    # Longueur du segment 2
    L2 = B2 - A2
    # Calculer la longueur du chevauchement
    overlap_length = max(0, min(B1, B2) - max(A1, A2))
    # Si L2 est 0, éviter la division par zéro (pas de segment valide)
    if L2 == 0:
        return 0.0
    # Calcul du pourcentage de chevauchement par rapport à la longueur du segment 2
    overlap_percentage = (overlap_length / L2) * 100
    return overlap_percentage

def label_spectrogram_portions(audio_parent_dir, positive_db_path, spectrogram_window_width, spectrogram_window_height, spectrogram_window_overlap, positive_label_value, return_sea_noise_portions):
    # idée de l'algorithme:
    # parcourir tous les lignes de la bdd, tant que c'est le meme fichier son ne calculer le spectro qu'1 fois
    # Parcourir tous les fichiers sons contenant des signaux d'interets (SI)
    # Les découper en portions égales et les étiqueter
    all_labels = []
    all_spectrograms_portions_expanded = []
    all_spectrograms_portions_end_times = []
    all_spectrograms_portions_start_times = []
    
    bdd_data = load_data(positive_db_path) #toute la base de données (c'est long)
    last_bdd_date = None
    bdd_portions_coordinate_grouped_by_date = []
    all_spectrograms_portions_expanded_grouped_by_date = []
    all_spectrograms_portions_end_times_grouped_by_date = []
    all_spectrograms_portions_start_times_grouped_by_date = []

    #boucle de calcule et découpage des spectros de fichiers contenant des PRW
    for i in tqdm(range(len(bdd_data)), desc="Découpage des fichiers sons"):
        if 'Date' in bdd_data.iloc[i]:
            bdd_date = bdd_data.iloc[i]['Date']
        else:
            bdd_date = bdd_data.iloc[i]['Filename'][:8]
        zone_bdd_coordinate = [
            bdd_data.iloc[i]['Start time'], 
            bdd_data.iloc[i]['End time']
        ]
        #si changement de fichier son (ou premier fichier) on calcule le spectro
        if (last_bdd_date is None) or (bdd_date != last_bdd_date):
            audio_rel_filepath = load_audio_for_date(audio_parent_dir, bdd_date, 1)
            audio_filepath = audio_parent_dir+"/"+audio_rel_filepath
            # on découpe le spectrogramme en portions de meme tailles
            
            portions_expanded, portions_start_times, portions_end_times = save_spectrograms_as_matrices(spectrogram_window_width, spectrogram_window_height, spectrogram_window_overlap, audio_filepath)
            all_spectrograms_portions_expanded_grouped_by_date.append(portions_expanded)
            all_spectrograms_portions_start_times_grouped_by_date.append(portions_start_times)
            all_spectrograms_portions_end_times_grouped_by_date.append(portions_end_times)
            last_bdd_date = bdd_date
            bdd_portions_coordinate_grouped_by_date.append([zone_bdd_coordinate])
        else:
            bdd_portions_coordinate_grouped_by_date[-1].append(zone_bdd_coordinate)

    #deuxieme boucle
    #on parcourt chaque portions de spectrogrammes
    #Assignation des labels pour l'entrainement
    for i in range(len(bdd_portions_coordinate_grouped_by_date)):
        zones_bdd_coordinate_current = bdd_portions_coordinate_grouped_by_date[i]

        labels = [0]*len(all_spectrograms_portions_start_times_grouped_by_date[i])
        start_times = [None]*len(all_spectrograms_portions_start_times_grouped_by_date[i])
        end_times = [None]*len(all_spectrograms_portions_start_times_grouped_by_date[i])
        for j in range(len(all_spectrograms_portions_start_times_grouped_by_date[i])):
            spectrogram_portion_start_time = all_spectrograms_portions_start_times_grouped_by_date[i][j]
            spectrogram_portion_end_time = all_spectrograms_portions_end_times_grouped_by_date[i][j]
            all_spectrograms_portions_expanded.append(all_spectrograms_portions_expanded_grouped_by_date[i][j])

            for k in range(len(zones_bdd_coordinate_current)):
                zone_bdd_coordinate = zones_bdd_coordinate_current[k]
                # Comparer les coordonnées des zones (start_time, end_time, min_freq, max_freq)
                p = time_overlap_percentage(spectrogram_portion_start_time, spectrogram_portion_end_time, zone_bdd_coordinate[0], zone_bdd_coordinate[1])
                if p > 0: #présence d'au moins une portion de PRW dans la portion:
                    rel_start_time = zone_bdd_coordinate[0] - spectrogram_portion_start_time
                    rel_end_time = zone_bdd_coordinate[1] - spectrogram_portion_start_time
                    start_times[j] = rel_start_time
                    end_times[j]= rel_end_time
                    if p == 100: #SI dans la portion
                        labels[j] = positive_label_value
                    else: #un peu de SI mais pas tout donc on n'utilise pas pour l'entrainement
                        labels[j] = -1

        all_labels += labels
        all_spectrograms_portions_start_times += start_times
        all_spectrograms_portions_end_times += end_times
    
    labels_np = np.array(all_labels)
    all_start_times_np = np.array(all_spectrograms_portions_start_times)
    all_end_times_np = np.array(all_spectrograms_portions_end_times)
    all_spectrograms_portions_expanded = np.array(all_spectrograms_portions_expanded)

    if not return_sea_noise_portions:
        #on enleve les labels=0 qui correspondent au bruit
        ind = np.where(labels_np!=0)[0]
        labels_np = labels_np[ind]
        all_start_times_np = all_start_times_np[ind]
        all_end_times_np = all_end_times_np[ind]
        all_spectrograms_portions_expanded = all_spectrograms_portions_expanded[ind]

    return labels_np, all_spectrograms_portions_expanded, all_start_times_np, all_end_times_np

def display_one_matrice(matrice, title, start_time, min_frequency,max_frequency,spectrogram_window_width):
    # Échelle de temps et de fréquence directement liée aux variables globales
    time_axis = np.linspace(0, spectrogram_window_width, matrice.shape[1])
    freq_axis = np.linspace(max_frequency, min_frequency, matrice.shape[0])

    # Si la matrice a une dernière dimension inutile, la supprimer
    if matrice.shape[-1] == 1:
        matrice = matrice.squeeze(-1)

    # Affichage de la matrice avec inversion de l'échelle verticale (fréquence)
    plt.figure(figsize=(4, 3))
    plt.imshow(matrice, cmap='viridis', aspect='auto', extent=[time_axis[0], time_axis[-1], freq_axis[0], freq_axis[-1]])
    ax = plt.gca()
    ax.invert_yaxis()
    
    # Dessiner le rectangle BDD
    duration = 1.5
    rect = patches.Rectangle((start_time, min_frequency), duration, max_frequency-min_frequency, linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    
    plt.title(title)
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.colorbar(label='Amplitude (dB)')
    plt.show()

def load_numpy_arrays_from_filepaths(filepaths):
    """Charger les matrices NumPy à partir des fichiers .npy."""
    matrices = []
    for filepath in tqdm(filepaths, desc="Chargement des matrices NumPy :"):
        try:
            matrix = np.load(filepath)
            matrices.append(matrix)
        except Exception as e:
            print(f"Erreur lors du chargement de {filepath}: {e}")

    matrices = np.array(matrices)
    print(f"Forme des matrices chargées : {matrices.shape}")
    return np.expand_dims(matrices, -1)


# Define parameters for each dataset configuration
datasets = [
]

# Initialize variables to store concatenated results
all_labels, all_portions, all_start_times, all_end_times = [], [], [], []
num_classes = 7

# Process each dataset
for dataset in datasets:
    labels, portions, start_times, end_times = label_spectrogram_portions(dataset["sound_folder"], dataset["bdd_name"], spectrogram_window_width, spectrogram_window_height, spectrogram_window_overlap, dataset["positive_label_value"], dataset["return_sea_noise_portions"])
    all_labels.append(labels)
    all_portions.append(portions)
    all_start_times.append(start_times)
    all_end_times.append(end_times)

# Concatenate all arrays
labels = np.concatenate(all_labels)
all_spectrograms_portions = np.concatenate(all_portions)

# Normalize each matrix individually
matrices_normalized = np.array([normalize_matrix(matrix) for matrix in all_spectrograms_portions])

all_start_times = np.concatenate(all_start_times)
all_end_times = np.concatenate(all_end_times)

# Print results
print("---")
print("Total number of labels:", labels.shape[0])
for i in range(0, num_classes):
    print(f"Number of labels for class {i}:", (labels == i).sum())
print("Number of labels for class -1:", (labels == -1).sum())
print("---")

def build_neural_network(input_shape, num_classes, l2_lambda):
    input_img = layers.Input(shape=input_shape)

    # Couche de convolution 1 avec régularisation L2
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same',
                      kernel_regularizer=regularizers.l2(l2_lambda))(input_img)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)

    # Couche de convolution 2 avec régularisation L2
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same',
                      kernel_regularizer=regularizers.l2(l2_lambda))(x)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)

    # Flatten et couches denses avec régularisation L2
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(l2_lambda))(x)
    x = layers.Dropout(0.5)(x)  # Dropout pour éviter l'overfitting

    # Couche de sortie avec Softmax pour la classification multi-classes
    last_row = layers.Dense(num_classes, activation='softmax', kernel_regularizer=regularizers.l2(l2_lambda))(x)

    # Modèle complet
    model = models.Model(input_img, last_row)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    print("Learning rate:", model.optimizer.learning_rate.numpy())

    return model

# Fonction de décroissance exponentielle du taux d'apprentissage
def exponential_decay(epoch, lr):
    decay_rate = 0.1
    return float(lr * tf.math.exp(-decay_rate * epoch))

# Fonction modifiée avec ajustement du taux d'apprentissage et enregistrement du modèle
def train_neural_network_expo_decay(neural_network, x_train, y_train, x_val, y_val, model_save_dir, batch_size=32, epochs=5):
    """Entraîne le réseau de neurones avec un taux d'apprentissage variable (exponential decay) et sauvegarde le modèle à chaque époque."""
    
    # Callback pour ajuster dynamiquement le taux d'apprentissage
    lr_scheduler = tf.keras.callbacks.LearningRateScheduler(exponential_decay, verbose=1)
    
    # Callback pour enregistrer le modèle à chaque époque
    model_checkpoint = ModelCheckpoint(
        filepath=os.path.join(model_save_dir, "model_epoch_{epoch:02d}.keras"),  # Sauvegarder avec le numéro d'époque
        save_weights_only=False,  # Sauvegarde complète du modèle
        save_best_only=False,  # Sauvegarder à chaque époque
        verbose=1
    )
    
    # Entraîner le modèle avec la pondération des classes et la modification du taux d'apprentissage
    history = neural_network.fit(x_train, y_train, 
                                 validation_data=(x_val, y_val),
                                 batch_size=batch_size, 
                                 epochs=epochs, 
                                 shuffle=True, 
                                 verbose=1, 
                                 callbacks=[lr_scheduler, model_checkpoint])
    
    print("Entraînement terminé.")
    return history

def save_model(model, model_path='model_neural_network.keras'):
    """Sauvegarde un modèle entraîné."""
    model.save(model_path)
    print(f"Modèle sauvegardé sous {model_path}")

def plot_loss(history, filename):
    """Affiche la courbe de la perte et de la précision au fil des époques."""
    plt.figure(figsize=(12, 4))

    # Courbe de perte
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.legend()
    # Courbe de précision
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')
    plt.legend()
    # Enregistrer la figure dans un fichier
    plt.savefig(filename)
    plt.show()
    print(f"Les courbes de performance ont été enregistrées dans le fichier '{filename}'.")

def save_classified_labels_to_csv(predictions, filepaths, output_csv='classified_labels.csv'):
    """
    Sauvegarde les labels des matrices classifiées dans un fichier CSV.
    
    :param predictions: Prédictions des labels (0 ou 1)
    :param filepaths: Chemins des fichiers d'origine
    :param output_csv: Chemin du fichier CSV où sauvegarder les labels
    """
    df = pd.DataFrame({
        'Filepath': filepaths,
        'Label': predictions.flatten()
    })
    df.to_csv(output_csv, index=False)
    print(f"Labels classifiés sauvegardés dans {output_csv}")

def display_matrices_by_class(matrices, predictions, class_label, num_images=10):
    """
    Affiche un nombre spécifié de matrices pour une classe donnée (0 ou 1) avec une échelle de temps et de fréquence.
    """
    class_indices = np.where(predictions == class_label)[0]
    selected_indices = class_indices[:num_images]

    time_axis = np.linspace(0, spectrogram_window_width, matrices.shape[2])
    freq_axis = np.linspace(max_frequency, min_frequency, matrices.shape[1])

    for idx in selected_indices:
        matrix = matrices[idx]
        # Si la matrice a une dernière dimension inutile, la supprimer
        if matrix.shape[-1] == 1:
            matrix = matrix.squeeze(-1)
        
        plt.figure(figsize=(5, 4))
        plt.imshow(matrix, cmap='viridis', aspect='auto', extent=[time_axis[0], time_axis[-1], freq_axis[0], freq_axis[-1]])
        plt.gca().invert_yaxis()
        plt.title(f'Class: {class_label}')
        plt.xlabel('Time (s)')
        plt.ylabel('Frequency (Hz)')
        plt.colorbar(label='Amplitude (dB)')
        plt.show()


def augment_matrix_with_uniform_shifts(matrix, start_idx, end_idx, factor):
    """
    Generates uniformly spaced shifts for a matrix along the time axis, preserving the portion of interest.
    """
    num_cols = matrix.shape[1]
    portion_length = end_idx - start_idx
    max_shift_right = num_cols - end_idx
    max_shift_left = start_idx

    # Calculate uniform shifts based on the factor
    shifts = np.linspace(-max_shift_left, max_shift_right, factor).astype(int)
    
    augmented_matrices = []
    new_start_times = []
    new_end_times = []
    
    for shift in shifts:
        # Apply the shift to the matrix
        rolled_matrix = np.roll(matrix, shift=shift, axis=1)
        new_start = start_idx + shift
        new_end = new_start + portion_length
        if new_start < 0:
            new_start = 0
        if new_end > num_cols:
            new_end = num_cols

        portion_to_copy = matrix[:, start_idx:end_idx]
        rolled_matrix[:, new_start:new_end] = portion_to_copy[:, :new_end - new_start]

        augmented_matrices.append(rolled_matrix)
        new_start_times.append(new_start / num_cols * spectrogram_window_width)
        new_end_times.append(new_end / num_cols * spectrogram_window_width)

    return augmented_matrices, new_start_times, new_end_times

def smart_duplicate_label_x_matrices(matrices, labels, label_to_duplicate, start_times, end_times, factor):
    """
    Duplicate matrices with a specified label by a given factor, applying uniform time shifts to the portion of interest.
    """
    indices_label_1 = np.where(labels == label_to_duplicate)[0]
    
    if len(indices_label_1) > 0:
        matrices_label_1 = matrices[indices_label_1]
        labels_label_1 = labels[indices_label_1]
        start_times_label_1 = start_times[indices_label_1]
        end_times_label_1 = end_times[indices_label_1]

        augmented_matrices = []
        augmented_labels = []
        augmented_start_times = []
        augmented_end_times = []

        for i in range(len(matrices_label_1)):
            matrix = matrices_label_1[i]
            start_time = start_times_label_1[i]
            end_time = end_times_label_1[i]

            # Convert start and end times to matrix indices
            num_cols = matrix.shape[1]
            start_idx = int((start_time / spectrogram_window_width) * num_cols)
            end_idx = int((end_time / spectrogram_window_width) * num_cols)

            # Generate augmented matrices with uniform shifts
            augmented_matrix_set, start_times_set, end_times_set = augment_matrix_with_uniform_shifts(
                matrix, start_idx, end_idx, factor
            )

            augmented_matrices.extend(augmented_matrix_set)
            augmented_labels.extend([labels_label_1[i]] * factor)
            augmented_start_times.extend(start_times_set)
            augmented_end_times.extend(end_times_set)

        augmented_matrices = np.array(augmented_matrices)
        augmented_labels = np.array(augmented_labels)
        augmented_start_times = np.array(augmented_start_times)
        augmented_end_times = np.array(augmented_end_times)

        matrices_combined = np.concatenate([matrices, augmented_matrices], axis=0)
        labels_combined = np.concatenate([labels, augmented_labels], axis=0)
        start_times_combined = np.concatenate([start_times, augmented_start_times], axis=0)
        end_times_combined = np.concatenate([end_times, augmented_end_times], axis=0)

    return matrices_combined, labels_combined, start_times_combined, end_times_combined




def distort_matrix(matrix, base_multiply_range=(0.85, 1.15)):
    """
    Distort the input matrix by multiplying each pixel by a random factor based on its intensity.
    
    Parameters:
    - matrix: 2D numpy array (values from 0 to 1) to distort.
    - intensity_range: Tuple specifying the range of pixel intensities to target for distortion.
    - base_multiply_range: Tuple specifying the base range for the random multiplication factor.
    
    Returns:
    - distorted_matrix: 2D numpy array with distortions applied based on pixel intensity.
    """
    distorted_matrix = matrix.copy()
    
    # Define scaling factors inversely proportional to pixel intensity
    scale_factors = 1 - distorted_matrix
    
    multiply_min = base_multiply_range[0] + (1 - base_multiply_range[0]) * scale_factors
    multiply_max = base_multiply_range[1] - (base_multiply_range[1] - 1) * scale_factors
    
    random_factors = np.random.uniform(multiply_min, multiply_max)
    
    distorted_matrix *= random_factors
    
    distorted_matrix = np.clip(distorted_matrix, 0, 1)
    return distorted_matrix

def augment_data_with_distortion(matrices, labels, start_times, end_times, augmentation_factors):
    """
    Augment the data by applying distortions to each matrix, creating new "child" matrices.

    Parameters:
    - matrices: 3D numpy array of shape (num_matrices, height, width) containing matrices to augment.
    - labels: Array of labels corresponding to each matrix.
    - start_times: Array of start times corresponding to each matrix.
    - end_times: Array of end times corresponding to each matrix.
    - augmentation_factors: Dictionary specifying the augmentation factor for each class label.

    Returns:
    - augmented_matrices: Array containing the original and augmented matrices.
    - augmented_labels: Array containing the labels for the augmented data.
    - augmented_start_times: Array containing the start times for the augmented data.
    - augmented_end_times: Array containing the end times for the augmented data.
    """
    augmented_matrices = []
    augmented_labels = []
    augmented_start_times = []
    augmented_end_times = []

    for matrix, label, start_time, end_time in zip(matrices, labels, start_times, end_times):
        # Append the original matrix and its data
        augmented_matrices.append(matrix)
        augmented_labels.append(label)
        augmented_start_times.append(start_time)
        augmented_end_times.append(end_time)

        # Generate augmented versions of the matrix
        factor = augmentation_factors[label]
        for _ in range(factor):
            distorted_matrix = distort_matrix(matrix)
            augmented_matrices.append(distorted_matrix)
            augmented_labels.append(label)
            augmented_start_times.append(start_time)
            augmented_end_times.append(end_time)

    # Convert lists to numpy arrays
    augmented_matrices = np.array(augmented_matrices)
    augmented_labels = np.array(augmented_labels)
    augmented_start_times = np.array(augmented_start_times)
    augmented_end_times = np.array(augmented_end_times)

    return augmented_matrices, augmented_labels, augmented_start_times, augmented_end_times

def test_augmentation(matrices, labels, start_times, end_times):
    # Generate a simple example matrix
    augmentation_factors = [0, 10]
    num_matrices = matrices.shape[0]

    # Run the augmentation
    augmented_matrices, augmented_labels, augmented_start_times, augmented_end_times = augment_data_with_distortion(
        matrices, labels, start_times, end_times, augmentation_factors
    )

    # Display the original matrices and their "children"
    for i in range(num_matrices):
        fig, axs = plt.subplots(1, augmentation_factors[labels[i]] + 1, figsize=(12, 4))
        axs[0].imshow(matrices[i], cmap='viridis', aspect='auto')
        axs[0].set_title("Original")
        axs[0].invert_yaxis()
        axs[0].axis('off')

        for j in range(augmentation_factors[1]):
            axs[j + 1].imshow(augmented_matrices[num_matrices + i * augmentation_factors[1] + j], cmap='viridis', aspect='auto')
            axs[j + 1].set_title(f"Child {j + 1}")
            axs[j+1].invert_yaxis()
            axs[j+1].axis('off')

        plt.show()




def main(labels, matrices_normalized, models_dir, epochs, num_classes, l2_lambda):
    
    matrix_size = matrices_normalized[0].shape

    matrices_augmented = matrices_normalized
    labels_augmented = labels

    # Séparer les données en ensemble d'entraînement et de validation (80% train, 20% test)

    x_train, x_val, y_train_int, y_val_int = train_test_split(
        matrices_normalized, labels_augmented, test_size=0.2, random_state=42
    )

    print("Shapes:",x_train.shape, y_train_int.shape, x_val.shape, y_val_int.shape)

    # Construire le modèle de réseau de neurones
    model = build_neural_network(matrix_size, num_classes, l2_lambda)

    # Convertir en one-hot
    print("nclasses", num_classes)
    print("y_train_int", y_train_int.shape)
    y_train = to_categorical(y_train_int, num_classes=num_classes)
    y_val = to_categorical(y_val_int, num_classes=num_classes)

    # Entraîner le modèle (et sauvegarder pour chaque époque)
    history = train_neural_network_expo_decay(model, x_train, y_train, x_val, y_val,models_dir,epochs=epochs)

    # Évaluation sur les données de validation
    val_predictions = model.predict(x_val)

    # Pour une classification multi-classes, on utilise argmax pour récupérer la classe prédite
    val_pred_classes = np.argmax(val_predictions, axis=1)
    val_true_classes = np.argmax(y_val, axis=1)  # Les vraies classes à partir des one-hot encodés

    print("Rapport de classification :")
    print(classification_report(val_true_classes, val_pred_classes))
    print(f"Précision : {accuracy_score(val_true_classes, val_pred_classes)}")

    return x_val, y_val, val_pred_classes, val_predictions, history


def plot_confusion_matrix(y_true, y_pred, labels, filename):
    """
    Affiche une matrice de confusion avec les valeurs en pourcentages et les nombres absolus entre parenthèses.
    
    :param y_true: Labels réels.
    :param y_pred: Prédictions faites par le modèle.
    :param labels: Liste des classes à afficher.
    """
    # Calcul de la matrice de confusion
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    # Convertir les valeurs en pourcentages
    cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    # Créer un texte annoté qui combine pourcentages et nombres absolus
    annotations = np.empty_like(cm).astype(str)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annotations[i, j] = f"{cm_percentage[i, j]:.2f}%\n({cm[i, j]})"
    
    # Affichage de la matrice de confusion avec seaborn
    plt.figure(figsize=(10, 9))
    sns.heatmap(cm_percentage, annot=annotations, fmt='', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predictions')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix (Percentage and Number)')
    # Enregistrer la figure dans un fichier
    plt.savefig(filename)
    # Afficher la figure à l'écran
    plt.show()
    # Imprimer un message de confirmation
    print(f"matrice de confusion enregistrée dans le fichier '{filename}'.")

def get_false_positives(y_true, y_pred, class_label):
    """
    Récupère les index des faux positifs pour une classe donnée.

    :param y_true: Labels réels.
    :param y_pred: Prédictions faites par le modèle.
    :param class_label: La classe pour laquelle on veut obtenir les faux positifs.
    :return: Index des faux positifs.
    """
    indexes = np.where((y_pred == class_label) & (y_true != class_label))[0]
    return indexes

def get_false_negatives(y_true, y_pred, class_label):
    """
    Récupère les index des faux négatifs pour une classe donnée.

    :param y_true: Labels réels.
    :param y_pred: Prédictions faites par le modèle.
    :param class_label: La classe pour laquelle on veut obtenir les faux négatifs.
    :return: Index des faux négatifs.
    """
    indexes = np.where((y_pred != class_label) & (y_true == class_label))[0]
    return indexes

from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

def plot_roc_curves(y_true, y_pred_proba, labels, filename):
    """
    Plots ROC curves for multi-class classification with AUC scores, with a zoomed subplot.

    :param y_true: Array of true labels.
    :param y_pred_proba: 2D array of predicted probabilities for each class.
    :param labels: List of class labels.
    :param filename: File path to save the ROC curves plot.
    """
    num_classes = len(labels)
    
    # Binarize the true labels for multi-class ROC analysis
    y_true_binarized = label_binarize(y_true, classes=np.arange(num_classes))
    
    # Initialize the plot with two subplots
    fig, (ax_full, ax_zoom) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot ROC curve for each class in both subplots
    for i, label in enumerate(labels):
        # Compute ROC curve and AUC for the current class
        fpr, tpr, _ = roc_curve(y_true_binarized[:, i], y_pred_proba[:, i])
        roc_auc = auc(fpr, tpr)
        
        # Plot on the main ROC subplot
        ax_full.plot(fpr, tpr, label=f"Class {label} (AUC = {roc_auc:.2f})")
        # Plot on the zoomed ROC subplot
        ax_zoom.plot(fpr, tpr)
    
    # Customize the full ROC plot
    ax_full.plot([0, 1], [0, 1], 'k--', label="Random Guess (AUC = 0.5)")
    ax_full.set_xlabel('False Positive Rate')
    ax_full.set_ylabel('True Positive Rate')
    ax_full.set_title('ROC Curves for Multi-Class Classification (Full)')
    
    # Customize the zoomed ROC plot
    ax_zoom.plot([0, 1], [0, 1], 'k--')  # Random guess line
    ax_zoom.set_xlim(0, 0.1)
    ax_zoom.set_ylim(0.9, 1.0)
    ax_zoom.set_xlabel('False Positive Rate')
    ax_zoom.set_ylabel('True Positive Rate')
    ax_zoom.set_title('Zoomed ROC Curves (FPR in [0, 0.1], TPR in [0.9, 1])')
    
    # Add one legend for both subplots
    handles, labels = ax_full.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower right")
    
    # Adjust layout and save the figure
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()
    print(f"ROC curves saved to '{filename}'.")


# %%
# Run the test function to visualize results
class_indices_x = np.where(labels == 1)[0][17:18]
a = matrices_normalized[class_indices_x]
b = np.ones(a.shape[0], dtype=int)
c = np.ones(a.shape[0])
d = np.ones(a.shape[0])
#test_augmentation(a,b,c,d)

# %%
#on enleve les labels=-1 pour l'entrainement
ind = np.where(labels!=-1)[0]
labels_without_not_full_prw = labels[ind]
all_portions_without_not_full_sound= matrices_normalized[ind]
selected_start_times_2 = all_start_times[ind]
selected_end_times_2 = all_end_times[ind]

# Identify indices where label is 0 and take only 1 out of 4
a = int((labels == 0).sum() / (labels == 1).sum())
a=1
label_0_indices = np.where(labels_without_not_full_prw == 0)[0]
reduced_label_0_indices = label_0_indices[::4]
print(label_0_indices.shape, reduced_label_0_indices.shape)
non_label_0_indices = np.where(labels_without_not_full_prw != 0)[0]
final_indices = np.sort(np.concatenate((reduced_label_0_indices, non_label_0_indices)))
labels_balanced = labels_without_not_full_prw[final_indices]
portions_balanced = all_portions_without_not_full_sound[final_indices]
start_times_balanced = selected_start_times_2[final_indices]
end_times_balanced = selected_end_times_2[final_indices]

max_d_factor = 5
augmentation_factors = [0]+[min(int((labels_balanced == 0).sum()/(labels_balanced == i).sum()), max_d_factor) for i in range(1,7)]
num_matrices = matrices_normalized.shape[0]

# Run the augmentation
matrices_augmented, labels_augmented, start_times_augmented, end_times_augmented = augment_data_with_distortion(
    portions_balanced, labels_balanced, start_times_balanced, end_times_balanced, augmentation_factors
)

for i in range(7):
    print(f"card classe {i}:",(labels_augmented == i).sum(), (labels_balanced == i).sum())



# %%
l2_lambda = 1e-4
l2_lambda_str = str(l2_lambda).replace(".", "p")
models_dir = f"trained_nn/model_nn_v20_3385_3382_3441_3444_3445_{l2_lambda_str}"
#models_dir = f"trained_nn/model_nn_v5_lambda_{l2_lambda_str}" #entrainé sur 3385 full et 3382 prw
x_val, y_val, val_predictions, val_predictions_raw, history = main(labels_augmented, 
                                                                    matrices_augmented,
                                                                    models_dir,
                                                                    epochs=12,
                                                                    num_classes=num_classes,
                                                                    l2_lambda=l2_lambda) 

# %%
print(val_predictions.shape)
num_classes=7
for i in range(num_classes):
    print(f"val={i}",(val_predictions==i).sum())

for i in range(num_classes):
    print(f"Portion classified as {i} :")
    display_matrices_by_class(x_val, val_predictions, i, 1)


# %%
plot_loss(history, models_dir+"/loss_precision_plot.png")
y_val_m = np.argmax(y_val, axis=1)
plot_confusion_matrix(y_val_m, val_predictions, np.arange(0,num_classes), f"{models_dir}/confusion_matrix_max_epoch.png")
plot_roc_curves(y_val, val_predictions_raw, np.arange(0,7,1), f"{models_dir}/roc_curves.png")

# %%
# Fonction pour vérifier le recouvrement entre deux rectangles
def check_overlap(portion1, portion2):
    start_time_1, end_time_1, start_freq_1, end_freq_1 = portion1
    start_time_2, end_time_2, start_freq_2, end_freq_2 = portion2
    
    # Chevauchement temporel
    overlap_time = max(0, min(end_time_1, end_time_2) - max(start_time_1, start_time_2))
    # Chevauchement fréquentiel
    overlap_freq = max(0, min(end_freq_1, end_freq_2) - max(start_freq_1, start_freq_2))
    
    return overlap_time > 0 and overlap_freq > 0

# %%
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
            detection_flags[i] = True  # Mark detection as validated

    return detection_flags


# Fonction principale de prédiction et de filtrage avec réseau de neurones
def neural_net_predict_on_audio_2(model_path, file_path, num_classes, context_size, seuil_PRW, seuil_non_1, seuil_non_2):
    # Charger le modèle Keras
    model = tf.keras.models.load_model(model_path)
    
    # Charger l'audio
    audio, sr = librosa.load(file_path, sr=None)
    audio_duration = librosa.get_duration(y=audio, sr=sr)
    
    # Calculer le spectrogramme
    hop_length = int(nfft * (1 - overlap_fft))
    S = librosa.stft(audio, n_fft=nfft, hop_length=hop_length, window=window_type_fft)
    S_magnitude = np.abs(S)
    # Assurez-vous que toutes les valeurs sont positives (ajoutez une petite constante si nécessaire)
    S_magnitude[S_magnitude == 0] = 1e-6  # Remplacer les zéros par un très petit nombre
    # Convertir en dB
    S_db = 20*np.log10(S_magnitude)
    
    # Création de la figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Afficher le spectrogramme
    img = librosa.display.specshow(S_db, sr=sr, hop_length=hop_length, x_axis='time', y_axis='hz', 
                                   cmap='viridis', fmin=min_frequency, fmax=max_frequency, ax=ax1)
    ax1.set_title(f"Spectrogramme avec prédictions ({file_path.split('/')[-1]})")
    ax1.set_ylim([0, max_frequency + 20])
    ax1.set_ylabel('Fréquence (Hz)')




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

        portion_start_time = x * step_time
        portions_start_times.append(portion_start_time)

    portions_start_times = np.array(portions_start_times)
    portions_dates = np.array(portions_dates)
    all_coordinates_px = np.array(all_coordinates_px)
    portions = np.array(portions)
    portions_expanded = np.expand_dims(portions, -1)  # Ajouter une dimension pour correspondre à (hauteur, largeur, 1)

    # Normalize each matrix individually
    matrices_normalized = np.array([normalize_matrix(matrix) for matrix in portions_expanded])
    
    # Prédictions du modèle
    predictions = model.predict(matrices_normalized)

    # Filtrer les détections par contexte
    detection_flags = validate_detections(predictions, context_size, seuil_PRW, seuil_non_1, seuil_non_2)
    
    # Appliquer le filtre de chevauchement pour n'afficher que la détection la plus forte
    final_rectangles = []
    passing_rectangles = []
    for i, portion in enumerate(all_coordinates_px):
        if detection_flags[i]:  # Prendre en compte seulement les détections validées
            start_time, end_time, start_freq, end_freq = portion
            prediction_value = predictions[i][1]
            passing_rectangles.append((portion, prediction_value, 1))
            
            overlap_found = False
            for j, (existing_portion, existing_prediction, _) in enumerate(final_rectangles):
                if check_overlap(portion, existing_portion):
                    # Remplacer la détection si elle est plus forte
                    if prediction_value > existing_prediction:
                        final_rectangles[j] = (portion, prediction_value, 1)
                    overlap_found = True
                    break
            
            if not overlap_found:
                final_rectangles.append((portion, prediction_value, 1))

    # Dessiner les rectangles qui valident
    for portion, prediction_value, _ in passing_rectangles:
        start_time_px, end_time_px, start_freq_px, end_freq_px = portion
        start_time = start_time_px*time_per_pixel
        end_time = end_time_px*time_per_pixel
        start_freq = start_freq_px*freq_per_pixel
        end_freq = end_freq_px*freq_per_pixel
        rect = patches.Rectangle((start_time, start_freq+min_frequency), end_time - start_time, end_freq - start_freq, 
                                 linewidth=1, edgecolor='blue', facecolor='none')
        ax1.add_patch(rect)

    # Dessiner les rectangles finaux
    for portion, prediction_value, _ in final_rectangles:
        start_time_px, end_time_px, start_freq_px, end_freq_px = portion
        start_time = start_time_px*time_per_pixel
        end_time = end_time_px*time_per_pixel
        start_freq = start_freq_px*freq_per_pixel
        end_freq = end_freq_px*freq_per_pixel
        rect = patches.Rectangle((start_time, start_freq+min_frequency), end_time - start_time, end_freq - start_freq, 
                                 linewidth=1, edgecolor='red', facecolor='none')
        ax1.add_patch(rect)
        ax1.text(start_time-5, end_freq + 2, f"{prediction_value * 100:.1f}%", color='red', fontsize=8, ha='center')

    # Afficher la courbe des prédictions
    ax2.set_title("Prédiction en fonction du temps (Lissée)")
    ax2.set_ylim([-0.1, 1.1])
    for class_idx in range(0,7):
        ax2.plot(portions_start_times, predictions[:, class_idx], label=f"Classe {class_idx}")
        #seuil PRW
    ax2.plot(portions_start_times, seuil_PRW*np.ones(all_coordinates_px[:, 0].shape), label="Seuil PRW", color="black", linestyle="--")
    ax2.plot(portions_start_times, seuil_non_1*np.ones(all_coordinates_px[:, 0].shape), label="Seuil non PRW", color="black", linestyle="-.")
    #somme des NON PRW
    ax2.plot(portions_start_times, np.sum(predictions[:, (2,3,4,5,6)], axis=1), label="Somme non PRW", color="grey")

    ax2.set_xlabel("Temps (s)")
    ax2.set_ylabel("Prédiction")
    ax2.legend(loc="upper right")

    # Ajouter la colorbar
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.87, 0.15, 0.02, 0.7])
    fig.colorbar(img, cax=cbar_ax, format='%+2.0f dB')

    plt.show()

    return predictions, matrices_normalized