import os
import numpy as np
import librosa
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import Parallel, delayed
import gc

# Suppress warnings globally
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Enhanced feature extraction function with clustering
def extract_features(file_path, n_mfcc=13, n_clusters=5):
    try:
        y, sr = librosa.load(file_path, sr=None)

        # Extract features
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(y))
        energy = np.mean(librosa.feature.rms(y=y))
        chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr), axis=1)
        spectral_contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr), axis=1)
        mel_spectrogram = np.mean(librosa.feature.melspectrogram(y=y, sr=sr), axis=1)
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
        spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))

        mfccs_mean = np.mean(mfccs, axis=1)

        # Perform KMeans clustering on the MFCC features
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        mfccs_cluster_labels = kmeans.fit_predict(mfccs.T)

        # Count the occurrences of each cluster label
        cluster_counts = np.bincount(mfccs_cluster_labels, minlength=n_clusters)

        # Concatenate all features
        features = np.concatenate((mfccs_mean, chroma, spectral_contrast, mel_spectrogram,
                                   [zero_crossing_rate, energy, spectral_centroid, spectral_bandwidth, spectral_rolloff], cluster_counts))

        return features
    except Exception as e:
        print(f"Error extracting features from {file_path}: {str(e)}")
        return None

train_data_dirs = {
    'Hebron': r'C:\Users\chris\OneDrive\Desktop\training\Hebron',
    'Jerusalem': r'C:\Users\chris\OneDrive\Desktop\training\Jerusalem',
    'Nablus': r'C:\Users\chris\OneDrive\Desktop\training\Nablus',
    'Ramallah_Reef': r'C:\Users\chris\OneDrive\Desktop\training\Ramallah_Reef'
}

# Paths to the test directories containing unlabeled test files
test_data_dirs = [
    r'C:\Users\chris\OneDrive\Desktop\testing\Ramallah-Reef',
    r'C:\Users\chris\OneDrive\Desktop\testing\Nablus',
    r'C:\Users\chris\OneDrive\Desktop\testing\Jerusalem',
    r'C:\Users\chris\OneDrive\Desktop\testing\Hebron'
]

# Function to load and extract features for a given set of data directories
def load_and_extract_features(data_dirs):
    X = []
    y = []
    for accent_label, dir_path in data_dirs.items():
        if os.path.exists(dir_path):
            files = [file for file in os.listdir(dir_path) if file.endswith('.wav')]
            features = Parallel(n_jobs=-1)(delayed(extract_features)(os.path.join(dir_path, file)) for file in files)
            features = [f for f in features if f is not None]  # Remove None values (failed extractions)
            X.extend(features)
            y.extend([accent_label] * len(features))
            gc.collect()  # Garbage collection to free memory
        else:
            print(f"Warning: Directory {dir_path} does not exist.")
    return np.array(X), np.array(y)

# Load and extract features from the training data
X_train, y_train = load_and_extract_features(train_data_dirs)

# Perform t-SNE for dimensionality reduction
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X_train)

# Plot the t-SNE visualization of the extracted features
plt.figure(figsize=(10, 7))
sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=y_train, palette='viridis', legend='full')
plt.title('t-SNE visualization of extracted features')
plt.xlabel('t-SNE component 1')
plt.ylabel('t-SNE component 2')
plt.legend(title='Accent')
plt.show()

# Split the training data into training and validation sets
X_train_split, X_val, y_train_split, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Define a pipeline with scaling and RandomForestClassifier
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Define the parameter grid for GridSearchCV
param_grid = {
    'classifier__n_estimators': [100, 200, 300, 400, 500],
    'classifier__max_depth': [10, 20, 30, 40, 50],
    'classifier__min_samples_split': [2, 5, 10, 15, 20],
    'classifier__min_samples_leaf': [1, 2, 4, 6, 8]
}

# Perform GridSearchCV to find the best parameters
grid_search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1, verbose=1)
grid_search.fit(X_train_split, y_train_split)

# Best parameters from GridSearchCV
print(f'Best Parameters: {grid_search.best_params_}')

# Evaluate the model on the validation set
y_val_pred = grid_search.predict(X_val)
val_accuracy = accuracy_score(y_val, y_val_pred)
print(f'Validation Accuracy: {val_accuracy}')
print(classification_report(y_val, y_val_pred, zero_division=1))

# Plot the confusion matrix for the validation set
cm = confusion_matrix(y_val, y_val_pred)
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=train_data_dirs.keys(), yticklabels=train_data_dirs.keys())
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix - Validation Set')
plt.show()

# Function to load and extract features from the test data
def load_and_extract_test_features(test_dirs):
    X = []
    test_files = []
    for test_dir in test_dirs:
        if os.path.exists(test_dir):
            files = [file for file in os.listdir(test_dir) if file.endswith('.wav')]
            features = Parallel(n_jobs=-1)(delayed(extract_features)(os.path.join(test_dir, file)) for file in files)
            features = [f for f in features if f is not None]  # Remove None values (failed extractions)
            X.extend(features)
            test_files.extend(files)
            gc.collect()  # Garbage collection to free memory
        else:
            print(f"Warning: Directory {test_dir} does not exist.")
    return np.array(X), test_files

# Load and extract features from the test data directories
X_test_unlabeled, test_files = load_and_extract_test_features(test_data_dirs)

# Predict the accents of the unlabeled test files using the trained model
y_test_unlabeled_pred = grid_search.predict(X_test_unlabeled)

# Display predictions for each test file
for file_path, predicted_label in zip(test_files, y_test_unlabeled_pred):
    print(f"File: {file_path}, Predicted Accent: {predicted_label}")

gc.collect()
