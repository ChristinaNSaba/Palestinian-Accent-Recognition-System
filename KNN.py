import os
import numpy as np
import librosa
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.model_selection import GridSearchCV, cross_val_score
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Function to extract multiple audio features
def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=None)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        energy = librosa.feature.rms(y=y)
        zero_crossings = librosa.feature.zero_crossing_rate(y)
        chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
        mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        # Calculate the mean of the features
        mfccs_mean = np.mean(mfccs.T, axis=0)
        energy_mean = np.mean(energy.T, axis=0)
        zero_crossings_mean = np.mean(zero_crossings.T, axis=0)
        chroma_stft_mean = np.mean(chroma_stft.T, axis=0)
        mel_spectrogram_mean = np.mean(mel_spectrogram.T, axis=0)
        spectral_contrast_mean = np.mean(spectral_contrast.T, axis=0)
        # Concatenate all features into a single feature vector
        feature_vector = np.hstack([mfccs_mean, energy_mean, zero_crossings_mean,
                                    chroma_stft_mean, mel_spectrogram_mean, spectral_contrast_mean])
        return feature_vector
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# Load data from a specified directory
def load_data(data_path):
    X, y, file_paths = [], [], []
    for root, dirs, files in os.walk(data_path):
        for file in files:
            if file.endswith('.wav'):
                file_path = os.path.join(root, file)
                label = os.path.basename(root)
                features = extract_features(file_path)
                if features is not None:
                    X.append(features)
                    y.append(label)
                    file_paths.append(file_path)
    return np.array(X), np.array(y), file_paths

# Paths to the datasets
train_data_path = r'C:\Users\chris\OneDrive\Desktop\training'
test_data_path = r'C:\Users\chris\OneDrive\Desktop\testing'

# Load training data
X_train, y_train, train_file_paths = load_data(train_data_path)

# Load testing data
X_test, y_test, test_file_paths = load_data(test_data_path)

# Normalize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Dimensionality reduction with PCA
pca = PCA(n_components=30)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# Hyperparameter tuning for KNN
param_grid = {'n_neighbors': np.arange(1, 31), 'metric': ['euclidean', 'manhattan', 'minkowski']}
knn = KNeighborsClassifier()
knn_cv = GridSearchCV(knn, param_grid, cv=5)
knn_cv.fit(X_train_pca, y_train)

# Best parameters
best_params = knn_cv.best_params_
best_k = best_params['n_neighbors']
best_metric = best_params['metric']
print(f"\nBest parameters: {best_params}")

# Initialize and train the KNN classifier with the best parameters
knn = KNeighborsClassifier(n_neighbors=best_k, metric=best_metric)
knn.fit(X_train_pca, y_train)

# Cross-validation score
cv_scores = cross_val_score(knn, X_train_pca, y_train, cv=5)
print(f"\nCross-validation accuracy: {np.mean(cv_scores):.2f} (+/- {np.std(cv_scores):.2f})")

# Make predictions
y_pred_train = knn.predict(X_train_pca)
y_pred_test = knn.predict(X_test_pca)

# Calculate performance metrics for training data
accuracy_train = accuracy_score(y_train, y_pred_train)
precision_train = precision_score(y_train, y_pred_train, average='weighted', zero_division=0)
recall_train = recall_score(y_train, y_pred_train, average='weighted', zero_division=0)
f1_train = f1_score(y_train, y_pred_train, average='weighted', zero_division=0)

# Calculate performance metrics for testing data
accuracy_test = accuracy_score(y_test, y_pred_test)
precision_test = precision_score(y_test, y_pred_test, average='weighted', zero_division=0)
recall_test = recall_score(y_test, y_pred_test, average='weighted', zero_division=0)
f1_test = f1_score(y_test, y_pred_test, average='weighted', zero_division=0)

# Print the results
print("\nTraining Data Performance:")
print(f"Accuracy: {accuracy_train:.2f}")
print(f"Precision: {precision_train:.2f}")
print(f"Recall: {recall_train:.2f}")
print(f"F1 Score: {f1_train:.2f}")

print("\nTesting Data Performance:")
print(f"Accuracy: {accuracy_test:.2f}")
print(f"Precision: {precision_test:.2f}")
print(f"Recall: {recall_test:.2f}")
print(f"F1 Score: {f1_test:.2f}")

# Ensure the labels in the classification report match the unique labels in the testing set
unique_labels = np.unique(np.concatenate((y_train, y_test)))
print("\nClassification Report for Test Data:")
print(classification_report(y_test, y_pred_test, labels=unique_labels, zero_division=0))

# Detailed results for each audio file in the test set
test_results = pd.DataFrame({
    'File Path': test_file_paths,
    'True Label': y_test,
    'Predicted Label': y_pred_test
})
print("\nDetailed Results for Each Test Audio File:")
print(test_results)

# Print predictions for each accent in the testing data
print("\nPredictions for Each Accent in the Testing Data:")
accents = np.unique(y_test)
for accent in accents:
    print(f"\nAccent: {accent}")
    accent_files = test_results[test_results['True Label'] == accent]
    for _, row in accent_files.iterrows():
        print(f"File: {row['File Path']}, True Label: {row['True Label']}, Predicted Label: {row['Predicted Label']}")

# Apply KMeans clustering
kmeans = KMeans(n_clusters=5, random_state=42)
clusters = kmeans.fit_predict(X_train_pca)

# Perform t-SNE for visualization
tsne = TSNE(n_components=2, random_state=42)
X_train_tsne = tsne.fit_transform(X_train_pca)

# Create a DataFrame for plotting
df_tsne = pd.DataFrame({'X': X_train_tsne[:, 0], 'Y': X_train_tsne[:, 1], 'Cluster': clusters, 'Label': y_train})

# Plot the t-SNE visualization with clusters
plt.figure(figsize=(10, 7))
sns.scatterplot(x='X', y='Y', hue='Cluster', palette='viridis', data=df_tsne, legend='full')
for label in np.unique(y_train):
    x = df_tsne[df_tsne['Label'] == label]['X'].mean()
    y = df_tsne[df_tsne['Label'] == label]['Y'].mean()
    plt.text(x, y, label, fontsize=12, ha='center')
plt.title('t-SNE visualization of training data with KMeans clusters')
plt.xlabel('t-SNE component 1')
plt.ylabel('t-SNE component 2')
plt.legend(title='Cluster')
plt.show()