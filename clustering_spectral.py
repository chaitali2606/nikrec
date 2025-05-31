from sklearn.cluster import SpectralClustering
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import joblib
import random

# Load precomputed embeddings and t-SNE results
X_scaled = joblib.load("results/X_scaled.pkl")
X_tsne = joblib.load("results/X_tsne.pkl")

# Sample the data for efficiency
SAMPLE_SIZE = 5000  # Can be tuned
random.seed(42)
indices = sorted(random.sample(range(len(X_scaled)), SAMPLE_SIZE))
X_sample = X_scaled[indices]
X_tsne_sample = X_tsne[indices]

# Optional: Dimensionality reduction before clustering (optional, helps performance)
X_pca = PCA(n_components=50, random_state=42).fit_transform(X_sample)

# Perform spectral clustering on reduced sample
print("Running Spectral Clustering on sample...")
spec = SpectralClustering(
    n_clusters=10,
    affinity='nearest_neighbors',
    assign_labels='kmeans',
    random_state=42,
    n_neighbors=10
)
labels = spec.fit_predict(X_pca)

# Plotting
plt.figure(figsize=(10, 8))
sns.scatterplot(x=X_tsne_sample[:, 0], y=X_tsne_sample[:, 1], hue=labels, palette='tab10', legend=None)
plt.title("Spectral Clustering (on 5K Sample)")
plt.savefig("results/spectral_clusters.png")
plt.close()
