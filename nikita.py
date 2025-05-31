import os
import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity

from wordcloud import WordCloud

import torch
import torch.nn as nn
import joblib
from sentence_transformers import SentenceTransformer

# Create results directory
os.makedirs("results", exist_ok=True)

# --- Load Datasets ---
print("Loading datasets...")
movies = pd.read_csv('/Users/chaitalichoudhary/Desktop/nikita/movie.csv', delimiter=',', encoding='utf-8', quotechar='"', on_bad_lines='skip')
ratings = pd.read_csv('/Users/chaitalichoudhary/Desktop/nikita/rating.csv', delimiter=',', encoding='utf-8', quotechar='"', on_bad_lines='skip')

# --- Preprocess ---
movie_ratings = ratings.groupby('movieId').agg(
    avg_rating=('rating', 'mean'),
    rating_count=('rating', 'count')
).reset_index()

movies = movies.merge(movie_ratings, on='movieId', how='inner')

# --- Visualization: Rating Distribution ---
plt.figure(figsize=(8, 6))
sns.histplot(movies['avg_rating'], bins=30, kde=True)
plt.title("Average Movie Ratings")
plt.savefig("results/avg_ratings_dist.png")
plt.close()

# --- WordCloud on Titles ---
text = " ".join(m for m in movies['title'].astype(str))
wordcloud = WordCloud(background_color="white").generate(text)
plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title("WordCloud of Movie Titles")
plt.savefig("results/wordcloud_titles.png")
plt.close()

# --- Encode Titles using SentenceTransformer ---
print("Encoding titles using SentenceTransformer...")
model = SentenceTransformer('all-MiniLM-L6-v2')
movies['embedding'] = movies['title'].apply(lambda x: model.encode(x, normalize_embeddings=True))

# --- Dimensionality Reduction ---
print("Applying t-SNE...")
X = np.stack(movies['embedding'].values)
X_scaled = StandardScaler().fit_transform(X)
X_tsne = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(X_scaled)

plt.figure(figsize=(10, 8))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=movies['avg_rating'], cmap='viridis', alpha=0.6)
plt.colorbar(label='Avg Rating')
plt.title("t-SNE Visualization of Movie Embeddings")
plt.savefig("results/tsne_plot.png")
plt.close()

# --- KMeans Clustering ---
print("Clustering using KMeans...")
start = time.time()
kmeans = KMeans(n_clusters=10, random_state=42)
labels = kmeans.fit_predict(X_scaled)
end = time.time()

movies['cluster'] = labels
silhouette = silhouette_score(X_scaled, labels)

plt.figure(figsize=(10, 8))
sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=labels, palette="tab10")
plt.title(f"KMeans Clustering (Silhouette: {silhouette:.2f})")
plt.savefig("results/kmeans_clusters.png")
plt.close()

# --- Save Clustering Runtime & Metrics ---
with open("results/runtime_complexity.txt", "w") as f:
    f.write(f"KMeans Clustering Runtime: {end - start:.2f} seconds\n")
    f.write(f"Number of Clusters: 10\n")
    f.write(f"Silhouette Score: {silhouette:.2f}\n")

# --- Precision/Recall Evaluation ---
print("Evaluating clustering precision/recall...")
similarities = cosine_similarity(X_scaled)
precision_list, recall_list = [], []

for idx in range(len(similarities)):
    true_cluster = labels[idx]
    top5 = similarities[idx].argsort()[-6:-1]  # Top 5 excluding itself
    pred_clusters = labels[top5]
    precision = (pred_clusters == true_cluster).sum() / 5
    recall = (pred_clusters == true_cluster).sum() / (labels == true_cluster).sum()
    precision_list.append(precision)
    recall_list.append(recall)

avg_precision = np.mean(precision_list)
avg_recall = np.mean(recall_list)

with open("results/evaluation_metrics.txt", "w") as f:
    f.write(f"Average Precision@5: {avg_precision:.2f}\n")
    f.write(f"Average Recall@5: {avg_recall:.2f}\n")

# --- Train Autoencoder ---
print("Training Autoencoder...")
class Autoencoder(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(dim, 128), nn.ReLU(),
            nn.Linear(128, 32)
        )
        self.decoder = nn.Sequential(
            nn.Linear(32, 128), nn.ReLU(),
            nn.Linear(128, dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

ae_model = Autoencoder(X_scaled.shape[1])
optimizer = torch.optim.Adam(ae_model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

for epoch in range(10):
    ae_model.train()
    output = ae_model(X_tensor)
    loss = loss_fn(output, X_tensor)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

torch.save(ae_model.state_dict(), "results/autoencoder_model.pth")

with open("results/deep_learning_note.txt", "w") as f:
    f.write("Trained a simple autoencoder to compress and reconstruct movie embeddings.\n")
    f.write("Can be used for future anomaly detection or enhanced similarity comparisons.\n")

# --- Save Data for Reuse ---
joblib.dump(X_scaled, "results/X_scaled.pkl")
joblib.dump(X_tsne, "results/X_tsne.pkl")
joblib.dump(labels, "results/kmeans_labels.pkl")

print("✅ All clustering and deep learning processes complete. Check the 'results' folder.")
