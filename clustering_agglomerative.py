from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import joblib

X_scaled = joblib.load("results/X_scaled.pkl")
X_tsne = joblib.load("results/X_tsne.pkl")

agg = AgglomerativeClustering(n_clusters=10).fit(X_scaled)
labels = agg.labels_

plt.figure(figsize=(10, 8))
sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=labels, palette='tab10')
plt.title("Agglomerative Clustering")
plt.savefig("results/agglomerative_clusters.png")
