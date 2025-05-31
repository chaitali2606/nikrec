from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import joblib

X_scaled = joblib.load("results/X_scaled.pkl")
X_tsne = joblib.load("results/X_tsne.pkl")

db = DBSCAN(eps=1.5, min_samples=5).fit(X_scaled)
labels = db.labels_

plt.figure(figsize=(10, 8))
sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=labels, palette='tab10')
plt.title("DBSCAN Clustering")
plt.savefig("results/dbscan_clusters.png")
