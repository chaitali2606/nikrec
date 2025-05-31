from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import joblib
from sklearn.cluster import AgglomerativeClustering, KMeans

X_sampled = joblib.load("results/X_sampled.pkl")

# Re-cluster with KMeans & Agglomerative on same sampled data
kmeans_labels = KMeans(n_clusters=10, random_state=42).fit_predict(X_sampled)
agg_labels = AgglomerativeClustering(n_clusters=10).fit_predict(X_sampled)

ari = adjusted_rand_score(kmeans_labels, agg_labels)
nmi = normalized_mutual_info_score(kmeans_labels, agg_labels)

with open("results/ari_nmi_scores.txt", "w") as f:
    f.write(f"Adjusted Rand Index (ARI): {ari:.2f}\n")
    f.write(f"Normalized Mutual Information (NMI): {nmi:.2f}")
