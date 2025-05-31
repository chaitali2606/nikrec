# Run clustering scripts
import clustering_dbscan
import clustering_agglomerative
import clustering_spectral
import kmeans_elbow
import metrics_evaluation

import streamlit as st
import os
import json

st.title("🧠 Movie Clustering Explorer")

# --- Section: Recommendation Outputs ---
st.header("📊 Outputs From Clustering Process")
clustering_outputs = {
    "results/avg_ratings_dist.png": "Average Ratings Distribution",
    "results/wordcloud_titles.png": "WordCloud of Movie Titles",
    "results/tsne_plot.png": "t-SNE Plot of Movie Embeddings",
    "results/kmeans_clusters.png": "KMeans Clustering Visualization"
}

for path, caption in clustering_outputs.items():
    if os.path.exists(path):
        st.image(path, caption=caption)
    else:
        st.warning(f"Missing image: {caption}")

# --- Section: Clustering Evaluation Metrics ---
st.subheader("📋 Clustering Evaluation Metrics")

if os.path.exists("results/runtime_complexity.txt"):
    with open("results/runtime_complexity.txt") as f:
        complexity = f.read()
    st.text("Clustering Runtime Complexity:\n" + complexity)
else:
    st.warning("runtime_complexity.txt not found.")

if os.path.exists("results/ari_nmi_scores.txt"):
    with open("results/ari_nmi_scores.txt") as f:
        scores = f.read()
    st.text("ARI/NMI Scores:\n" + scores)
else:
    st.warning("ari_nmi_scores.txt not found.")

# --- Section: Elbow Curve ---
st.header("📌 KMeans Elbow Curve")
if os.path.exists("results/elbow_curve.png"):
    st.image("results/elbow_curve.png", caption="Elbow Curve for Optimal K")
else:
    st.warning("Elbow curve image not found.")

# --- Section: Comparison with Other Clustering Algorithms ---
st.header("🔍 Comparison of Clustering Techniques")
clustering_images = {
    "results/dbscan_clusters.png": "DBSCAN Clustering",
    "results/agglomerative_clusters.png": "Agglomerative Clustering",
    "results/spectral_clusters.png": "Spectral Clustering"
}

for path, caption in clustering_images.items():
    if os.path.exists(path):
        st.image(path, caption=caption)
    else:
        st.warning(f"{caption} image not found.")

st.success("✅ Clustering analysis results loaded successfully!")
