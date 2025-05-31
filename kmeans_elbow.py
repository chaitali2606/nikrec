import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import joblib

# Load precomputed scaled features
X_scaled = joblib.load("results/X_scaled.pkl")

# Elbow Method to determine optimal k
inertia = []
K = range(2, 15)
print("Computing inertia for KMeans with k from 2 to 14...")

for k in K:
    print(f"Fitting KMeans with k={k}")
    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

# Plot and save elbow curve
plt.figure(figsize=(8, 6))
plt.plot(K, inertia, marker='o')
plt.title("Elbow Method For Optimal k")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia (Within-cluster sum of squares)")
plt.grid(True)
plt.savefig("results/elbow_curve.png")
plt.close()

print("Elbow curve saved to results/elbow_curve.png")
