# sample_data.py
import joblib
from sklearn.utils import resample

# Load full scaled data
X_scaled = joblib.load("results/X_scaled.pkl")

# Downsample to 5000 points
X_sampled = resample(X_scaled, n_samples=5000, random_state=42)

# Save for use in other scripts
joblib.dump(X_sampled, "results/X_sampled.pkl")
print("Sampled dataset saved as X_sampled.pkl")
