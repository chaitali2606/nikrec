import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import joblib
import pandas as pd

# Load SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load preprocessed movie data and embeddings
movies = pd.read_csv('/Users/chaitalichoudhary/Desktop/nikita/movie.csv', delimiter=',', encoding='utf-8', quotechar='"', on_bad_lines='skip')
X_scaled = joblib.load("results/X_scaled.pkl")

# Load title embeddings (optional if separate)
# If needed, recompute:
# movies['embedding'] = movies['title'].apply(lambda x: model.encode(x, normalize_embeddings=True))

# Recommendation Function
def recommend_movies(user_text, movies_df, embeddings, model, top_n=10):
    """
    Recommend top N similar movies based on cosine similarity.

    Args:
        user_text (str): Input query or movie description.
        movies_df (pd.DataFrame): DataFrame containing movie titles.
        embeddings (np.ndarray): Matrix of movie embeddings.
        model (SentenceTransformer): SentenceTransformer model.
        top_n (int): Number of recommendations to return.

    Returns:
        list of tuples: (title, similarity_score)
    """
    user_emb = model.encode(user_text, normalize_embeddings=True).reshape(1, -1)
    sims = cosine_similarity(user_emb, embeddings).flatten()
    top_idx = sims.argsort()[-top_n:][::-1]
    return [(movies_df.iloc[i]['title'], round(sims[i], 3)) for i in top_idx]

# Example usage
if __name__ == "__main__":
    query = input("Enter a movie title or description: ")
    recommendations = recommend_movies(query, movies, X_scaled, model, top_n=5)

    print("\nTop Recommendations:")
    for i, (title, score) in enumerate(recommendations, 1):
        print(f"{i}. {title} (score: {score})")
