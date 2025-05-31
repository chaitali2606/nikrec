import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load resources
@st.cache_resource
def load_model_and_data():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    movies = pd.read_csv('/Users/chaitalichoudhary/Desktop/nikita/movie.csv', delimiter=',', encoding='utf-8', quotechar='"', on_bad_lines='skip')
    embeddings = joblib.load("results/X_scaled.pkl")  # 🔁 Must match order
    return model, movies, embeddings

# Recommendation logic
def recommend_movies(user_text, movies_df, embeddings, model, top_n=5):
    user_emb = model.encode(user_text, normalize_embeddings=True).reshape(1, -1)
    sims = cosine_similarity(user_emb, embeddings).flatten()
    top_idx = sims.argsort()[-top_n:][::-1]
    return [(movies_df.iloc[i]['title'], round(sims[i], 3)) for i in top_idx]

# UI Layout
st.set_page_config(page_title="🎬 Movie Recommendation", layout="centered")
st.title("🎬 Smart Movie Recommendation System")
st.markdown("Get movie suggestions based on your input title or keywords.")

# Load data and model
model, movies, embeddings = load_model_and_data()

# Input from user
user_query = st.text_input("Enter a movie title or keyword:", "")

if user_query:
    results = recommend_movies(user_query, movies, embeddings, model, top_n=5)
    st.subheader(f"Top 5 Recommendations for: _{user_query}_")
    for idx, (title, score) in enumerate(results, 1):
        st.markdown(f"{idx}. **{title}** — Similarity: `{score}`")

# Footer
st.markdown("---")
st.caption("Built with ❤️ using Streamlit and Sentence Transformers")
