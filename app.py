from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
MOVIES_FILE = DATA_DIR / "movies.csv"
RATINGS_FILE = DATA_DIR / "ratings.csv"


st.set_page_config(page_title="Film Recommendation", page_icon="🎬", layout="wide")
st.title("Film Recommendation")
st.caption("Interactive Streamlit version of the film recommender project.")

if not MOVIES_FILE.exists() or not RATINGS_FILE.exists():
    st.warning("Place `movies.csv` and `ratings.csv` inside `data/`.")
    st.stop()

movies = pd.read_csv(MOVIES_FILE)
ratings = pd.read_csv(RATINGS_FILE)
movies = movies.copy()
ratings = ratings.copy()
movies["genres_clean"] = movies["genres"].str.replace("|", " ", regex=False)

st.sidebar.header("Filters")
genre_list = sorted({g for s in movies["genres"].dropna() for g in s.split("|")})
genre_filter = st.sidebar.multiselect("Genres", genre_list, default=genre_list[:8])
min_ratings = st.sidebar.slider("Min ratings per movie", 1, 500, 50)

movie_genres = movies.assign(genres_split=movies["genres"].str.split("|")).explode("genres_split")
filtered_movies = movies[
    movies["genres"].fillna("").apply(lambda x: any(g in x.split("|") for g in genre_filter))
].copy()

col1, col2 = st.columns(2)
col1.metric("Movies", f"{len(movies):,}".replace(",", "."))
col2.metric("Ratings", f"{len(ratings):,}".replace(",", "."))

tab_overview, tab_reco, tab_data = st.tabs(["Overview", "Recommend", "Data"])

with tab_overview:
    left, right = st.columns(2)
    with left:
        genre_counts = movie_genres["genres_split"].value_counts().head(10).reset_index()
        genre_counts.columns = ["genre", "count"]
        st.plotly_chart(
            px.bar(genre_counts, x="count", y="genre", orientation="h", title="Top Genres"),
            use_container_width=True,
        )
    with right:
        st.plotly_chart(
            px.histogram(ratings, x="rating", nbins=10, title="Rating Distribution"),
            use_container_width=True,
        )

with tab_reco:
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(movies["genres_clean"])
    sim = cosine_similarity(tfidf_matrix)
    title = st.selectbox("Pick an anchor movie", movies["title"].sort_values().tolist())
    top_n = st.slider("Top N recommendations", 3, 10, 5)
    idx = movies.index[movies["title"] == title][0]
    scores = list(enumerate(sim[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1 : top_n + 1]
    rec = movies.iloc[[i for i, _ in scores]][["title", "genres"]].copy()
    rec["similarity"] = [round(s, 3) for _, s in scores]
    st.dataframe(rec, use_container_width=True)
    st.write("Similarity is based on genre TF-IDF + cosine similarity.")

with tab_data:
    movie_rating_stats = (
        ratings.merge(movies, on="movieId", how="left")
        .groupby("title")
        .agg(avg_rating=("rating", "mean"), rating_count=("rating", "size"))
        .reset_index()
    )
    movie_rating_stats = movie_rating_stats[movie_rating_stats["rating_count"] >= min_ratings]
    movie_rating_stats = movie_rating_stats.sort_values(
        ["avg_rating", "rating_count"], ascending=[False, False]
    ).head(250)
    st.dataframe(movie_rating_stats, use_container_width=True)
