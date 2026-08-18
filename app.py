import html as html_lib
from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
MOVIES_FILE = DATA_DIR / "movies.csv"
RATINGS_FILE = DATA_DIR / "ratings.csv"

# ---------------------------------------------------------------------------
# Design tokens — "single-screen movie theatre" palette.
# Warm near-black lobby walls, marquee gold, velvet-curtain maroon, and a
# cream ticket-paper tone used only inside the ticket cards.
# ---------------------------------------------------------------------------
BG = "#15100D"           # lobby wall (matches .streamlit/config.toml backgroundColor)
BG_PANEL = "#1F1712"     # box-office / card surface
CURTAIN = "#5C1A22"      # velvet maroon accent
GOLD = "#D4A72C"         # marquee gold
GOLD_BRIGHT = "#F4D976"  # hot bulb gold (glow / high-match)
PAPER = "#F3E9D2"        # ticket paper
INK = "#231A15"          # ink on paper

st.set_page_config(page_title="Film Recommendation", page_icon="🎬", layout="wide")

# ---------------------------------------------------------------------------
# Theme CSS
# ---------------------------------------------------------------------------
st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:wght@400;500;700&family=Space+Mono:wght@400;700&display=swap');

    :root {{
        --app-bg: {BG};
        --panel-bg: {BG_PANEL};
        --curtain: {CURTAIN};
        --gold: {GOLD};
        --gold-bright: {GOLD_BRIGHT};
        --paper: {PAPER};
        --ink: {INK};
    }}

    html, body, [class*="css"] {{
        font-family: 'DM Sans', sans-serif;
    }}

    .stApp {{
        background: radial-gradient(circle at 12% 8%, rgba(212,167,44,0.06) 0px, rgba(212,167,44,0) 55px) ,
                    radial-gradient(circle at 82% 4%, rgba(212,167,44,0.05) 0px, rgba(212,167,44,0) 45px),
                    radial-gradient(circle at 60% 92%, rgba(92,26,34,0.20) 0px, rgba(92,26,34,0) 60%),
                    var(--app-bg) !important;
    }}

    /* ---- Marquee header ---- */
    .marquee-title {{
        font-family: 'Bebas Neue', sans-serif;
        font-size: 3rem;
        letter-spacing: 0.12em;
        color: var(--gold-bright);
        text-shadow: 0 0 10px rgba(244,217,118,0.45), 0 0 2px rgba(244,217,118,0.6);
        margin-bottom: 0;
        line-height: 1.1;
    }}
    .marquee-sub {{
        font-family: 'Space Mono', monospace;
        font-size: 0.85rem;
        letter-spacing: 0.08em;
        color: #C9BBA6;
        text-transform: uppercase;
        margin-top: 2px;
    }}
    .marquee-rule {{
        height: 2px;
        background: repeating-linear-gradient(90deg, var(--gold) 0 10px, transparent 10px 18px);
        margin: 10px 0 22px 0;
        opacity: 0.7;
    }}

    /* ---- Sidebar: "Box Office" ---- */
    [data-testid="stSidebar"] {{
        background: linear-gradient(180deg, var(--curtain) 0%, #2A0E13 45%, var(--app-bg) 100%);
        border-right: 2px dotted rgba(212,167,44,0.35);
    }}
    [data-testid="stSidebar"] * {{
        color: var(--paper) !important;
    }}
    .box-office-label {{
        font-family: 'Bebas Neue', sans-serif;
        font-size: 1.5rem;
        letter-spacing: 0.15em;
        color: var(--gold-bright) !important;
        border-bottom: 1px dashed rgba(244,217,118,0.4);
        padding-bottom: 6px;
        margin-bottom: 10px;
    }}

    /* ---- Metrics ---- */
    [data-testid="stMetric"] {{
        background: var(--panel-bg);
        border: 1px solid rgba(212,167,44,0.25);
        border-radius: 10px;
        padding: 12px 16px;
    }}
    [data-testid="stMetricValue"] {{
        font-family: 'Bebas Neue', sans-serif;
        color: var(--gold-bright) !important;
        letter-spacing: 0.05em;
    }}
    [data-testid="stMetricLabel"] {{
        font-family: 'Space Mono', monospace;
        color: #C9BBA6 !important;
        text-transform: uppercase;
        font-size: 0.72rem !important;
        letter-spacing: 0.08em;
    }}

    /* ---- Tabs: ticket-booth windows ---- */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 4px;
        border-bottom: 1px solid rgba(212,167,44,0.25);
    }}
    .stTabs [data-baseweb="tab"] {{
        font-family: 'Bebas Neue', sans-serif;
        font-size: 1.05rem;
        letter-spacing: 0.08em;
        color: #C9BBA6;
        background: var(--panel-bg);
        border-radius: 8px 8px 0 0;
        padding: 8px 18px;
    }}
    .stTabs [aria-selected="true"] {{
        color: var(--gold-bright) !important;
        background: #26190F !important;
        box-shadow: inset 0 -3px 0 var(--gold);
    }}

    /* ---- Dataframe / table shell ---- */
    [data-testid="stDataFrame"] {{
        border: 1px solid rgba(212,167,44,0.25);
        border-radius: 10px;
        overflow: hidden;
    }}

    .section-caption {{
        font-family: 'Space Mono', monospace;
        font-size: 0.78rem;
        color: #C9BBA6;
        letter-spacing: 0.03em;
    }}

    /* ---- Ticket cards (the signature element) ---- */
    .ticket-wrapper {{
        background: var(--app-bg);
        padding: 8px 0;
    }}
    .ticket-card {{
        position: relative;
        display: flex;
        min-height: 118px;
        margin: 10px 2px;
        border-radius: 12px;
        background: linear-gradient(135deg, var(--paper) 0%, #EADFC1 100%);
        box-shadow: 0 10px 22px rgba(0,0,0,0.5);
        border: 1px solid rgba(35,26,21,0.15);
    }}
    .ticket-stub {{
        width: 64px;
        flex-shrink: 0;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        gap: 6px;
        background: repeating-linear-gradient(135deg, #1A130E, #1A130E 6px, #241A13 6px, #241A13 12px);
        border-radius: 12px 0 0 12px;
        color: var(--gold);
        writing-mode: vertical-rl;
        transform: rotate(180deg);
        padding: 10px 0;
    }}
    .ticket-stub .stub-label {{
        font-family: 'Bebas Neue', sans-serif;
        font-size: 0.95rem;
        letter-spacing: 0.2em;
    }}
    .ticket-stub .stub-no {{
        font-family: 'Space Mono', monospace;
        font-size: 0.65rem;
        opacity: 0.8;
        letter-spacing: 0.05em;
    }}
    .ticket-perforation {{
        border-left: 2px dashed rgba(35,26,21,0.3);
    }}
    .ticket-body {{
        flex: 1;
        display: flex;
        flex-direction: column;
        justify-content: center;
        gap: 4px;
        padding: 14px 84px 14px 18px;
        color: var(--ink);
    }}
    .ticket-genre {{
        font-family: 'Space Mono', monospace;
        font-size: 0.68rem;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: var(--curtain);
        font-weight: 700;
    }}
    .ticket-title {{
        font-family: 'Bebas Neue', sans-serif;
        font-size: 1.55rem;
        letter-spacing: 0.02em;
        line-height: 1.15;
    }}
    .ticket-tag {{
        display: inline-block;
        margin-top: 4px;
        font-family: 'Space Mono', monospace;
        font-size: 0.62rem;
        letter-spacing: 0.08em;
        background: var(--ink);
        color: var(--gold);
        padding: 2px 9px;
        border-radius: 20px;
        width: fit-content;
    }}
    .match-badge {{
        position: absolute;
        top: 16px;
        right: 16px;
        width: 58px;
        height: 58px;
        border-radius: 50%;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        border: 2px solid var(--ink);
    }}
    .match-badge .match-pct {{
        font-family: 'Bebas Neue', sans-serif;
        font-size: 1.05rem;
        line-height: 1;
        color: var(--ink);
    }}
    .match-badge .match-label {{
        font-family: 'Space Mono', monospace;
        font-size: 0.5rem;
        letter-spacing: 0.05em;
        color: var(--ink);
    }}
    .match-badge.tier-high {{
        background: radial-gradient(circle at 30% 30%, var(--gold-bright), var(--gold) 75%);
        box-shadow: 0 0 14px rgba(244,217,118,0.65), inset 0 0 0 2px rgba(255,255,255,0.35);
    }}
    .match-badge.tier-mid {{
        background: radial-gradient(circle at 30% 30%, #E4C463, var(--gold) 80%);
    }}
    .match-badge.tier-low {{
        background: radial-gradient(circle at 30% 30%, #C9BBA6, #A8987F 80%);
    }}
    .notch {{
        position: absolute;
        width: 20px;
        height: 20px;
        border-radius: 50%;
        top: 50%;
        transform: translateY(-50%);
        background: var(--app-bg);
        box-shadow: inset 0 0 5px rgba(0,0,0,0.35);
    }}
    .notch-left {{ left: -10px; }}
    .notch-right {{ right: -10px; }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="marquee-title">🎬 FILM RECOMMENDATION</div>
    <div class="marquee-sub">Tonight's showings · Genre-based recommendation engine</div>
    <div class="marquee-rule"></div>
    """,
    unsafe_allow_html=True,
)

if not MOVIES_FILE.exists() or not RATINGS_FILE.exists():
    st.warning("Place `movies.csv` and `ratings.csv` inside `data/`.")
    st.stop()

movies = pd.read_csv(MOVIES_FILE)
ratings = pd.read_csv(RATINGS_FILE)
movies = movies.copy()
ratings = ratings.copy()
movies["genres_clean"] = movies["genres"].str.replace("|", " ", regex=False)

st.sidebar.markdown('<div class="box-office-label">🎟️ Box Office</div>', unsafe_allow_html=True)
st.sidebar.caption("Filters apply to the Overview tab.")
genre_list = sorted({g for s in movies["genres"].dropna() for g in s.split("|")})
genre_filter = st.sidebar.multiselect("Genres", genre_list, default=genre_list[:8])
min_ratings = st.sidebar.slider("Min ratings per movie", 1, 500, 50)
st.sidebar.caption("`Min ratings` applies to the Data tab.")

# Movies matching the selected genres — this is now actually wired into the
# Overview tab below (previously computed but unused).
filtered_movies = movies[
    movies["genres"].fillna("").apply(lambda x: any(g in x.split("|") for g in genre_filter))
].copy()

col1, col2, col3 = st.columns(3)
col1.metric("Movies", f"{len(movies):,}".replace(",", "."))
col2.metric("Ratings", f"{len(ratings):,}".replace(",", "."))
col3.metric("Genres in view", f"{len(genre_filter)}/{len(genre_list)}")

tab_overview, tab_reco, tab_data = st.tabs(["Overview", "Recommend", "Data"])

with tab_overview:
    if filtered_movies.empty:
        st.warning("No genres selected — pick at least one genre in the sidebar to see the charts.")
    else:
        st.markdown(
            f'<span class="section-caption">Showing {len(filtered_movies):,} movies '
            f"across {len(genre_filter)} selected genre(s).</span>".replace(",", "."),
            unsafe_allow_html=True,
        )
        st.write("")
        left, right = st.columns(2)
        with left:
            st.caption("Genre mix (selected genres)")
            filtered_movie_genres = filtered_movies.assign(
                genres_split=filtered_movies["genres"].str.split("|")
            ).explode("genres_split")
            genre_counts = (
                filtered_movie_genres["genres_split"].value_counts().head(10).reset_index()
            )
            genre_counts.columns = ["genre", "count"]
            genre_counts = genre_counts.set_index("genre")
            st.bar_chart(genre_counts, use_container_width=True, color="#D4A72C")
        with right:
            st.caption("Rating distribution (selected genres only)")
            filtered_ratings = ratings.merge(
                filtered_movies[["movieId"]], on="movieId", how="inner"
            )
            if filtered_ratings.empty:
                st.info("No ratings found for the selected genres.")
            else:
                rating_counts = filtered_ratings["rating"].value_counts().sort_index().reset_index()
                rating_counts.columns = ["rating", "count"]
                rating_counts = rating_counts.set_index("rating")
                st.bar_chart(rating_counts, use_container_width=True, color="#5C1A22")

with tab_reco:
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(movies["genres_clean"])
    sim = cosine_similarity(tfidf_matrix)
    title = st.selectbox("Pick an anchor movie", movies["title"].sort_values().tolist())
    top_n = st.slider("Top N recommendations", 3, 10, 5)
    idx = movies.index[movies["title"] == title][0]
    scores = list(enumerate(sim[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1 : top_n + 1]
    rec = movies.iloc[[i for i, _ in scores]][["movieId", "title", "genres"]].copy()
    rec["similarity"] = [round(s, 3) for _, s in scores]

    st.write("")

    def render_ticket(movie_id, movie_title, genres, similarity, seat_no):
        pct = int(round(similarity * 100))
        tier = "tier-high" if pct >= 70 else "tier-mid" if pct >= 40 else "tier-low"
        safe_title = html_lib.escape(str(movie_title))
        safe_genres = html_lib.escape(str(genres).replace("|", "  ·  "))
        return f"""
        <div class="ticket-wrapper">
          <div class="ticket-card">
            <div class="notch notch-left"></div>
            <div class="ticket-stub">
              <span class="stub-label">ADMIT ONE</span>
              <span class="stub-no">NO. {int(movie_id):05d}</span>
            </div>
            <div class="ticket-perforation"></div>
            <div class="ticket-body">
              <span class="ticket-genre">{safe_genres}</span>
              <span class="ticket-title">{safe_title}</span>
              <span class="ticket-tag">SEAT {seat_no:02d} · NOW SHOWING</span>
            </div>
            <div class="match-badge {tier}">
              <span class="match-pct">{pct}%</span>
              <span class="match-label">MATCH</span>
            </div>
            <div class="notch notch-right"></div>
          </div>
        </div>
        """

    tickets_html = "".join(
        render_ticket(row.movieId, row.title, row.genres, row.similarity, seat_no=i + 1)
        for i, row in enumerate(rec.itertuples(index=False))
    )
    st.markdown(tickets_html, unsafe_allow_html=True)
    st.markdown(
        '<span class="section-caption">🎞️ Match % is genre TF-IDF + cosine similarity '
        "against the anchor movie.</span>",
        unsafe_allow_html=True,
    )

with tab_data:
    st.markdown('<span class="section-caption">Box-office ledger — top-rated movies</span>', unsafe_allow_html=True)
    st.write("")
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
    movie_rating_stats["avg_rating"] = movie_rating_stats["avg_rating"].round(2)
    st.dataframe(movie_rating_stats, use_container_width=True)
