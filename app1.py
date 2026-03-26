# app1.py — Emirates Review Sentiment App (Updated & Modernised)
# =============================================================
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import seaborn as sns
from PIL import Image
import joblib
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud, STOPWORDS
import textwrap

# ─────────────────────────────────────────────
# PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Emirates · Sentiment Intelligence",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS — luxury dark theme
# ─────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* ── Google Font ─────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@400;600;700&family=DM+Sans:wght@300;400;500&display=swap');

    /* ── Root palette ───────────────────────── */
    :root {
        --ek-red:    #D71921;
        --ek-gold:   #C9963E;
        --ek-cream:  #F5EFE6;
        --ek-dark:   #0D0D0D;
        --ek-panel:  #161616;
        --ek-border: #2A2A2A;
        --ek-muted:  #888888;
        --ek-light:  #EFEFEF;
    }

    /* ── Global reset ───────────────────────── */
    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
        background-color: var(--ek-dark) !important;
        color: var(--ek-light) !important;
    }

    .stApp { background: var(--ek-dark) !important; }

    .block-container {
        padding: 2rem 2.5rem 4rem 2.5rem !important;
        max-width: 1300px;
    }

    /* ── Sidebar ────────────────────────────── */
    [data-testid="stSidebar"] {
        background: var(--ek-panel) !important;
        border-right: 1px solid var(--ek-border) !important;
    }
    
    /* ── Typography ─────────────────────────── */
    h1 {
        font-family: 'Cormorant Garamond', serif !important;
        font-size: 2.6rem !important;
        font-weight: 700 !important;
        color: var(--ek-cream) !important;
        margin-bottom: 0.25rem !important;
    }
    .ek-section-title {
        font-family: 'DM Sans', sans-serif;
        font-size: 0.7rem;
        font-weight: 500;
        letter-spacing: 0.18em;
        text-transform: uppercase;
        color: var(--ek-gold) !important;
        margin-bottom: 0.5rem;
    }

    /* ── Metric cards ───────────────────────── */
    [data-testid="metric-container"] {
        background: var(--ek-panel) !important;
        border: 1px solid var(--ek-border) !important;
        border-radius: 6px !important;
        padding: 1.2rem 1.4rem !important;
    }
    
    /* ── Buttons ────────────────────────────── */
    .stButton > button {
        background: transparent !important;
        border: 1px solid var(--ek-gold) !important;
        color: var(--ek-gold) !important;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        transition: all 0.2s ease;
    }
    .stButton > button:hover {
        background: var(--ek-gold) !important;
        color: var(--ek-dark) !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────
for key, default in {
    "results_df": None,
    "last_query": "",
    "pdf_bytes": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ─────────────────────────────────────────────
# LOAD DATA & MODELS
# ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data(path: str = "emirates_reviews.csv") -> pd.DataFrame:
    data = pd.read_csv(path)
    data["Recommended"] = data["Recommended"].astype(str).str.strip().str.lower()
    data["sentiment"] = data["Recommended"].map({"yes": "Positive", "no": "Negative"}).fillna("Neutral")
    return data

@st.cache_resource(show_spinner=False)
def load_ml_artifacts():
    vectorizer, model, loaded = None, None, False
    try:
        pipeline = joblib.load('pipeline_tfidf_logistic_binary.pkl')
        vectorizer = pipeline.named_steps['tfidf']
        model = pipeline.named_steps['clf']
        loaded = True
    except:
        try:
            vectorizer = joblib.load('tfidf_vectorizer.pkl')
            model = joblib.load('logistic_model_binary.pkl')
            loaded = True
        except: pass
    return vectorizer, model, loaded

df = load_data()
vectorizer, sentiment_model, pipeline_loaded = load_ml_artifacts()

# Vectorize for chatbot search
review_texts = df['Review'].fillna('').astype(str).tolist()
review_vectors = vectorizer.transform(review_texts) if vectorizer else None

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        """<div style="text-align:center; padding: 1.5rem 0;">
            <img src="https://upload.wikimedia.org/wikipedia/commons/d/d0/Emirates_logo.svg" width="140">
        </div>""", unsafe_allow_html=True
    )
    st.markdown("<p class='ek-section-title' style='text-align:center;'>Sentiment Intelligence</p>", unsafe_allow_html=True)
    st.markdown("<hr style='margin:0.5rem 0 1rem 0;'>", unsafe_allow_html=True)
    
    section = st.radio("Navigate", ["📊 Dashboard", "🤖 Chatbot"], label_visibility="collapsed")
    
    st.markdown("<hr style='margin:1rem 0 0.5rem 0;'>", unsafe_allow_html=True)
    if pipeline_loaded:
        st.success("✅ Models Loaded")
    else:
        st.warning("⚠️ Model Artifacts Missing")

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def highlight_keywords(text, keywords, color="yellow"):
    safe = str(text)
    for kw in sorted(set(keywords), key=len, reverse=True):
        if kw.strip():
            safe = safe.replace(kw, f"<mark style='background:{color};color:#111;border-radius:2px;padding:0 2px;'>{kw}</mark>")
    return safe

def get_col(df, name):
    cols = {c.lower(): c for c in df.columns}
    return cols.get(name.lower())

def create_pdf_from_df(results_df, title="Emirates Review Insights"):
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
    from reportlab.lib.units import inch
    from io import BytesIO
    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    y = 750
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, y, title)
    y -= 30
    c.setFont("Helvetica", 10)
    for i, row in results_df.iterrows():
        if y < 100: 
            c.showPage()
            y = 750
        txt = f"{i+1}. [{row.get('Sentiment')}] - {str(row.get('Review'))[:100]}..."
        c.drawString(50, y, txt)
        y -= 20
    c.save()
    buf.seek(0)
    return buf.read()

# ─────────────────────────────────────────────
# DASHBOARD
# ─────────────────────────────────────────────
if "Dashboard" in section:
    st.markdown("<p class='ek-section-title'>Analytics Overview</p><h1>Emirates Airline<br>Sentiment Analysis</h1>", unsafe_allow_html=True)
    
    # KPI Row
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Reviews", f"{len(df):,}")
    k2.metric("Positive", f"{round(100*(df['sentiment']=='Positive').sum()/len(df),1)}%")
    k3.metric("Negative", f"{round(100*(df['sentiment']=='Negative').sum()/len(df),1)}%")
    k4.metric("Avg Rating", round(df[get_col(df, 'Overall Rating')].mean(), 2) if get_col(df, 'Overall Rating') else "N/A")

    # Visuals
    st.markdown("<br><p class='ek-section-title'>01 — Sentiment Distribution</p>", unsafe_allow_html=True)
    fig1, ax1 = plt.subplots(figsize=(8, 3))
    fig1.patch.set_facecolor("#161616")
    ax1.set_facecolor("#161616")
    sns.countplot(data=df, x='sentiment', palette=['#2ECC71', '#D71921', '#C9963E'], ax=ax1)
    ax1.tick_params(colors="#888888")
    st.pyplot(fig1)

    st.markdown("<p class='ek-section-title'>02 — Global Reach</p>", unsafe_allow_html=True)
    if get_col(df, "Country"):
        c_counts = df[get_col(df, "Country")].value_counts().reset_index()
        c_counts.columns = ["Country", "Count"]
        fig_map = px.choropleth(c_counts, locations="Country", locationmode="country names", color="Count",
                                color_continuous_scale=[[0, "#1E1E1E"], [1, "#D71921"]])
        fig_map.update_layout(paper_bgcolor="#161616", plot_bgcolor="#161616", font_color="#EFEFEF", margin={"r":0,"t":0,"l":0,"b":0})
        st.plotly_chart(fig_map, use_container_width=True)

    # Static Images from GitHub
    for i, (label, img) in enumerate([("03 — Travel Class", "avg_rating_by_travel_class.png"), 
                                     ("04 — Travel Type", "avg_rating_by_travel_type.png"),
                                     ("05 — Service Ratings", "key_ratings_distribution.png")], 3):
        st.markdown(f"<p class='ek-section-title'>{i:02} — {label}</p>", unsafe_allow_html=True)
        try: st.image(Image.open(img), use_container_width=True)
        except: st.info(f"Asset `{img}` not found.")

# ─────────────────────────────────────────────
# CHATBOT
# ─────────────────────────────────────────────
elif "Chatbot" in section:
    st.markdown("<p class='ek-section-title'>AI Review Assistant</p><h1>Review Insights</h1>", unsafe_allow_html=True)
    
    if not vectorizer:
        st.error("Chatbot disabled: Missing model files.")
    else:
        user_query = st.text_input("Enter a topic (e.g. 'food', 'cabin crew')", value=st.session_state.last_query)
        if st.button("Search Reviews", type="primary"):
            st.session_state.last_query = user_query
            q_vec = vectorizer.transform([user_query])
            sims = cosine_similarity(q_vec, review_vectors).flatten()
            top_idx = sims.argsort()[-5:][::-1]
            
            st.markdown("<p class='ek-section-title'>Top Matches</p>", unsafe_allow_html=True)
            for idx in top_idx:
                sentiment = df.loc[idx, 'sentiment']
                color = "#2ECC71" if sentiment == "Positive" else "#D71921"
                st.markdown(f"""
                <div style="border:1px solid #2A2A2A; padding:15px; border-radius:5px; margin-bottom:10px; background:#161616;">
                    <span style="color:{color}; font-weight:bold;">{sentiment}</span> | Similarity: {sims[idx]:.2f}<br>
                    <p style="color:#BDBDBD; margin-top:5px;">"{df.loc[idx, 'Review']}"</p>
                </div>
                """, unsafe_allow_html=True)
