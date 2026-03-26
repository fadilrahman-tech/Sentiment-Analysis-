# app1.py — Emirates Review Sentiment App (Updated Logo)
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

    /* ── App wrapper ────────────────────────── */
    .stApp {
        background: var(--ek-dark) !important;
    }

    /* ── Main content padding ───────────────── */
    .block-container {
        padding: 2rem 2.5rem 4rem 2.5rem !important;
        max-width: 1300px;
    }

    /* ── Sidebar ────────────────────────────── */
    [data-testid="stSidebar"] {
        background: var(--ek-panel) !important;
        border-right: 1px solid var(--ek-border) !important;
    }
    [data-testid="stSidebar"] .stRadio label {
        color: var(--ek-light) !important;
        font-size: 0.88rem;
        letter-spacing: 0.05em;
    }
    [data-testid="stSidebar"] hr {
        border-color: var(--ek-border);
    }

    /* ── Page headings ──────────────────────── */
    h1 {
        font-family: 'Cormorant Garamond', serif !important;
        font-size: 2.6rem !important;
        font-weight: 700 !important;
        color: var(--ek-cream) !important;
        letter-spacing: -0.02em;
        line-height: 1.1;
        margin-bottom: 0.25rem !important;
    }
    h2, h3 {
        font-family: 'Cormorant Garamond', serif !important;
        color: var(--ek-cream) !important;
        letter-spacing: -0.01em;
    }

    /* ── Section subheaders ─────────────────── */
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
    [data-testid="metric-container"] label {
        color: var(--ek-muted) !important;
        font-size: 0.72rem !important;
        letter-spacing: 0.12em !important;
        text-transform: uppercase;
    }
    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: var(--ek-cream) !important;
        font-family: 'Cormorant Garamond', serif !important;
        font-size: 2rem !important;
    }
    [data-testid="metric-container"] [data-testid="stMetricDelta"] {
        color: var(--ek-gold) !important;
    }

    /* ── Plotly / chart backgrounds ─────────── */
    .js-plotly-plot .plotly .main-svg {
        background: transparent !important;
    }

    /* ── Buttons ────────────────────────────── */
    .stButton > button {
        background: transparent !important;
        border: 1px solid var(--ek-gold) !important;
        color: var(--ek-gold) !important;
        border-radius: 3px !important;
        font-size: 0.78rem !important;
        letter-spacing: 0.12em !important;
        text-transform: uppercase;
        padding: 0.5rem 1.2rem !important;
        transition: all 0.2s ease;
    }
    .stButton > button:hover {
        background: var(--ek-gold) !important;
        color: var(--ek-dark) !important;
    }

    /* ── Text input ─────────────────────────── */
    .stTextInput > div > div > input {
        background: var(--ek-panel) !important;
        border: 1px solid var(--ek-border) !important;
        border-radius: 3px !important;
        color: var(--ek-light) !important;
        font-family: 'DM Sans', sans-serif;
    }
    .stTextInput > div > div > input:focus {
        border-color: var(--ek-gold) !important;
        box-shadow: 0 0 0 2px rgba(201,150,62,0.15) !important;
    }

    /* ── Download button ────────────────────── */
    .stDownloadButton > button {
        background: var(--ek-red) !important;
        border: none !important;
        color: white !important;
        border-radius: 3px !important;
        font-size: 0.78rem !important;
        letter-spacing: 0.1em !important;
        text-transform: uppercase;
    }

    /* ── Expander ───────────────────────────── */
    .streamlit-expanderHeader {
        background: var(--ek-panel) !important;
        border: 1px solid var(--ek-border) !important;
        border-radius: 4px !important;
        color: var(--ek-light) !important;
    }
    .streamlit-expanderContent {
        background: var(--ek-panel) !important;
        border: 1px solid var(--ek-border) !important;
    }

    /* ── Dividers ───────────────────────────── */
    hr { border-color: var(--ek-border) !important; }

    /* ── Info / Warning / Error alerts ──────── */
    .stAlert {
        border-radius: 4px !important;
        font-size: 0.82rem !important;
    }

    /* ── Scrollbar ──────────────────────────── */
    ::-webkit-scrollbar { width: 5px; }
    ::-webkit-scrollbar-track { background: var(--ek-dark); }
    ::-webkit-scrollbar-thumb { background: var(--ek-border); border-radius: 10px; }
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
# SIDEBAR — Logo + navigation
# ─────────────────────────────────────────────
with st.sidebar:
    # Updated official Emirates Logo from Wikimedia
    st.markdown(
        """
        <div style="text-align:center; padding: 1.5rem 0 1rem 0;">
            <img src="https://upload.wikimedia.org/wikipedia/commons/d/d0/Emirates_logo.svg" 
                 width="140" style="margin-bottom: 10px;">
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        "<p class='ek-section-title' style='text-align:center;'>Sentiment Intelligence</p>",
        unsafe_allow_html=True,
    )
    st.markdown("<hr style='margin:0.5rem 0 1rem 0;'>", unsafe_allow_html=True)

    section = st.radio(
        "Navigate",
        ["📊  Dashboard", "🤖  Chatbot"],
        label_visibility="collapsed",
    )
    section = section.split("  ", 1)[1]   # strip the emoji prefix

    st.markdown("<hr style='margin:1rem 0 0.5rem 0;'>", unsafe_allow_html=True)
    st.caption("Emirates Airlines · Review Analytics")
    st.caption("Model: Logistic Regression (TF-IDF)")

# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data(path: str = "emirates_reviews.csv") -> pd.DataFrame:
    data = pd.read_csv(path)
    data["Recommended"] = data["Recommended"].astype(str).str.strip().str.lower()
    data["sentiment"] = data["Recommended"].map({"yes": "Positive", "no": "Negative"}).fillna("Neutral")
    return data


try:
    df = load_data()
except FileNotFoundError:
    st.error("⚠️  `emirates_reviews.csv` not found. Please place the file in the app directory.")
    st.stop()

# ─────────────────────────────────────────────
# LOAD ML ARTIFACTS
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    vectorizer_ = None
    model_ = None
    pipeline_ = None
    pipeline_loaded_ = False

    try:
        pipeline_ = joblib.load("pipeline_tfidf_logistic_binary.pkl")
        if hasattr(pipeline_, "named_steps") and "tfidf" in pipeline_.named_steps and "clf" in pipeline_.named_steps:
            vectorizer_ = pipeline_.named_steps["tfidf"]
            model_ = pipeline_.named_steps["clf"]
            pipeline_loaded_ = True
        else:
            pipeline_ = None
    except FileNotFoundError:
        pipeline_ = None
    except Exception:
        pipeline_ = None

    if not pipeline_loaded_:
        try:
            vectorizer_ = joblib.load("tfidf_vectorizer.pkl")
            model_ = joblib.load("logistic_model_binary.pkl")
        except Exception:
            vectorizer_ = None
            model_ = None

    return vectorizer_, model_, pipeline_loaded_


vectorizer, sentiment_model, pipeline_loaded = load_model()

if vectorizer and sentiment_model:
    st.sidebar.success("✅  Model loaded successfully")
else:
    st.sidebar.warning("⚠️  Model files not found — chatbot limited")

# Pre-vectorise all reviews for similarity search
review_texts = df["Review"].fillna("").astype(str).tolist()
review_vectors = None
if vectorizer is not None:
    try:
        review_vectors = vectorizer.transform(review_texts)
    except Exception:
        review_vectors = None

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def get_top_features(vec, model, n: int = 12):
    if model is None or vec is None:
        return [], []
    feature_names = vec.get_feature_names_out()
    if len(model.classes_) != 2:
        return [], []
    coef = model.coef_[0]
    pos_words = [feature_names[i] for i in np.argsort(coef)[-n:]]
    neg_words = [feature_names[i] for i in np.argsort(coef)[:n]]
    return pos_words, neg_words


top_positive_words, top_negative_words = get_top_features(vectorizer, sentiment_model)


def get_col(dataframe: pd.DataFrame, name: str):
    """Case-insensitive column getter; returns real column name or None."""
    mapping = {c.lower(): c for c in dataframe.columns}
    return mapping.get(name.lower())


def highlight_keywords(text: str, keywords: list, color: str = "yellow") -> str:
    safe = str(text)
    for kw in sorted(set(keywords), key=len, reverse=True):
        if kw.strip():
            safe = safe.replace(
                kw,
                f"<mark style='background:{color};color:#111;border-radius:2px;"
                f"padding:0 2px;font-weight:500;'>{kw}</mark>",
            )
    return safe


def build_results_subset(dataframe: pd.DataFrame, indices: list) -> pd.DataFrame:
    wanted = ["Review", "sentiment", "Overall Rating", "Date Published", "Country"]
    real_cols = [get_col(dataframe, c) for c in wanted if get_col(dataframe, c)]
    subset = dataframe.iloc[indices][real_cols].copy()
    rename = {
        get_col(dataframe, "Review"): "Review",
        get_col(dataframe, "sentiment"): "Sentiment",
        get_col(dataframe, "Overall Rating"): "Overall Rating",
        get_col(dataframe, "Date Published"): "Date Published",
        get_col(dataframe, "Country"): "Country",
    }
    subset.rename(columns={k: v for k, v in rename.items() if k}, inplace=True)
    return subset


def create_pdf_from_df(results_df: pd.DataFrame, title: str = "Emirates Review Insights") -> bytes:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas as rl_canvas
    from reportlab.lib.units import inch
    from io import BytesIO

    buf = BytesIO()
    c = rl_canvas.Canvas(buf, pagesize=letter)
    W, H = letter
    ML = MR = MT = MB = 0.75 * inch
    max_w = W - ML - MR
    y = H - MT

    # Title
    c.setFont("Helvetica-Bold", 14)
    c.drawString(ML, y, title)
    y -= 22
    c.setFont("Helvetica", 10)

    def wrapped_draw(txt: str, y_pos: float, lh: int = 14) -> float:
        for line in textwrap.TextWrapper(width=95).wrap(txt):
            if y_pos < MB + 20:
                c.showPage()
                c.setFont("Helvetica", 10)
                y_pos = H - MT
            c.drawString(ML, y_pos, line)
            y_pos -= lh
        return y_pos

    if results_df is None or results_df.empty:
        y = wrapped_draw("No results to export.", y)
    else:
        for i, row in results_df.iterrows():
            review = str(row.get("Review", "N/A")).strip()
            sentiment = row.get("Sentiment", "N/A")
            rating = row.get("Overall Rating", "N/A")
            date = row.get("Date Published", "N/A")
            country = row.get("Country", "N/A")
            header = f"{i+1}. Sentiment: {sentiment} | Rating: {rating} | Date: {date} | Country: {country}"
            y = wrapped_draw(header, y)
            y = wrapped_draw(f"Review: {review}", y)
            y -= 6

    c.save()
    buf.seek(0)
    return buf.read()


# ─────────────────────────────────────────────
# CHART STYLE HELPER
# ─────────────────────────────────────────────
def apply_dark_style(fig, ax):
    """Apply Emirates dark theme to a matplotlib figure/axes."""
    fig.patch.set_facecolor("#161616")
    ax.set_facecolor("#1E1E1E")
    for spine in ax.spines.values():
        spine.set_edgecolor("#2A2A2A")
    ax.tick_params(colors="#888888", labelsize=9)
    ax.xaxis.label.set_color("#888888")
    ax.yaxis.label.set_color("#888888")
    ax.title.set_color("#F5EFE6")


# ══════════════════════════════════════════════
# DASHBOARD
# ══════════════════════════════════════════════
if section == "Dashboard":

    # ── Header ───────────────────────────────
    st.markdown(
        """
        <p class='ek-section-title'>Analytics Overview</p>
        <h1>Emirates Airline<br>Sentiment Analysis</h1>
        <p style='color:#888;font-size:0.9rem;margin-top:0.4rem;margin-bottom:2rem;'>
            Machine-learning powered review intelligence dashboard
        </p>
        """,
        unsafe_allow_html=True,
    )

    # ── KPI Row ───────────────────────────────
    total = len(df)
    pos_pct = round(100 * (df["sentiment"] == "Positive").sum() / total, 1)
    neg_pct = round(100 * (df["sentiment"] == "Negative").sum() / total, 1)
    rating_col_name = get_col(df, "Overall Rating")
    avg_rating = round(df[rating_col_name].mean(), 2) if rating_col_name else "N/A"

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Reviews", f"{total:,}")
    k2.metric("Positive Sentiment", f"{pos_pct}%", delta=f"+{pos_pct}%")
    k3.metric("Negative Sentiment", f"{neg_pct}%", delta=f"-{neg_pct}%")
    k4.metric("Avg. Overall Rating", avg_rating)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── 1  Sentiment Distribution ─────────────
    st.markdown("<p class='ek-section-title'>01 — Sentiment Distribution</p>", unsafe_allow_html=True)
    sentiment_counts = df["sentiment"].value_counts()
    colors_bar = {"Positive": "#2ECC71", "Negative": "#D71921", "Neutral": "#C9963E"}
    fig1, ax1 = plt.subplots(figsize=(8, 3.5))
    apply_dark_style(fig1, ax1)
    bars = ax1.barh(
        sentiment_counts.index,
        sentiment_counts.values,
        color=[colors_bar.get(s, "#888") for s in sentiment_counts.index],
        height=0.55,
    )
    for bar in bars:
        w = bar.get_width()
        ax1.text(w + 8, bar.get_y() + bar.get_height() / 2,
                 f"{w:,}", va="center", color="#888", fontsize=9)
    ax1.set_xlabel("Number of Reviews")
    ax1.set_xlim(0, sentiment_counts.max() * 1.12)
    ax1.invert_yaxis()
    plt.tight_layout()
    st.pyplot(fig1)
    plt.close(fig1)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── 2  Geographic Distribution ────────────
    st.markdown("<p class='ek-section-title'>02 — Geographic Distribution</p>", unsafe_allow_html=True)
    country_col = get_col(df, "Country")
    if country_col:
        country_counts = df[country_col].value_counts().reset_index()
        country_counts.columns = ["Country", "Review_Count"]
        fig_map = px.choropleth(
            country_counts,
            locations="Country",
            locationmode="country names",
            color="Review_Count",
            hover_name="Country",
            color_continuous_scale=[[0, "#1E1E1E"], [0.3, "#8B1A1A"], [1, "#D71921"]],
            title="",
        )
        fig_map.update_geos(
            showcoastlines=True, coastlinecolor="#2A2A2A",
            showland=True, landcolor="#1E1E1E",
            showocean=True, oceancolor="#0D0D0D",
            showframe=False,
            bgcolor="#161616",
        )
        fig_map.update_layout(
            paper_bgcolor="#161616",
            plot_bgcolor="#161616",
            font_color="#EFEFEF",
            margin={"r": 0, "t": 10, "l": 0, "b": 0},
            coloraxis_colorbar=dict(
                thickness=10,
                title=dict(text="Reviews", font=dict(color="#888")),
                tickfont=dict(color="#888"),
            ),
        )
        st.plotly_chart(fig_map, use_container_width=True)
    else:
        st.info("Country column not found in dataset.")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── 3-5  Static Analysis Images ───────────
    img_data = {
        "03 — Avg. Rating by Travel Class": "avg_rating_by_travel_class.png",
        "04 — Avg. Rating by Travel Type":  "avg_rating_by_travel_type.png",
        "05 — Key Service Ratings":          "key_ratings_distribution.png",
    }
    for label, fname in img_data.items():
        st.markdown(f"<p class='ek-section-title'>{label}</p>", unsafe_allow_html=True)
        try:
            st.image(Image.open(fname), use_container_width=True)
        except Exception:
            st.info(f"Image not found: `{fname}`")
        st.markdown("<br>", unsafe_allow_html=True)

    # ── 6  Model Comparison ───────────────────
    st.markdown("<p class='ek-section-title'>06 — Model Comparison (Accuracy)</p>", unsafe_allow_html=True)
    model_results = pd.DataFrame({
        "Model": [
            "Logistic Regression\n(Baseline)",
            "Logistic Regression\n(Optimised)",
            "SVM",
        ],
        "Accuracy": [0.8445, 0.8515, 0.79],
    })
    fig_cmp, ax_cmp = plt.subplots(figsize=(9, 4))
    apply_dark_style(fig_cmp, ax_cmp)
    bar_colors = ["#555", "#D71921", "#C9963E"]
    bars_cmp = ax_cmp.bar(
        model_results["Model"], model_results["Accuracy"],
        color=bar_colors, width=0.5,
    )
    for p in bars_cmp:
        ax_cmp.text(
            p.get_x() + p.get_width() / 2, p.get_height() + 0.005,
            f"{p.get_height():.4f}", ha="center", color="#EFEFEF", fontsize=9,
        )
    ax_cmp.set_ylim(0.7, 0.92)
    ax_cmp.set_ylabel("Accuracy")
    ax_cmp.set_title("Classifier Accuracy Comparison", pad=10)
    plt.xticks(rotation=0, ha="center", fontsize=8.5)
    plt.tight_layout()
    st.pyplot(fig_cmp)
    plt.close(fig_cmp)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── 7  ROC Curve ──────────────────────────
    st.markdown("<p class='ek-section-title'>07 — ROC Curve (Optimised LR)</p>", unsafe_allow_html=True)
    try:
        st.image(Image.open("ROC.png"), use_container_width=True)
    except Exception:
        st.info("Image not found: `ROC.png`")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── 8  Word Clouds ────────────────────────
    st.markdown("<p class='ek-section-title'>08 — Word Clouds</p>", unsafe_allow_html=True)
    wc_col1, wc_col2 = st.columns(2)
    stop_words = set(STOPWORDS)

    if "Review" in df.columns:
        def make_wordcloud(texts, bg_color, colormap):
            joined = " ".join(texts)
            if not joined.strip():
                return None
            return WordCloud(
                stopwords=stop_words,
                background_color=bg_color,
                colormap=colormap,
                max_words=200,
                width=800,
                height=380,
                prefer_horizontal=0.9,
            ).generate(joined)

        with wc_col1:
            st.markdown(
                "<p style='color:#2ECC71;font-size:0.75rem;letter-spacing:0.1em;"
                "text-transform:uppercase;'>Positive Reviews</p>",
                unsafe_allow_html=True,
            )
            pos_texts = df.loc[df["sentiment"] == "Positive", "Review"].dropna().astype(str).tolist()
            wc_pos = make_wordcloud(pos_texts, "#1E1E1E", "Greens")
            if wc_pos:
                fig_wp, ax_wp = plt.subplots(figsize=(8, 4))
                fig_wp.patch.set_facecolor("#1E1E1E")
                ax_wp.imshow(wc_pos, interpolation="bilinear")
                ax_wp.axis("off")
                st.pyplot(fig_wp)
                plt.close(fig_wp)
            else:
                st.info("No positive reviews.")

        with wc_col2:
            st.markdown(
                "<p style='color:#D71921;font-size:0.75rem;letter-spacing:0.1em;"
                "text-transform:uppercase;'>Negative Reviews</p>",
                unsafe_allow_html=True,
            )
            neg_texts = df.loc[df["sentiment"] == "Negative", "Review"].dropna().astype(str).tolist()
            wc_neg = make_wordcloud(neg_texts, "#1E1E1E", "Reds")
            if wc_neg:
                fig_wn, ax_wn = plt.subplots(figsize=(8, 4))
                fig_wn.patch.set_facecolor("#1E1E1E")
                ax_wn.imshow(wc_neg, interpolation="bilinear")
                ax_wn.axis("off")
                st.pyplot(fig_wn)
                plt.close(fig_wn)
            else:
                st.info("No negative reviews.")
    else:
        st.info("Column 'Review' not found for word clouds.")


# ══════════════════════════════════════════════
# CHATBOT
# ══════════════════════════════════════════════
elif section == "Chatbot":

    st.markdown(
        """
        <p class='ek-section-title'>AI Review Assistant</p>
        <h1>Emirates Review<br>Chatbot</h1>
        <p style='color:#888;font-size:0.9rem;margin-top:0.4rem;margin-bottom:2rem;'>
            Ask anything about Emirates airline reviews — I'll surface relevant insights.
        </p>
        """,
        unsafe_allow_html=True,
    )

    if vectorizer is None or sentiment_model is None or review_vectors is None:
        st.warning("⚠️  Chatbot unavailable — model/vectorizer files not loaded.")
    else:
        with st.expander("💡  Example queries"):
            st.markdown(
                """
- **Small talk:** `hi`, `hello`, `bye`
- **Overview:** `tell me about Emirates`
- **Topics:** `food quality`, `cabin crew service`, `seat comfort`, `wifi`, `in-flight entertainment`
- **Filter by sentiment:** `negative comments on service`, `positive reviews about food`
                """
            )

        # Quick action chips
        chip1, chip2, chip3 = st.columns(3)
        with chip1:
            if st.button("🍽  Food"):
                st.session_state.last_query = "food quality"
        with chip2:
            if st.button("🧑‍✈️  Service"):
                st.session_state.last_query = "cabin crew service"
        with chip3:
            if st.button("💺  Seats"):
                st.session_state.last_query = "seat comfort"

        user_query = st.text_input(
            "Your query",
            value=st.session_state.last_query,
            placeholder="e.g. food quality, cabin crew, wifi ...",
        )

        if st.button("Get Insights", type="primary"):
            query_lower = (user_query or "").lower().strip()
            st.session_state.last_query = query_lower

            # ── Small talk ────────────────────────
            if query_lower in {"hi", "hello"}:
                st.success("👋  Hello! Welcome to the Emirates Review Chatbot. Ask about food, service, seats, routes and more.")
                st.stop()
            if query_lower in {"bye", "goodbye"}:
                st.success("👋  Goodbye! Safe travels.")
                st.stop()
            if "emirates" in query_lower and any(w in query_lower for w in ("about", "introduction", "intro")):
                st.markdown(
                    """
                    > ✈️ **Emirates** is the world's largest international airline by scheduled revenue
                    > passenger-kilometres flown, headquartered in Dubai, UAE.  
                    > Renowned for its A380 fleet, award-winning ICE in-flight entertainment,
                    > premium cabin products, and a global network spanning six continents.
                    """
                )
                st.stop()

            # ── Sentiment prediction ───────────────
            st.markdown("<hr>", unsafe_allow_html=True)
            try:
                q_vec = vectorizer.transform([query_lower])
                pred_label = sentiment_model.predict(q_vec)[0]
                confidence = float(np.max(sentiment_model.predict_proba(q_vec)) * 100)
                colour = "#2ECC71" if pred_label == "Positive" else "#D71921"
                st.markdown(
                    f"<p style='font-size:0.82rem;color:#888;letter-spacing:0.08em;"
                    f"text-transform:uppercase;margin-bottom:0.2rem;'>Predicted Sentiment</p>"
                    f"<p style='font-size:1.5rem;font-family:Cormorant Garamond,serif;"
                    f"color:{colour};margin:0;'>{pred_label}"
                    f"<span style='font-size:0.9rem;color:#888;margin-left:0.6rem;'>"
                    f"{confidence:.1f}% confidence</span></p>",
                    unsafe_allow_html=True,
                )
                if pred_label == "Positive" and top_positive_words:
                    st.markdown(
                        f"<p style='color:#888;font-size:0.8rem;'>"
                        f"<b style='color:#C9963E;'>Key Positive Words:</b> "
                        f"{', '.join(top_positive_words)}</p>",
                        unsafe_allow_html=True,
                    )
                elif pred_label == "Negative" and top_negative_words:
                    st.markdown(
                        f"<p style='color:#888;font-size:0.8rem;'>"
                        f"<b style='color:#D71921;'>Key Negative Words:</b> "
                        f"{', '.join(top_negative_words)}</p>",
                        unsafe_allow_html=True,
                    )
            except Exception as exc:
                st.error(f"Prediction error: {exc}")

            # ── Relevant reviews ───────────────────
            similarities = cosine_similarity(
                vectorizer.transform([query_lower]), review_vectors
            ).flatten()
            sorted_indices = similarities.argsort()[::-1]

            filtered: list[int] = []
            for idx in sorted_indices:
                rev_sent = df.loc[idx, "sentiment"]
                if "negative" in query_lower and rev_sent == "Negative":
                    filtered.append(idx)
                elif "positive" in query_lower and rev_sent == "Positive":
                    filtered.append(idx)
                elif "all reviews" in query_lower or (
                    "negative" not in query_lower and "positive" not in query_lower
                ):
                    filtered.append(idx)
                if len(filtered) >= 5:
                    break

            if not filtered:
                st.info("No relevant reviews found for this query.")
                st.session_state.results_df = None
            else:
                results_df = build_results_subset(df, filtered)
                st.session_state.results_df = results_df

                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown(
                    "<p class='ek-section-title'>Top Matching Reviews</p>",
                    unsafe_allow_html=True,
                )

                for i, idx in enumerate(filtered):
                    review_text = df.loc[idx, "Review"]
                    review_sentiment = df.loc[idx, "sentiment"]
                    rating = df.loc[idx, get_col(df, "Overall Rating")] if get_col(df, "Overall Rating") else "N/A"
                    date   = df.loc[idx, get_col(df, "Date Published")]  if get_col(df, "Date Published")  else "N/A"
                    country = df.loc[idx, get_col(df, "Country")]        if get_col(df, "Country")         else "N/A"
                    sim = similarities[idx]

                    badge_color = "#2ECC71" if review_sentiment == "Positive" else "#D71921" if review_sentiment == "Negative" else "#C9963E"
                    st.markdown(
                        f"""
                        <div style='border:1px solid #2A2A2A;border-radius:6px;
                                    padding:1rem 1.2rem;margin-bottom:0.8rem;
                                    background:#161616;'>
                          <div style='display:flex;align-items:center;gap:0.6rem;
                                      margin-bottom:0.5rem;flex-wrap:wrap;'>
                            <span style='background:{badge_color};color:#fff;font-size:0.68rem;
                                         font-weight:600;padding:2px 8px;border-radius:20px;
                                         letter-spacing:0.06em;text-transform:uppercase;'>
                              {review_sentiment}
                            </span>
                            <span style='color:#888;font-size:0.78rem;'>
                              ★ {rating}/10 &nbsp;·&nbsp; {country} &nbsp;·&nbsp; {date}
                            </span>
                            <span style='margin-left:auto;color:#555;font-size:0.72rem;'>
                              Similarity: {sim:.2f}
                            </span>
                          </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    if review_sentiment == "Positive":
                        highlighted = highlight_keywords(review_text, top_positive_words, "#1a3a1a")
                    elif review_sentiment == "Negative":
                        highlighted = highlight_keywords(review_text, top_negative_words, "#3a1a1a")
                    else:
                        highlighted = str(review_text)

                    st.markdown(
                        f"<p style='color:#BDBDBD;font-size:0.88rem;line-height:1.6;"
                        f"margin:0;'>{highlighted}</p></div>",
                        unsafe_allow_html=True,
                    )

        # ── Export ────────────────────────────
        if st.session_state.results_df is not None and not st.session_state.results_df.empty:
            st.markdown("<hr>", unsafe_allow_html=True)
            st.markdown("<p class='ek-section-title'>Export</p>", unsafe_allow_html=True)

            if st.button("Generate PDF Report"):
                pdf_bytes = create_pdf_from_df(
                    st.session_state.results_df, title="Emirates Review Insights"
                )
                st.session_state.pdf_bytes = pdf_bytes

            if st.session_state.pdf_bytes:
                st.download_button(
                    label="⬇  Download PDF",
                    data=st.session_state.pdf_bytes,
                    file_name="emirates_review_insights.pdf",
                    mime="application/pdf",
                )
