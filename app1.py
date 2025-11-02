# app1.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import joblib
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud, STOPWORDS
import textwrap
# app1.py (Start of file)
import streamlit as st


# -----------------------------
# Page config
st.set_page_config(page_title="Emirates Review Sentiment App", layout="wide")



# -----------------------------
# Session state (persist UI state across reruns)
# -----------------------------
if "results_df" not in st.session_state:
    st.session_state.results_df = None
if "last_query" not in st.session_state:
    st.session_state.last_query = ""
if "pdf_bytes" not in st.session_state:
    st.session_state.pdf_bytes = None

# -----------------------------
# Load dataset
# -----------------------------
df = pd.read_csv("emirates_reviews.csv")

# Create 'sentiment' column from 'Recommended'
df["Recommended"] = df["Recommended"].astype(str).str.strip().str.lower()
df["sentiment"] = df["Recommended"].map({"yes": "Positive", "no": "Negative"})
df["sentiment"] = df["sentiment"].fillna("Neutral")

# -----------------------------
# Load saved model and vectorizer (prefer pipeline)
# -----------------------------
vectorizer = None
sentiment_model = None
pipeline_loaded = False
try:
    # Prefer the pipeline which bundles tfidf + classifier
    pipeline = joblib.load('pipeline_tfidf_logistic_binary.pkl')
    if hasattr(pipeline, 'named_steps') and 'tfidf' in pipeline.named_steps and 'clf' in pipeline.named_steps:
        vectorizer = pipeline.named_steps['tfidf']
        sentiment_model = pipeline.named_steps['clf']
        pipeline_loaded = True
        st.sidebar.success('✅ Pipeline loaded (tfidf + classifier).')
    else:
        st.sidebar.warning('⚠️ Pipeline file loaded but steps not found; falling back to separate files.')
        pipeline = None
except FileNotFoundError:
    pipeline = None
except Exception as e:
    pipeline = None
    st.sidebar.warning(f'⚠️ Could not load pipeline (will try separate files): {e}')

if not pipeline_loaded:
    # Fallback to separate artifacts
    try:
        vectorizer = joblib.load('tfidf_vectorizer.pkl')
        sentiment_model = joblib.load('logistic_model_binary.pkl')
        st.sidebar.success('✅ Vectorizer and model loaded.')
    except FileNotFoundError:
        st.sidebar.error('❌ Missing model/vectorizer files. Chatbot functionality will be limited.')
        vectorizer = None
        sentiment_model = None
    except Exception as e:
        st.sidebar.error(f'❌ Error loading artifacts: {e}')
        vectorizer = None
        sentiment_model = None

# Vectorize all reviews for similarity search (safe for sparse matrices)
review_texts = df['Review'].fillna('').astype(str).tolist()
if vectorizer is not None:
    try:
        review_vectors = vectorizer.transform(review_texts)
    except Exception as e:
        st.sidebar.error(f'Error vectorizing reviews: {e}')
        review_vectors = None
else:
    review_vectors = None

# -----------------------------
# Helpers
# -----------------------------
def get_top_features_per_class(vectorizer, model, n_top_features=10):
    feature_names = vectorizer.get_feature_names_out()
    if len(model.classes_) != 2:
        return [], []
    # Assume model.classes_ sorted alphabetically => ['Negative','Positive']
    coef = model.coef_[0]
    top_pos_idx = np.argsort(coef)[-n_top_features:]
    top_neg_idx = np.argsort(coef)[:n_top_features]
    pos_words = [feature_names[i] for i in top_pos_idx]
    neg_words = [feature_names[i] for i in top_neg_idx]
    return pos_words, neg_words

top_positive_words, top_negative_words = ([], [])
if sentiment_model is not None and vectorizer is not None:
    top_positive_words, top_negative_words = get_top_features_per_class(vectorizer, sentiment_model, 12)

def highlight_keywords(text, keywords, color="yellow"):
    safe_text = str(text)
    for kw in sorted(set(keywords), key=len, reverse=True):
        if kw.strip():
            safe_text = safe_text.replace(
                kw, f"<mark style='background-color:{color};'>{kw}</mark>"
            )
    return safe_text

def get_col(df, name):
    """Case-insensitive safe getter for a column; returns None if not found."""
    cols = {c.lower(): c for c in df.columns}
    return cols.get(name.lower())

def build_results_subset(df, indices):
    cols_needed = ["Review", "sentiment", "Overall Rating", "Date Published", "Country"]
    safe_cols = []
    for c in cols_needed:
        real = get_col(df, c)
        if real is not None:
            safe_cols.append(real)
    out = df.iloc[indices][safe_cols].copy()
    # Standardize column labels in the subset for downstream use
    rename_map = {get_col(df, "Review"): "Review",
                  get_col(df, "sentiment"): "Sentiment",
                  get_col(df, "Overall Rating"): "Overall Rating",
                  get_col(df, "Date Published"): "Date Published",
                  get_col(df, "Country"): "Country"}
    out.rename(columns=rename_map, inplace=True)
    return out

def create_pdf_from_df(results_df, title="Emirates Review Insights"):
    """Create a single PDF (bytes) with wrapped lines; tolerant to missing columns."""
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
    from reportlab.lib.units import inch
    from io import BytesIO

    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    margin_left = 0.75 * inch
    margin_right = 0.75 * inch
    margin_top = 0.75 * inch
    margin_bottom = 0.75 * inch
    max_width = width - margin_left - margin_right
    y = height - margin_top

    c.setFont("Helvetica-Bold", 14)
    c.drawString(margin_left, y, title)
    y -= 20

    c.setFont("Helvetica", 10)

    def draw_wrapped(text, y_pos, line_height=14):
        """Draw wrapped text and return new y position; page-break if needed."""
        wrapper = textwrap.TextWrapper(width=95)  # tune if to change font/size
        for line in wrapper.wrap(text):
            if y_pos < margin_bottom + 20:  # new page
                c.showPage()
                c.setFont("Helvetica", 10)
                y_pos = height - margin_top
            c.drawString(margin_left, y_pos, line)
            y_pos -= line_height
        return y_pos

    if results_df is None or results_df.empty:
        y = draw_wrapped("No results to export.", y)
    else:
        for i, row in results_df.iterrows():
            review = str(row.get("Review", "N/A")).strip()
            sentiment = row.get("Sentiment", "N/A")
            rating = row.get("Overall Rating", "N/A")
            date = row.get("Date Published", "N/A")
            country = row.get("Country", "N/A")

            header = f"{i+1}. Sentiment: {sentiment} | Rating: {rating} | Date: {date} | Country: {country}"
            y = draw_wrapped(header, y)
            y = draw_wrapped(f"Review: {review}", y)
            y -= 6  # small gap between reviews

    c.save()
    buffer.seek(0)
    return buffer.read()

# -----------------------------
# Sidebar navigation
# -----------------------------
st.sidebar.title("Navigation")
section = st.sidebar.radio("Go to", ["Dashboard", "Chatbot"])

# =========================================================
# DASHBOARD
# =========================================================
if section == "Dashboard":
    st.title("📊 Dashboard – Emirates Airline Sentiment Analysis")

    # 1) Sentiment Distribution
    st.subheader("1. Sentiment Distribution")
    sentiment_counts = df["sentiment"].value_counts()
    fig1, ax1 = plt.subplots(figsize=(8, 4))
    sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette="Set2", ax=ax1)
    ax1.set_ylabel("Number of Reviews")
    ax1.set_xlabel("Sentiment")
    st.pyplot(fig1)
    plt.close(fig1)

    # 2) Geographic Distribution (Interactive)
    st.subheader("2. Geographic Distribution of Customer Reviews (Interactive)")
    if get_col(df, "Country"):
        country_col = get_col(df, "Country")
        country_counts = df[country_col].value_counts().reset_index()
        country_counts.columns = ["Country", "Review_Count"]

        fig_map = px.choropleth(
            country_counts,
            locations="Country",
            locationmode="country names",
            color="Review_Count",
            hover_name="Country",
            hover_data={"Review_Count": True},
            color_continuous_scale=px.colors.sequential.Plasma,
            title="Reviews by Country"
        )
        fig_map.update_geos(
            showcoastlines=True, coastlinecolor="Black",
            showland=True, landcolor="lightgray",
            showocean=True, oceancolor="lightblue"
        )
        fig_map.update_layout(margin={"r":0,"t":50,"l":0,"b":0})
        st.plotly_chart(fig_map, use_container_width=True)

    # 3) Average Overall Rating by Travel Class (static image if present)
    st.subheader("3. Average Overall Rating by Travel Class")
    try:
        st.image(Image.open("avg_rating_by_travel_class.png"), use_container_width=True)
    except Exception:
        st.info("Image not found: avg_rating_by_travel_class.png")

    # 4) Average Overall Rating by Travel Type (static image if present)
    st.subheader("4. Average Overall Rating by Travel Type")
    try:
        st.image(Image.open("avg_rating_by_travel_type.png"), use_container_width=True)
    except Exception:
        st.info("Image not found: avg_rating_by_travel_type.png")

    # 5) Distribution of Key Service Ratings (static image if present)
    st.subheader("5. Distribution of Key Service Ratings")
    try:
        st.image(Image.open("key_ratings_distribution.png"), use_container_width=True)
    except Exception:
        st.info("Image not found: key_ratings_distribution.png")

    # 6) Model Comparison (bar chart)
    st.subheader("6. Model Comparison (Accuracy)")
    # --- UPDATE: Values are corrected to match validated dissertation results (85.15% Optimized) ---
    model_results = pd.DataFrame({
        "Model": ["Logistic Regression (Baseline)", "Logistic Regression (Optimized)", "SVM"],
        # Baseline model score was around 0.8445, Optimized is 0.8515
        "Accuracy": [0.8445, 0.8515, 0.79] 
    })
    fig_cmp, ax_cmp = plt.subplots(figsize=(8, 4))
    sns.barplot(data=model_results, x="Model", y="Accuracy", palette="viridis", ax=ax_cmp)
    ax_cmp.set_ylim(0, 1)
    ax_cmp.set_ylabel("Accuracy")
    for p in ax_cmp.patches:
        ax_cmp.annotate(f"{p.get_height():.4f}", (p.get_x()+p.get_width()/2, p.get_height()),
                        ha='center', va='bottom', fontsize=9)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    st.pyplot(fig_cmp)
    plt.close(fig_cmp)

    # 7) ROC Curve (image)
    st.subheader("7. ROC Curve (Logistic Regression Optimized)")
    try:
        st.image(Image.open("ROC.png"), use_container_width=True)
    except Exception:
        st.info("Image not found: ROC.png")

    # 8) Word Clouds
    st.subheader("8. Word Clouds (Positive vs Negative Reviews)")
    wc_col1, wc_col2 = st.columns(2)
    stop_words = set(STOPWORDS)

    if "Review" in df.columns:
        # Positive
        with wc_col1:
            pos_text = " ".join(
                df.loc[df["sentiment"] == "Positive", "Review"].dropna().astype(str)
            )
            if pos_text.strip():
                wc_pos = WordCloud(stopwords=stop_words, background_color="white",
                                   max_words=200, width=800, height=400).generate(pos_text)
                fig_wc_pos, ax_wc_pos = plt.subplots(figsize=(8, 4))
                ax_wc_pos.imshow(wc_pos, interpolation="bilinear")
                ax_wc_pos.axis("off")
                ax_wc_pos.set_title("Positive Reviews")
                st.pyplot(fig_wc_pos)
                plt.close(fig_wc_pos)
            else:
                st.info("No positive reviews to generate a word cloud.")

        # Negative
        with wc_col2:
            neg_text = " ".join(
                df.loc[df["sentiment"] == "Negative", "Review"].dropna().astype(str)
            )
            if neg_text.strip():
                wc_neg = WordCloud(stopwords=stop_words, background_color="white",
                                   max_words=200, width=800, height=400).generate(neg_text)
                fig_wc_neg, ax_wc_neg = plt.subplots(figsize=(8, 4))
                ax_wc_neg.imshow(wc_neg, interpolation="bilinear")
                ax_wc_neg.axis("off")
                ax_wc_neg.set_title("Negative Reviews")
                st.pyplot(fig_wc_neg)
                plt.close(fig_wc_neg)
            else:
                st.info("No negative reviews to generate a word cloud.")
    else:
        st.info("Column 'Review' not found to create word clouds.")

# =========================================================
# CHATBOT
# =========================================================
elif section == "Chatbot":
    st.title("🤖 Emirates Review Chatbot")
    st.write("Ask anything about Emirates airline reviews. I’ll find relevant insights.")

    if vectorizer is None or sentiment_model is None or review_vectors is None:
        st.warning("❌ Chatbot not working because model/vectorizer didn't load.")
    else:
        # Quick intro + small talk examples
        with st.expander("💡 What can I ask?"):
            st.markdown("""
- **Small talk:** "hi", "hello", "bye", "goodbye"  
- **Intro:** "tell me about Emirates", "introduction to Emirates"  
- **Queries:** "food quality", "negative comments on service", "seats comfort", "wifi", "IFE", etc.
""")

        # Quick action buttons to append helpful keywords
        btn_col1, btn_col2, btn_col3 = st.columns(3)
        with btn_col1:
            if st.button("🍽️ Food"):
                st.session_state.last_query = "food quality"
        with btn_col2:
            if st.button("🧑‍✈️ Service"):
                st.session_state.last_query = "cabin crew service"
        with btn_col3:
            if st.button("💺 Seats"):
                st.session_state.last_query = "seat comfort"

        user_query = st.text_input("✍️ Enter your query:", value=st.session_state.last_query)

        # Process query
        if st.button("Get Insights"):
            query_lower = (user_query or "").lower().strip()
            st.session_state.last_query = query_lower

            # Small talk & intro
            if query_lower in ["hi", "hello"]:
                st.write("👋 Hello! Welcome to the Emirates Review Chatbot. Ask about food, service, seats, routes, and more.")
                st.stop()
            if query_lower in ["bye", "goodbye"]:
                st.write("👋 Goodbye! Safe travels.")
                st.stop()
            if ("emirates" in query_lower) and ("about" in query_lower or "introduction" in query_lower or "intro" in query_lower):
                st.markdown("✈️ **Emirates** is the largest airline in the Middle East, based in Dubai, UAE. It operates across six continents and is known for premium cabins, extensive ICE in-flight entertainment, and a large widebody fleet (A380s & 777s).")
                st.stop()

            # Sentiment Prediction for the query text
            try:
                q_vec = vectorizer.transform([query_lower])
                prediction_label = sentiment_model.predict(q_vec)[0]
                confidence = float(np.max(sentiment_model.predict_proba(q_vec)) * 100.0)
                st.write(f"**Predicted Sentiment:** {prediction_label} (Confidence: {confidence:.2f}%)")
                if prediction_label == "Positive" and top_positive_words:
                    st.markdown(f"**Influential Positive Words:** {', '.join(top_positive_words)}")
                elif prediction_label == "Negative" and top_negative_words:
                    st.markdown(f"**Influential Negative Words:** {', '.join(top_negative_words)}")
            except Exception as e:
                st.error(f"Prediction error: {e}")

            # Retrieve relevant reviews (filter if user asked for positive/negative explicitly)
            similarities = cosine_similarity(vectorizer.transform([query_lower]), review_vectors).flatten()
            sorted_indices = similarities.argsort()[::-1]

            filtered_indices = []
            for idx in sorted_indices:
                review_sent = df.loc[idx, "sentiment"]
                if "negative" in query_lower and review_sent == "Negative":
                    filtered_indices.append(idx)
                elif "positive" in query_lower and review_sent == "Positive":
                    filtered_indices.append(idx)
                elif "all reviews" in query_lower or ("negative" not in query_lower and "positive" not in query_lower):
                    filtered_indices.append(idx)
                if len(filtered_indices) >= 5:
                    break

            if not filtered_indices:
                st.info("No relevant reviews found.")
                st.session_state.results_df = None
            else:
                # Build tidy results table for display + PDF export
                results_df = build_results_subset(df, filtered_indices)
                st.session_state.results_df = results_df

                for i, idx in enumerate(filtered_indices):
                    review_text = df.loc[idx, "Review"]
                    review_sentiment = df.loc[idx, "sentiment"]
                    rating_col = get_col(df, "Overall Rating")
                    date_col = get_col(df, "Date Published")
                    country_col = get_col(df, "Country")

                    rating = df.loc[idx, rating_col] if rating_col else "N/A"
                    date = df.loc[idx, date_col] if date_col else "N/A"
                    country = df.loc[idx, country_col] if country_col else "N/A"
                    sim = similarities[idx]

                    st.markdown(
                        f"**Review {i+1}** (Sentiment: `{review_sentiment}`, Rating: `{rating}/10`, "
                        f"Date: `{date}`, Country: `{country}`, Similarity: `{sim:.2f}`)"
                    )
                    if review_sentiment == "Positive":
                        highlighted = highlight_keywords(review_text, top_positive_words, "lightgreen")
                    elif review_sentiment == "Negative":
                        highlighted = highlight_keywords(review_text, top_negative_words, "salmon")
                    else:
                        highlighted = review_text
                    st.markdown(f"\"{highlighted}\"", unsafe_allow_html=True)
                    st.markdown("---")

        # Show export button after results exist
        if st.session_state.results_df is not None and not st.session_state.results_df.empty:
            st.markdown("### 📄 Export")
            if st.button("Generate PDF from results"):
                pdf_bytes = create_pdf_from_df(st.session_state.results_df, title="Emirates Review Insights")
                st.session_state.pdf_bytes = pdf_bytes

            if st.session_state.pdf_bytes:
                st.download_button(
                    label="⬇️ Download PDF",
                    data=st.session_state.pdf_bytes,
                    file_name="emirates_review_insights.pdf",
                    mime="application/pdf"

                )
