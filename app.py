import streamlit as st
import pickle
import pandas as pd
import re
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from difflib import get_close_matches

nltk.download('vader_lexicon')

# ─── LOAD MODELS ─────────────────────────────────────────────
model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

# ─── LOAD DATA ───────────────────────────────────────────────
@st.cache_data
def load_movie_list():
    url = "https://raw.githubusercontent.com/nitishghosal/IMDB-Data-Analysis/master/movie_metadata.csv"
    df = pd.read_csv(url)
    return set(df['movie_title'].str.lower().str.strip().tolist())

@st.cache_data
def load_reviews():
    df = pd.read_csv("reviews_small.csv")
    return df

# ─── ML LOGIC ────────────────────────────────────────────────
def normalize(text):
    return text.strip().lower()

def validate_movie(movie_input, movie_list):
    normalized = normalize(movie_input)
    if normalized in movie_list:
        return True, None
    close = get_close_matches(normalized, movie_list, n=1, cutoff=0.6)
    suggestion = close[0].title() if close else None
    return False, suggestion

def preprocess(text):
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r"\bnot\s+(\w+)", r"not_\1", text)
    text = re.sub(r"\bnever\s+(\w+)", r"never_\1", text)
    text = re.sub(r"\bno\s+(\w+)", r"no_\1", text)
    return text

def predict_sentiment(text):
    # ── ML part: TF-IDF + Logistic Regression ──
    cleaned = preprocess(text)
    vectorized = vectorizer.transform([cleaned])
    lr_prob = model.predict_proba(vectorized)[0]
    lr_score = lr_prob[1] - lr_prob[0]  # positive - negative

    # ── Rule based part: VADER (handles negations) ──
    sia = SentimentIntensityAnalyzer()
    vader_score = sia.polarity_scores(text)['compound']

    # ── Combine both (average) ──
    final_score = (lr_score + vader_score) / 2

    label = "Positive" if final_score >= 0 else "Negative"
    confidence = round(abs(final_score) * 100, 2)
    confidence = min(confidence, 99.99)  # cap at 99.99%

    return label, confidence

def get_top_reviews(df, n=5):
    sample = df.sample(n=n)
    reviews = []
    sia = SentimentIntensityAnalyzer()
    for _, row in sample.iterrows():
        text = row['review']
        short = text[:200] + "..." if len(text) > 200 else text
        score = sia.polarity_scores(text)['compound']
        sentiment = "Positive" if score >= 0.05 else "Negative"
        reviews.append({"text": short, "sentiment": sentiment})
    return reviews

def get_confidence_style(confidence):
    if confidence > 80:
        return "#2ecc71", "🔥"
    elif confidence > 60:
        return "#a8e063", "🤔"
    else:
        return "#f39c12", "😕"

def build_confidence_bar(confidence, color, emoji):
    return f"""
        <div style="background:#e0e0e0;border-radius:20px;height:40px;width:100%;position:relative;margin-top:15px;">
            <div style="background:{color};width:{confidence}%;height:100%;border-radius:20px;"></div>
            <span style="position:absolute;left:{confidence/2}%;top:50%;transform:translate(-50%,-50%);font-size:20px;">{emoji}</span>
        </div>
        <p style="text-align:center;margin-top:8px;">Confidence: <b>{confidence}%</b></p>
    """

def build_review_card(review):
    color = "#1a472a" if review['sentiment'] == "Positive" else "#7b1a1a"
    badge_color = "#2ecc71" if review['sentiment'] == "Positive" else "#e74c3c"
    emoji = "✅" if review['sentiment'] == "Positive" else "❌"
    return f"""
        <div style="
            background:{color};
            border-radius:12px;
            padding:15px 20px;
            margin-bottom:12px;
            border-left:4px solid {badge_color};
        ">
            <span style="
                background:{badge_color};
                color:white;
                padding:3px 10px;
                border-radius:20px;
                font-size:12px;
                font-weight:bold;
            ">{emoji} {review['sentiment']}</span>
            <p style="color:#f0f0f0;margin-top:10px;font-size:14px;line-height:1.6;">
                "{review['text']}"
            </p>
        </div>
    """

# ─── STREAMLIT UI ────────────────────────────────────────────
st.title("🎬 Sentiment Analyzer")

movie_list = load_movie_list()
df_reviews = load_reviews()

movie_input = st.text_input("🎥 Enter Movie Name:")

if movie_input:
    is_valid, suggestion = validate_movie(movie_input, movie_list)

    if is_valid:
        st.markdown("✅ Valid Movie!")
        text = st.text_area("Enter your review:")

        if st.button("Analyze"):
            label, confidence = predict_sentiment(text)
            color, emoji = get_confidence_style(confidence)
            bar_html = build_confidence_bar(confidence, color, emoji)

            if label == "Positive":
                st.success("✅ Positive Sentiment")
            else:
                st.error("❌ Negative Sentiment")

            st.markdown(bar_html, unsafe_allow_html=True)

        # ─── REVIEWS SECTION ─────────────────────────────────
        st.markdown("---")
        st.markdown("### 💬 Sample Reviews from Database")
        st.caption("*These are random reviews from our IMDB database*")

        top_reviews = get_top_reviews(df_reviews)
        cards_html = "".join([build_review_card(r) for r in top_reviews])
        st.markdown(cards_html, unsafe_allow_html=True)

    else:
        st.markdown("❌ Movie not found!")
        if suggestion:
            st.markdown(f"💡 Did you mean? **{suggestion}**")
