import streamlit as st
import pickle

model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

def preprocess(text):
    return text.lower()

def predict_sentiment(text):
    cleaned = preprocess(text)
    vectorized = vectorizer.transform([cleaned])
    result = model.predict(vectorized)[0]
    prob = model.predict_proba(vectorized)[0]
    confidence = round(max(prob) * 100, 2)
    label = "Positive" if result == 1 else "Negative"
    return label, confidence

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

st.title("🎬 Sentiment Analyzer")
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
