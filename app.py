import streamlit as st
import pickle

model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

st.title("🎬 Sentiment Analyzer")
text = st.text_area("Enter your review:")

if st.button("Analyze"):
    cleaned = text.lower()
    vectorized = vectorizer.transform([cleaned])
    result = model.predict(vectorized)[0]
    
    if result == 1:
        st.success("✅ Positive Sentiment")
    else:
        st.error("❌ Negative Sentiment")
    st.write(f"Confidence: {confidence}%") 
