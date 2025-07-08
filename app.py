# fake-news-triad
import streamlit as st
import pickle

# Load model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

st.title("📰 Fake News Detector")

news_text = st.text_area("Paste a news article below:")

if st.button("Check"):
    if news_text.strip() == "":
        st.warning("Please enter some news text.")
    else:
        input_vector = vectorizer.transform([news_text])
        prediction = model.predict(input_vector)[0]
        if prediction == 1:
            st.error("❌ This news is likely FAKE.")
        else:
            st.success("✅ This news is likely REAL.")
