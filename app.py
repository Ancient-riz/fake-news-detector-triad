import streamlit as st
import pickle

# Load model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Page config
st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")

# Header
st.markdown("<h1 style='text-align: center; color: #333;'>📰 Fake News Detector</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 18px;'>Enter a news article to check whether it's real or fake.</p>", unsafe_allow_html=True)
st.markdown("---")

# Text input area
with st.container():
    news_text = st.text_area("Paste a news article here:", height=250, help="Example: The government has announced a new policy...")

    if st.button("🔍 Analyze"):
        if news_text.strip() == "":
            st.warning("⚠️ Please enter some news text.")
        else:
            input_vector = vectorizer.transform([news_text])
            prediction = model.predict(input_vector)[0]

            if prediction == 1:
                st.error("❌ This news is likely **FAKE**.")
            else:
                st.success("✅ This news is likely **REAL**.")

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center; font-size: 13px;'>Made with ❤️ using Streamlit | <a href='https://github.com/Ancient-riz/fake-news-triad' target='_blank'>GitHub</a></p>",
    unsafe_allow_html=True,
)
