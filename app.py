import streamlit as st
import pickle
from random import choice
from streamlit_lottie import st_lottie
import requests

# Load model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Load complex gear animation

def load_lottie_url(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

gear_animation = load_lottie_url("https://assets3.lottiefiles.com/packages/lf20_svy4ivvy.json")  # updated complex gears

st.set_page_config(page_title="Fake News Detector", page_icon="🧠", layout="wide")

st.markdown("""
    <style>
        .main {
            background-color: #0f1117;
            color: white;
        }
        .stTextArea textarea {
            background-color: #1e1e2f;
            color: white;
        }
        .css-1q8dd3e, .st-emotion-cache-j5r0tf {  /* hide Made with Streamlit */
            display: none !important;
        }
        footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

st.title("📰 AI-powered Fake News Detector")
st.write("Uncover misinformation with the power of Machine Learning")

# Display animated gears
st_lottie(gear_animation, height=220, key="gear")

# Random examples
examples = [
    "The government has launched a new scheme for free education.",
    "Aliens were spotted near the White House last night!",
    "COVID-19 vaccines cause people to become magnetic.",
    "NASA confirms the moon is shrinking.",
    "Chocolate is found to cure all types of cancer.",
    "A new smartphone app can detect lies using AI.",
    "Scientists discover water on Mars again.",
    "The Earth will become uninhabitable in 5 years, say sources.",
    "Eating pineapple cures the common cold."
]

col1, col2 = st.columns([3, 1])
with col1:
    news_text = st.text_area("Paste a news article or click for a random example:", height=200, key="text_input")
with col2:
    if st.button("🎲 Generate Example"):
        st.session_state.text_input = choice(examples)
        st.experimental_rerun()

if st.button("🔍 Analyze News"):
    if news_text.strip() == "":
        st.warning("⚠️ Please enter some news text.")
    else:
        input_vector = vectorizer.transform([news_text])
        prediction = model.predict(input_vector)[0]
        if prediction == 1:
            st.error("❌ This news is likely FAKE.")
        else:
            st.success("✅ This news is likely REAL.")
