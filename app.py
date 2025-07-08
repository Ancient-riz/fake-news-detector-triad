import streamlit as st
import pickle
import time

# Load model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Page settings
st.set_page_config(page_title="📰 Fake News Detector", page_icon="🧠", layout="centered")

# ---- Custom Styling & Animations ----
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@600&display=swap');

    html, body, [class*="css"]  {
        font-family: 'Orbitron', sans-serif;
        background-color: #0e1117;
        color: #f1f1f1;
    }

    .title-animate {
        font-size: 36px;
        text-align: center;
        background: linear-gradient(90deg, #00ffe1, #0066ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: glow 2s ease-in-out infinite alternate;
        margin-bottom: 10px;
    }

    @keyframes glow {
        from {
            text-shadow: 0 0 10px #00ffe1;
        }
        to {
            text-shadow: 0 0 20px #0066ff;
        }
    }

    .news-box {
        border: 2px solid #00ffe1;
        border-radius: 15px;
        padding: 20px;
        background-color: #1c1f26;
        box-shadow: 0 0 15px rgba(0,255,225,0.2);
    }

    .stButton>button {
        background-color: #00ffe1;
        color: black;
        font-weight: bold;
        border-radius: 10px;
        height: 50px;
        width: 150px;
        transition: 0.3s ease-in-out;
    }

    .stButton>button:hover {
        background-color: #0066ff;
        color: white;
        transform: scale(1.05);
    }
    </style>
""", unsafe_allow_html=True)

# ---- Animated Header ----
st.markdown('<div class="title-animate">🧠 AI Fake News Detector</div>', unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 16px;'>Detect whether a news article is REAL or FAKE using Machine Learning.</p>", unsafe_allow_html=True)
st.markdown("---")

# ---- Input Box ----
with st.container():
    st.markdown('<div class="news-box">', unsafe_allow_html=True)
    news_text = st.text_area("Paste your news article here 👇", height=200, placeholder="Type or paste your article...")
    st.markdown('</div>', unsafe_allow_html=True)

# ---- Button ----
if st.button("🔍 Analyze"):
    if news_text.strip() == "":
        st.warning("⚠️ Please enter some news text.")
    else:
        with st.spinner("Analyzing with AI..."):
            time.sleep(1.2)  # simulating processing delay
            input_vector = vectorizer.transform([news_text])
            prediction = model.predict(input_vector)[0]

        if prediction == 1:
            st.error("❌ This news is likely **FAKE**.")
        else:
            st.success("✅ This news is likely **REAL**.")
