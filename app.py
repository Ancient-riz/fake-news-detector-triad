import streamlit as st
import pickle
import time

# Load model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Page config
st.set_page_config(page_title="🧠 Fake News Detector", page_icon="🧠", layout="centered")

# ---- Custom Styles ----
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500&display=swap');

    html, body, [class*="css"] {
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
        margin-bottom: 20px;
    }

    @keyframes glow {
        from { text-shadow: 0 0 10px #00ffe1; }
        to   { text-shadow: 0 0 20px #0066ff; }
    }

    .news-box {
        border: 2px solid #00ffe1;
        border-radius: 15px;
        padding: 20px;
        background-color: #1c1f26;
        box-shadow: 0 0 15px rgba(0,255,225,0.2);
        margin-bottom: 20px;
    }

    .stButton>button {
        background-color: #00ffe1;
        color: black;
        font-weight: bold;
        border-radius: 10px;
        height: 50px;
        width: 180px;
        transition: 0.3s ease-in-out;
        margin: 10px;
    }

    .stButton>button:hover {
        background-color: #0066ff;
        color: white;
        transform: scale(1.05);
    }

    footer, .css-164nlkn, .css-cio0dv {visibility: hidden;}  /* Remove Made with Streamlit */

    .gear {
        margin-top: 30px;
        text-align: center;
    }

    .gear img {
        width: 80px;
        animation: rotate 2s linear infinite;
    }

    @keyframes rotate {
        from {transform: rotate(0deg);}
        to   {transform: rotate(360deg);}
    }
    </style>
""", unsafe_allow_html=True)

# ---- Title ----
st.markdown('<div class="title-animate">🧠 AI Fake News Detector</div>', unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Detect whether a news article is REAL or FAKE using Machine Learning.</p>", unsafe_allow_html=True)
st.markdown("---")

# ---- Input Box ----
with st.container():
    st.markdown('<div class="news-box">', unsafe_allow_html=True)
    if 'news_text' not in st.session_state:
        st.session_state['news_text'] = ""
    news_text = st.text_area("Paste your news article here 👇", height=200, key="main_input", value=st.session_state['news_text'])
    st.markdown('</div>', unsafe_allow_html=True)

# ---- Button Row ----
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("🔍 Analyze"):
        if news_text.strip() == "":
            st.warning("⚠️ Please enter some news text.")
        else:
            with st.spinner("Analyzing with AI..."):
                time.sleep(1.2)
                input_vector = vectorizer.transform([news_text])
                prediction = model.predict(input_vector)[0]
            if prediction == 1:
                st.error("❌ This news is likely FAKE.")
            else:
                st.success("✅ This news is likely REAL.")
            st.session_state['news_text'] = news_text

with col2:
    if st.button("🧪 Example"):
        st.session_state['news_text'] = "The Prime Minister announced a new health policy which will be implemented next month."

with col3:
    if st.button("❌ Clear"):
        st.session_state['news_text'] = ""

# ---- Gear Animation Footer ----
st.markdown("""
<div class="gear">
    <img src="https://cdn-icons-png.flaticon.com/512/189/189792.png" alt="gear spinning"/>
</div>
""", unsafe_allow_html=True)
