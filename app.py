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
        margin-bottom: 20px;
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

    .footer {
        text-align: center;
        font-size: 12px;
        color: #888;
        padding-top: 20px;
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
    news_text = st.text_area("Paste your news article here 👇", height=200, placeholder="Type or paste your article...")
    st.markdown('</div>', unsafe_allow_html=True)

# ---- Button Row ----
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("🔍 Analyze"):
        if news_text.strip() == "":
            st.warning("⚠️ Please enter some news text.")
        else:
            with st.spinner("Analyzing..."):
                time.sleep(1.2)
                input_vector = vectorizer.transform([news_text])
                prediction = model.predict(input_vector)[0]
            if prediction == 1:
                st.error("❌ This news is likely FAKE.")
            else:
                st.success("✅ This news is likely REAL.")

with col2:
    if st.button("🧪 Example"):
        st.session_state['example'] = "The Prime Minister announced a new health policy which will be implemented next month."
        st.experimental_rerun()

with col3:
    if st.button("❌ Clear"):
        st.session_state['example'] = ""
        st.experimental_rerun()

# ---- Handle Example Text ----
if 'example' in st.session_state:
    news_text = st.session_state['example']
    st.text_area("🧾 Example loaded:", news_text, height=200, disabled=True)

# ---- Additional Animation Fill ----
st.markdown("""
<div style='margin-top:40px; text-align:center;'>
    <img src='https://media4.giphy.com/media/5GoVLqeAOo6PK/giphy.gif?cid=ecf05e47an2oblgto43wvpb2l0sxyli6j5f3rvyy0d6ypzev&ep=v1_gifs_search&rid=giphy.gif&ct=g' width='180'/>
    <p style='font-size:13px;color:#aaa;'>Powered by Logistic Regression & TF-IDF • Real-time AI Detection</p>
</div>
""", unsafe_allow_html=True)
