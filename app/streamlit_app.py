import streamlit as st
import numpy as np
import pickle

with open('../models/logistic_model.pkl', 'rb') as f:
    model = pickle.load(f)

w = model['weights']
b = model['bias']
scaler = model['scaler']
feature_names = model['feature_names']

SPAM_KEYWORDS = ['free', 'win', 'winner', 'prize', 'claim',
                 'urgent', 'cash', 'offer', 'call', 'text',
                 'mobile', 'won', 'awarded', 'guranteed']

def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

def extract_features(text):
    words = text.split()
    letters = [c for c in text if c.isalpha()]
    return np.array([[
        len(text),
        len(words),
        sum(c.isdigit() for c in text),
        sum(c.isupper() for c in letters) / max(len(letters) ,1),
        sum(c in '!$?' for c in text),
        sum(word.lower() in SPAM_KEYWORDS for word in words)
    ]])

def predict(text):
    features = extract_features(text)
    features_scaled = scaler.transform(features)
    z = features_scaled @ w + b
    prob = sigmoid(z)
    label = 'spam' if prob >= 0.5 else 'legit'
    confidence = prob if label == 'spam' else 1 - prob
    return label, round(float(confidence.item()) * 100, 2), features[0]

st.set_page_config(page_title="Spam Classifier", page_icon=None, layout="centered")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&display=swap');

[data-testid="stAppViewContainer"] { background-color: #1E1E1E; }
[data-testid="stHeader"] { background-color: #1E1E1E; }
section[data-testid="stMain"] > div { padding-top: 2rem; }
[data-testid="stDecoration"] { display: none; }
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
* { font-family: 'JetBrains Mono', monospace !important; }

textarea {
    background-color: #2D2D2D !important;
    border: 1px solid #454545 !important;
    border-radius: 3px !important;
    color: #D4D4D4 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.85rem !important;
}
textarea:focus { border-color: #569CD6 !important; }

.stButton > button {
    background-color: #0E639C !important;
    color: #FFFFFF !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-weight: 500 !important;
    border: none !important;
    border-radius: 3px !important;
    width: 100% !important;
    padding: 0.55rem !important;
    font-size: 0.85rem !important;
}
.stButton > button:hover { background-color: #1177BB !important; }

.stat-card {
    background-color: #252526;
    border: 1px solid #3C3C3C;
    border-radius: 3px;
    padding: 1rem 1.25rem;
}
.stat-label {
    color: #858585;
    font-size: 0.68rem;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin: 0 0 0.3rem 0;
}
.stat-value {
    font-size: 1.3rem;
    font-weight: 600;
    margin: 0;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div style="border-bottom:1px solid #3C3C3C; padding-bottom:1rem; margin-bottom:1.5rem;">
    <p style="font-size:0.68rem; color:#569CD6; letter-spacing:2px;
              text-transform:uppercase; margin:0 0 0.3rem 0;">
        SPAM CLASSIFIER
    </p>
    <h1 style="font-size:1.5rem; font-weight:600; color:#D4D4D4; margin:0 0 0.25rem 0;">
        SMS Spam Detector
    </h1>
    <p style="color:#6A6A6A; font-size:0.75rem; margin:0;">
        Logistic Regression from Scratch &nbsp;&mdash;&nbsp; Gradient Descent &nbsp;&mdash;&nbsp; 96.14% accuracy
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<p style="color:#4EC9B0; font-size:0.75rem; margin-bottom:0.4rem;">
&gt; enter_message
</p>
""", unsafe_allow_html=True)

message = st.text_area("message", placeholder="Type or paste an SMS message here...",
                        height=120, label_visibility="collapsed")
analyse = st.button("Classify Message")

st.markdown("""
<p style="color:#454545; font-size:0.75rem; margin-top:0.5rem; font-style:italic;">
// try: &nbsp;
<span style="color:#CE9178; font-style:normal;">
"Congratulations! You've won a FREE prize. Call 09061743386 to claim now!"
</span>
</p>
""", unsafe_allow_html=True)

if not analyse or not message.strip():
    st.markdown('<hr style="border:none;border-top:1px solid #3C3C3C;margin:1.5rem 0;">',
                unsafe_allow_html=True)
    st.markdown('<p style="color:#858585;font-size:0.68rem;letter-spacing:1.5px;text-transform:uppercase;margin:0 0 0.6rem 0;">Model Stats</p>',
                unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""<div class="stat-card">
            <p class="stat-label">Accuracy</p>
            <p class="stat-value" style="color:#4EC9B0;">96.14%</p>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""<div class="stat-card">
            <p class="stat-label">Dataset</p>
            <p class="stat-value" style="color:#569CD6;">5,572</p>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown("""<div class="stat-card">
            <p class="stat-label">Features</p>
            <p class="stat-value" style="color:#CE9178;">6</p>
        </div>""", unsafe_allow_html=True)

else:
    label, confidence, raw_features = predict(message)

    st.markdown('<hr style="border:none;border-top:1px solid #3C3C3C;margin:1.5rem 0;">',
                unsafe_allow_html=True)

    left, right = st.columns([1.2, 1])

    with left:
        if label == "spam":
            st.markdown(f"""
            <div style="background:#2D1A1A; border-left:3px solid #F44747;
                        border-radius:3px; padding:1rem 1.25rem; margin-bottom:1rem;">
                <p style="font-size:0.68rem;color:#858585;letter-spacing:1.5px;
                          text-transform:uppercase;margin:0 0 0.3rem 0;">Verdict</p>
                <p style="font-size:1.35rem;font-weight:600;color:#F44747;margin:0 0 0.5rem 0;">
                    Spam Detected</p>
                <p style="font-size:1.8rem;font-weight:600;color:#F44747;margin:0;line-height:1.1;">
                    {confidence}%</p>
                <p style="color:#6A6A6A;font-size:0.68rem;margin:0.15rem 0 0 0;">confidence</p>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="background:#1A2D1E; border-left:3px solid #4EC9B0;
                        border-radius:3px; padding:1rem 1.25rem; margin-bottom:1rem;">
                <p style="font-size:0.68rem;color:#858585;letter-spacing:1.5px;
                          text-transform:uppercase;margin:0 0 0.3rem 0;">Verdict</p>
                <p style="font-size:1.35rem;font-weight:600;color:#4EC9B0;margin:0 0 0.5rem 0;">
                    Legitimate</p>
                <p style="font-size:1.8rem;font-weight:600;color:#4EC9B0;margin:0;line-height:1.1;">
                    {confidence}%</p>
                <p style="color:#6A6A6A;font-size:0.68rem;margin:0.15rem 0 0 0;">confidence</p>
            </div>""", unsafe_allow_html=True)

    with right:
        st.markdown('<p style="color:#858585;font-size:0.68rem;letter-spacing:1.5px;text-transform:uppercase;margin:0 0 0.6rem 0;">Extracted Features</p>',
                    unsafe_allow_html=True)
        for name, val in zip(feature_names, raw_features):
            st.markdown(f"""
            <div style="display:flex;justify-content:space-between;
                        padding:0.35rem 0;border-bottom:1px solid #2D2D2D;">
                <span style="color:#CE9178;font-size:0.78rem;">{name}</span>
                <span style="color:#D4D4D4;font-size:0.78rem;">{int(val) if val == int(val) else f'{val:.3f}'}</span>
            </div>""", unsafe_allow_html=True)

st.markdown("""
<div style="border-top:1px solid #3C3C3C;margin-top:2rem;padding-top:0.75rem;">
    <p style="color:#454545;font-size:0.72rem;margin:0;">
        CS2806 Calculus &nbsp;|&nbsp; CIE-3 &nbsp;|&nbsp; RV University, Bengaluru
    </p>
</div>
""", unsafe_allow_html=True)