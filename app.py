"""
Streamlit web app for automated acne severity detection.
Run with: streamlit run app.py
"""

import os
import json
import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH  = "dataset/model/acne_model.h5"
USERS_FILE  = "users.json"
IMG_SIZE    = (224, 224)
CLASSES     = ["Mild", "Moderate", "Severe"]

SEVERITY_META = {
    "Mild": {
        "color"  : "#27ae60",
        "icon"   : "🟢",
        "advice" : (
            "Your acne appears **mild**. Maintain a gentle daily cleansing routine, "
            "use non-comedogenic moisturisers, and avoid touching your face. "
            "Over-the-counter products with salicylic acid can help."
        ),
    },
    "Moderate": {
        "color"  : "#e67e22",
        "icon"   : "🟡",
        "advice" : (
            "**Moderate acne** detected. Consider OTC treatments containing benzoyl "
            "peroxide or retinoids. If no improvement is seen in 6–8 weeks, "
            "consult a dermatologist."
        ),
    },
    "Severe": {
        "color"  : "#e74c3c",
        "icon"   : "🔴",
        "advice" : (
            "**Severe acne** detected. Please consult a dermatologist as soon as "
            "possible. Prescription medications such as antibiotics, isotretinoin, "
            "or hormonal therapy may be required."
        ),
    },
}

# ── User storage ───────────────────────────────────────────────────────────────
def load_users():
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, "r") as f:
            return json.load(f)
    return {}

def save_users(users):
    with open(USERS_FILE, "w") as f:
        json.dump(users, f)

# ── Page setup ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AcneScan AI",
    page_icon="🔬",
    layout="centered",
)

# ── Global CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700;800&display=swap');

* { font-family: 'Poppins', sans-serif !important; }

.stApp {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e) !important;
    min-height: 100vh;
}

/* Animated background blobs */
.stApp::before {
    content: '';
    position: fixed;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(ellipse at 20% 50%, rgba(120,40,200,0.15) 0%, transparent 50%),
                radial-gradient(ellipse at 80% 20%, rgba(233,69,96,0.15) 0%, transparent 50%),
                radial-gradient(ellipse at 50% 80%, rgba(0,200,255,0.1) 0%, transparent 50%);
    animation: blobMove 8s ease-in-out infinite alternate;
    pointer-events: none;
    z-index: 0;
}

@keyframes blobMove {
    0%   { transform: translate(0,0) rotate(0deg); }
    100% { transform: translate(30px,20px) rotate(10deg); }
}

@keyframes float {
    0%, 100% { transform: translateY(0px) rotate(0deg); }
    33%       { transform: translateY(-15px) rotate(5deg); }
    66%       { transform: translateY(-8px) rotate(-3deg); }
}

@keyframes shimmer {
    0%   { background-position: -400% center; }
    100% { background-position: 400% center; }
}

@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(50px); }
    to   { opacity: 1; transform: translateY(0); }
}

@keyframes glow {
    0%, 100% { box-shadow: 0 0 20px rgba(233,69,96,0.4), 0 0 40px rgba(233,69,96,0.2); }
    50%       { box-shadow: 0 0 30px rgba(233,69,96,0.7), 0 0 60px rgba(233,69,96,0.4); }
}

@keyframes borderGlow {
    0%, 100% { border-color: rgba(120,40,200,0.5); }
    50%       { border-color: rgba(233,69,96,0.8); }
}

@keyframes pulse {
    0%, 100% { transform: scale(1); }
    50%       { transform: scale(1.08); }
}

/* Hide streamlit default elements */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2rem !important; }

.login-logo {
    font-size: 5.5rem;
    display: block;
    text-align: center;
    animation: float 4s ease-in-out infinite;
    filter: drop-shadow(0 0 20px rgba(233,69,96,0.6));
}

.login-title {
    font-size: 2.8rem;
    font-weight: 800;
    text-align: center;
    background: linear-gradient(90deg, #ff6b9d, #c44dff, #4daaff, #ff6b9d);
    background-size: 300% auto;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: shimmer 4s linear infinite;
    margin: 0.3rem 0 0.1rem 0;
    letter-spacing: -1px;
}

.login-subtitle {
    text-align: center;
    font-size: 0.9rem;
    color: rgba(255,255,255,0.55);
    margin-bottom: 0.8rem;
    letter-spacing: 2px;
    text-transform: uppercase;
}

.feature-badge {
    display: inline-block;
    background: rgba(255,255,255,0.07);
    border: 1px solid rgba(255,255,255,0.15);
    border-radius: 20px;
    padding: 0.3rem 0.9rem;
    font-size: 0.75rem;
    color: rgba(255,255,255,0.8);
    font-weight: 500;
    margin: 0.2rem;
    backdrop-filter: blur(10px);
}

/* Card */
.stTabs {
    background: rgba(255,255,255,0.04) !important;
    border-radius: 24px !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    backdrop-filter: blur(20px) !important;
    padding: 1.5rem !important;
    box-shadow: 0 25px 60px rgba(0,0,0,0.5), inset 0 1px 0 rgba(255,255,255,0.1) !important;
    animation: fadeInUp 0.7s ease, borderGlow 3s ease-in-out infinite !important;
}

/* Tab buttons */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.05) !important;
    border-radius: 14px !important;
    padding: 4px !important;
    gap: 4px !important;
}

.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: rgba(255,255,255,0.5) !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    padding: 0.5rem 1.5rem !important;
    transition: all 0.3s ease !important;
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #e94560, #9b59b6) !important;
    color: white !important;
    box-shadow: 0 4px 15px rgba(233,69,96,0.4) !important;
}

/* Inputs */
.stTextInput > div > div > input {
    background: rgba(255,255,255,0.07) !important;
    border: 1.5px solid rgba(255,255,255,0.15) !important;
    border-radius: 14px !important;
    color: white !important;
    padding: 0.8rem 1.1rem !important;
    font-size: 0.95rem !important;
    transition: all 0.3s ease !important;
    backdrop-filter: blur(10px) !important;
}

.stTextInput > div > div > input:focus {
    border-color: #e94560 !important;
    background: rgba(255,255,255,0.1) !important;
    box-shadow: 0 0 0 3px rgba(233,69,96,0.2), 0 0 20px rgba(233,69,96,0.1) !important;
}

.stTextInput > div > div > input::placeholder {
    color: rgba(255,255,255,0.3) !important;
}

.stTextInput label {
    color: rgba(255,255,255,0.75) !important;
    font-size: 0.85rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.5px !important;
}

/* Button */
.stButton > button {
    background: linear-gradient(135deg, #e94560 0%, #9b59b6 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 14px !important;
    padding: 0.85rem 2rem !important;
    font-size: 1rem !important;
    font-weight: 700 !important;
    width: 100% !important;
    transition: all 0.3s ease !important;
    animation: glow 2s ease-in-out infinite !important;
    letter-spacing: 1px !important;
    text-transform: uppercase !important;
}

.stButton > button:hover {
    transform: translateY(-3px) scale(1.02) !important;
    box-shadow: 0 15px 35px rgba(233,69,96,0.5) !important;
}

/* Main app header */
.main-header {
    background: linear-gradient(135deg, #1a237e 0%, #6a1b9a 50%, #e94560 100%);
    padding: 1.5rem 2rem;
    border-radius: 20px;
    margin-bottom: 1.5rem;
    box-shadow: 0 8px 30px rgba(106,27,154,0.4);
    animation: pulse 3s ease-in-out infinite;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f0c29 0%, #302b63 100%) !important;
    border-right: 1px solid rgba(255,255,255,0.1) !important;
}
[data-testid="stSidebar"] * { color: white !important; }

/* Main content text */
.stMarkdown p, .stMarkdown li { color: rgba(255,255,255,0.85) !important; }
h1, h2, h3 { color: white !important; }
.stAlert { border-radius: 14px !important; }

/* Divider */
hr { border-color: rgba(255,255,255,0.1) !important; }

/* File uploader */
[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.05) !important;
    border: 2px dashed rgba(233,69,96,0.4) !important;
    border-radius: 16px !important;
    padding: 1rem !important;
}

/* Progress bar */
.stProgress > div > div {
    background: linear-gradient(90deg, #e94560, #9b59b6) !important;
    border-radius: 10px !important;
}
</style>
""", unsafe_allow_html=True)

# ── Session state ──────────────────────────────────────────────────────────────
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = ""

# ── Auth page ──────────────────────────────────────────────────────────────────
def show_auth():
    st.markdown("<span class='login-logo'>🔬</span>", unsafe_allow_html=True)
    st.markdown("<div class='login-title'>AcneScan AI</div>", unsafe_allow_html=True)
    st.markdown("<div class='login-subtitle'>✦ Advanced Acne Severity Detection ✦</div>", unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align:center; margin-bottom:1.8rem;'>
        <span class='feature-badge'>🧠 Deep Learning</span>
        <span class='feature-badge'>📊 3-Level Analysis</span>
        <span class='feature-badge'>⚡ Instant Results</span>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([0.5, 3, 0.5])
    with col2:
        tab1, tab2 = st.tabs(["🔑  Login", "📝  Register"])

        with tab1:
            st.markdown("<br>", unsafe_allow_html=True)
            username = st.text_input("👤  Username", placeholder="Enter your username", key="login_user")
            password = st.text_input("🔒  Password", type="password", placeholder="Enter your password", key="login_pass")
            st.markdown("<br>", unsafe_allow_html=True)

            if st.button("🚀  SIGN IN", use_container_width=True, key="login_btn"):
                users = load_users()
                if username == "" or password == "":
                    st.warning("⚠️ Please fill in all fields.")
                elif username in users and users[username] == password:
                    st.session_state.logged_in = True
                    st.session_state.username = username
                    st.success(f"✅ Welcome back, **{username}**!")
                    st.rerun()
                else:
                    st.error("❌ Invalid username or password.")

            st.markdown("""
            <p style='text-align:center; color:rgba(255,255,255,0.35); font-size:0.78rem; margin-top:1rem;'>
                Don't have an account? Switch to Register tab ↑
            </p>
            """, unsafe_allow_html=True)

        with tab2:
            st.markdown("<br>", unsafe_allow_html=True)
            new_user = st.text_input("👤  Choose Username", placeholder="Min. 3 characters", key="reg_user")
            new_pass = st.text_input("🔒  Choose Password", type="password", placeholder="Min. 6 characters", key="reg_pass")
            confirm  = st.text_input("🔒  Confirm Password", type="password", placeholder="Re-enter password", key="reg_confirm")
            st.markdown("<br>", unsafe_allow_html=True)

            if st.button("✅  CREATE ACCOUNT", use_container_width=True, key="reg_btn"):
                users = load_users()
                if new_user == "" or new_pass == "" or confirm == "":
                    st.warning("⚠️ Please fill in all fields.")
                elif len(new_user) < 3:
                    st.warning("⚠️ Username must be at least 3 characters.")
                elif len(new_pass) < 6:
                    st.warning("⚠️ Password must be at least 6 characters.")
                elif new_pass != confirm:
                    st.error("❌ Passwords do not match.")
                elif new_user in users:
                    st.error("❌ Username already exists. Choose another.")
                else:
                    users[new_user] = new_pass
                    save_users(users)
                    st.success(f"🎉 Account created! Login as **{new_user}**.")

            st.markdown("""
            <p style='text-align:center; color:rgba(255,255,255,0.35); font-size:0.78rem; margin-top:1rem;'>
                Already have an account? Switch to Login tab ↑
            </p>
            """, unsafe_allow_html=True)


# ── Model loading ──────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model…")
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return tf.keras.models.load_model(MODEL_PATH)


def preprocess(image: Image.Image) -> np.ndarray:
    img = image.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


# ── Main app ───────────────────────────────────────────────────────────────────
def show_main_app():
    col1, col2 = st.columns([4, 1])
    with col1:
        st.markdown(f"""
        <div class='main-header'>
            <h2 style='color:white; margin:0; font-size:1.5rem;'>🔬 AcneScan AI</h2>
            <p style='color:rgba(255,255,255,0.7); margin:0.2rem 0 0 0; font-size:0.85rem;'>
                Logged in as <b style='color:#ffd54f;'>{st.session_state.username}</b>
            </p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("<br><br>", unsafe_allow_html=True)
        if st.button("🚪 Logout"):
            st.session_state.logged_in = False
            st.session_state.username = ""
            st.rerun()

    st.markdown(
        "<p style='color:rgba(255,255,255,0.7);'>Upload a clear facial image to instantly assess acne severity using deep learning.</p>",
        unsafe_allow_html=True
    )
    st.divider()

    model = load_model()

    if model is None:
        st.warning("⚠️ No trained model found. Please run `python train_model.py` first.")

    uploaded = st.file_uploader(
        "📁 Upload a facial image (JPG / PNG)",
        type=["jpg", "jpeg", "png"],
        disabled=(model is None),
    )

    if uploaded is not None:
        image = Image.open(uploaded)
        col1, col2 = st.columns([1, 1], gap="large")

        with col1:
            st.subheader("📷 Uploaded Image")
            st.image(image, use_container_width=True)

        with col2:
            st.subheader("📊 Analysis Result")
            with st.spinner("Analysing…"):
                arr   = preprocess(image)
                probs = model.predict(arr)[0]

            pred_idx   = int(np.argmax(probs))
            pred_label = CLASSES[pred_idx]
            confidence = float(probs[pred_idx]) * 100
            meta       = SEVERITY_META[pred_label]

            st.markdown(
                f"<h2 style='color:{meta['color']}'>"
                f"{meta['icon']}  {pred_label} Acne"
                f"</h2>",
                unsafe_allow_html=True,
            )
            st.markdown(f"**Confidence:** `{confidence:.1f}%`")
            st.divider()

            st.markdown("**Severity breakdown:**")
            for cls, prob in zip(CLASSES, probs):
                m = SEVERITY_META[cls]
                st.markdown(f"{m['icon']} **{cls}**")
                st.progress(float(prob), text=f"{prob*100:.1f}%")

        st.divider()
        st.subheader("💡 Recommendation")
        st.info(meta["advice"])
        st.caption(
            "⚠️ **Disclaimer:** This tool is for educational purposes only. "
            "Always consult a qualified dermatologist."
        )
    else:
        st.info("👆 Upload an image above to get started.")

    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown("""
        **Automated Acne Severity Detection**

        Uses a **MobileNetV2**-based CNN fine-tuned on labelled acne images.

        | Severity | Description |
        |----------|-------------|
        | 🟢 Mild     | Few comedones, minimal inflammation |
        | 🟡 Moderate | Multiple papules/pustules |
        | 🔴 Severe   | Nodules/cysts, widespread inflammation |
        """)
        st.divider()
        if model:
            st.success("✅ Model loaded successfully")
        else:
            st.error("❌ Model not found — train first")


# ── Router ─────────────────────────────────────────────────────────────────────
if not st.session_state.logged_in:
    show_auth()
else:
    show_main_app()
