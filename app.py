import streamlit as st
import numpy as np
import joblib
from PIL import Image
import cv2
from skimage.feature import hog
from skimage import exposure
import matplotlib.pyplot as plt

# ------------------------------
# Konfigurasi Halaman
# ------------------------------
st.set_page_config(
    page_title="EcoVision - AI Waste Classifier",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ------------------------------
# Custom CSS (Modern & Elegant)
# ------------------------------
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');

    * {
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }

    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        min-height: 100vh;
    }

    .main-container {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(20px);
        border-radius: 24px;
        padding: 2rem;
        margin: 2rem auto;
        max-width: 1200px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 25px 50px rgba(0, 0, 0, 0.1);
    }

    .hero-section {
        text-align: center;
        padding: 3rem 0;
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.1), rgba(255, 255, 255, 0.05));
        border-radius: 20px;
        margin-bottom: 2rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }

    .hero-title {
        font-family: 'Poppins', sans-serif;
        font-size: 3.5rem;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 1rem;
        text-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
        background: linear-gradient(135deg, #ffffff, #f0f0f0);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    .hero-subtitle {
        font-family: 'Poppins', sans-serif;
        font-size: 1.3rem;
        font-weight: 300;
        color: rgba(255, 255, 255, 0.9);
        margin-bottom: 2rem;
    }

    .hero-badge {
        display: inline-block;
        background: linear-gradient(135deg, #4facfe, #00f2fe);
        color: white;
        padding: 0.5rem 1.5rem;
        border-radius: 50px;
        font-size: 0.9rem;
        font-weight: 500;
        margin-bottom: 1rem;
        box-shadow: 0 10px 30px rgba(79, 172, 254, 0.3);
    }

    .stTabs {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 1rem;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
        background: transparent;
        border-bottom: none;
    }

    .stTabs [data-baseweb="tab"] {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.1), rgba(255, 255, 255, 0.05));
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 12px;
        padding: 1rem 2rem;
        color: rgba(255, 255, 255, 0.8);
        font-weight: 500;
        transition: all 0.3s ease;
    }

    .stTabs [data-baseweb="tab"]:hover {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.2), rgba(255, 255, 255, 0.1));
        border-color: rgba(255, 255, 255, 0.3);
        color: white;
        transform: translateY(-2px);
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #4facfe, #00f2fe) !important;
        border-color: transparent !important;
        color: white !important;
        box-shadow: 0 10px 30px rgba(79, 172, 254, 0.3);
    }

    .stTabs [data-baseweb="tab-panel"] {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        padding: 2rem;
        margin-top: 1rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }

    .upload-section {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.1), rgba(255, 255, 255, 0.05));
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
        border: 2px dashed rgba(255, 255, 255, 0.3);
        transition: all 0.3s ease;
        margin-bottom: 2rem;
    }

    .upload-section:hover {
        border-color: rgba(255, 255, 255, 0.5);
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.15), rgba(255, 255, 255, 0.08));
    }

    .stFileUploader > div {
        border: none !important;
        background: transparent !important;
    }

    .stFileUploader label {
        color: rgba(255, 255, 255, 0.9) !important;
        font-weight: 500 !important;
    }

    .result-card {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.15), rgba(255, 255, 255, 0.08));
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
        margin: 2rem 0;
        border: 1px solid rgba(255, 255, 255, 0.2);
        backdrop-filter: blur(10px);
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
    }

    .result-icon {
        font-size: 4rem;
        margin-bottom: 1rem;
        display: block;
    }

    .result-label {
        font-size: 2rem;
        font-weight: 600;
        color: white;
        margin-bottom: 1rem;
    }

    .result-desc {
        font-size: 1.1rem;
        color: rgba(255, 255, 255, 0.8);
        line-height: 1.6;
    }

    .stButton > button {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        border: none;
        border-radius: 50px;
        font-size: 1rem;
        font-weight: 600;
        padding: 0.8rem 2rem;
        transition: all 0.3s ease;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
        width: 100%;
    }

    .stButton > button:hover {
        background: linear-gradient(135deg, #764ba2, #667eea);
        transform: translateY(-2px);
        box-shadow: 0 15px 40px rgba(102, 126, 234, 0.4);
    }

    .stButton > button:active {
        transform: translateY(0);
    }

    .feature-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 2rem;
        margin: 2rem 0;
    }

    .feature-card {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.1), rgba(255, 255, 255, 0.05));
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.1);
        transition: all 0.3s ease;
    }

    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.2);
        border-color: rgba(255, 255, 255, 0.3);
    }

    .feature-icon {
        font-size: 3rem;
        margin-bottom: 1rem;
    }

    .feature-title {
        font-size: 1.3rem;
        font-weight: 600;
        color: white;
        margin-bottom: 1rem;
    }

    .feature-desc {
        color: rgba(255, 255, 255, 0.7);
        line-height: 1.6;
    }

    .stats-container {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.1), rgba(255, 255, 255, 0.05));
        border-radius: 20px;
        padding: 2rem;
        margin: 2rem 0;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }

    .stats-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 2rem;
    }

    .stat-item {
        text-align: center;
        padding: 1rem;
    }

    .stat-number {
        font-size: 3rem;
        font-weight: 700;
        color: #4facfe;
        margin-bottom: 0.5rem;
    }

    .stat-label {
        color: rgba(255, 255, 255, 0.8);
        font-weight: 500;
    }

    .footer {
        text-align: center;
        padding: 2rem 0;
        margin-top: 3rem;
        border-top: 1px solid rgba(255, 255, 255, 0.1);
        color: rgba(255, 255, 255, 0.7);
    }

    .stExpander {
        background: rgba(255, 255, 255, 0.05) !important;
        border-radius: 12px !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
    }

    .stExpander > div > div {
        color: rgba(255, 255, 255, 0.9) !important;
    }

    .stSpinner {
        color: #4facfe !important;
    }

    /* Responsiveness */
    @media (max-width: 768px) {
        .hero-title {
            font-size: 2.5rem;
        }
        
        .hero-subtitle {
            font-size: 1.1rem;
        }
        
        .main-container {
            margin: 1rem;
            padding: 1rem;
        }
        
        .feature-grid {
            grid-template-columns: 1fr;
        }
    }
    </style>
    """, unsafe_allow_html=True)

# ------------------------------
# Load Model
# ------------------------------
@st.cache_resource
def load_model():
    try:
        return joblib.load("svm_hog_model.joblib")
    except Exception as e:
        st.error(f"❌ Gagal memuat model: {e}")
        return None

model = load_model()

# ------------------------------
# Preprocessing Functions
# ------------------------------
def preprocess_image(image):
    image = np.array(image)
    image = cv2.resize(image, (128, 128))
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return image.astype("float32") / 255.0

def extract_hog_features(image):
    features, hog_img = hog(
        image, orientations=9, pixels_per_cell=(8, 8),
        cells_per_block=(2, 2), visualize=True, block_norm="L2-Hys"
    )
    hog_img = exposure.rescale_intensity(hog_img, in_range=(0, 10))
    return features, hog_img

def predict(model, features):
    pred = model.predict(features.reshape(1, -1))[0]
    label = "Organik" if pred == 1 else "Anorganik"
    icon = "🌿" if pred == 1 else "♻️"
    desc = "Sampah organik dapat terurai secara alami, seperti sisa makanan dan daun kering. Cocok untuk kompos!" if pred == 1 else "Sampah anorganik sulit terurai dan perlu didaur ulang, seperti plastik, logam, dan kaca."
    return label, icon, desc

# ------------------------------
# Main Application
# ------------------------------
st.markdown('<div class="main-container">', unsafe_allow_html=True)

# Hero Section
st.markdown("""
<div class="hero-section">
    <div class="hero-badge">🚀 Powered by AI & Machine Learning</div>
    <h1 class="hero-title">EcoVision</h1>
    <p class="hero-subtitle">Klasifikasi Sampah Cerdas dengan Teknologi AI</p>
</div>
""", unsafe_allow_html=True)

# Feature Cards
st.markdown("""
<div class="feature-grid">
    <div class="feature-card">
        <div class="feature-icon">🔍</div>
        <div class="feature-title">Deteksi Akurat</div>
        <div class="feature-desc">Akurasi 99% dalam mengklasifikasikan sampah organik dan anorganik</div>
    </div>
    <div class="feature-card">
        <div class="feature-icon">⚡</div>
        <div class="feature-title">Proses Cepat</div>
        <div class="feature-desc">Hasil klasifikasi dalam hitungan detik dengan teknologi HOG</div>
    </div>
    <div class="feature-card">
        <div class="feature-icon">🌍</div>
        <div class="feature-title">Ramah Lingkungan</div>
        <div class="feature-desc">Mendukung pengelolaan sampah yang berkelanjutan</div>
    </div>
</div>
""", unsafe_allow_html=True)

# Main Tabs
tab1, tab2, tab3 = st.tabs(["📁 Upload Gambar", "📸 Kamera", "📊 Tentang"])

with tab1:
    st.markdown("""
    <div class="upload-section">
        <h3 style="color: white; margin-bottom: 1rem;">📁 Unggah Gambar Sampah</h3>
        <p style="color: rgba(255, 255, 255, 0.8); margin-bottom: 1rem;">
            Pilih gambar sampah dalam format JPG, JPEG, atau PNG
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded = st.file_uploader("", type=["jpg", "jpeg", "png"], key="upload_file")
    
    if uploaded:
        image = Image.open(uploaded)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.image(image, caption="Gambar yang Diunggah", use_container_width=True)
        
        with col2:
            if st.button("🔍 Klasifikasikan Sampah", key="btn_upload"):
                with st.spinner("🤖 AI sedang menganalisis gambar..."):
                    processed = preprocess_image(image)
                    features, hog_img = extract_hog_features(processed)
                    label, icon, desc = predict(model, features)

                    st.markdown(f"""
                    <div class="result-card">
                        <span class="result-icon">{icon}</span>
                        <div class="result-label">{label}</div>
                        <div class="result-desc">{desc}</div>
                    </div>
                    """, unsafe_allow_html=True)

                    with st.expander("🔬 Visualisasi Fitur HOG"):
                        col3, col4 = st.columns(2)
                        with col3:
                            fig, ax = plt.subplots(figsize=(6, 6))
                            ax.imshow(processed, cmap="gray")
                            ax.set_title("Gambar Preprocessed", color='white', fontsize=12)
                            ax.axis("off")
                            fig.patch.set_facecolor('none')
                            st.pyplot(fig)
                        with col4:
                            fig, ax = plt.subplots(figsize=(6, 6))
                            ax.imshow(hog_img, cmap="hot")
                            ax.set_title("Fitur HOG", color='white', fontsize=12)
                            ax.axis("off")
                            fig.patch.set_facecolor('none')
                            st.pyplot(fig)

with tab2:
    st.markdown("""
    <div class="upload-section">
        <h3 style="color: white; margin-bottom: 1rem;">📸 Ambil Foto Sampah</h3>
        <p style="color: rgba(255, 255, 255, 0.8); margin-bottom: 1rem;">
            Gunakan kamera untuk mengambil foto sampah secara langsung
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    camera_img = st.camera_input("📷 Klik untuk mengambil foto")
    
    if camera_img:
        image = Image.open(camera_img)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.image(image, caption="Foto dari Kamera", use_container_width=True)
        
        with col2:
            if st.button("🔍 Klasifikasikan Sampah", key="btn_camera"):
                with st.spinner("🤖 AI sedang menganalisis foto..."):
                    processed = preprocess_image(image)
                    features, hog_img = extract_hog_features(processed)
                    label, icon, desc = predict(model, features)

                    st.markdown(f"""
                    <div class="result-card">
                        <span class="result-icon">{icon}</span>
                        <div class="result-label">{label}</div>
                        <div class="result-desc">{desc}</div>
                    </div>
                    """, unsafe_allow_html=True)

                    with st.expander("🔬 Visualisasi Fitur HOG"):
                        col3, col4 = st.columns(2)
                        with col3:
                            fig, ax = plt.subplots(figsize=(6, 6))
                            ax.imshow(processed, cmap="gray")
                            ax.set_title("Gambar Preprocessed", color='white', fontsize=12)
                            ax.axis("off")
                            fig.patch.set_facecolor('none')
                            st.pyplot(fig)
                        with col4:
                            fig, ax = plt.subplots(figsize=(6, 6))
                            ax.imshow(hog_img, cmap="hot")
                            ax.set_title("Fitur HOG", color='white', fontsize=12)
                            ax.axis("off")
                            fig.patch.set_facecolor('none')
                            st.pyplot(fig)

with tab3:
    # Performance Stats
    st.markdown("""
    <div class="stats-container">
        <h4 style="color: white; text-align: center; margin-bottom: 2rem;">📈 Performa Model</h4>
        <div class="stats-grid">
            <div class="stat-item">
                <div class="stat-number">99%</div>
                <div class="stat-label">Akurasi Keseluruhan</div>
            </div>
            <div class="stat-item">
                <div class="stat-number">0.99</div>
                <div class="stat-label">F1-Score</div>
            </div>
            <div class="stat-item">
                <div class="stat-number">0.98</div>
                <div class="stat-label">Precision</div>
            </div>
            <div class="stat-item">
                <div class="stat-number">1.00</div>
                <div class="stat-label">Recall</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="footer">
    <p>🌱 EcoVision - Klasifikasi Sampah Cerdas | Dibuat dengan ❤️ menggunakan Streamlit</p>
    <p style="margin-top: 0.5rem; font-size: 0.9rem;">
        Mendukung Sustainable Development Goals (SDGs) untuk lingkungan yang lebih bersih
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)