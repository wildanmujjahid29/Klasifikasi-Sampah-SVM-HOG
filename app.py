import streamlit as st
import joblib
import numpy as np
import cv2
import pandas as pd
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.transform import resize
from skimage.feature import hog
from skimage.filters import sobel

# =============================================================================
# Konfigurasi Halaman dan Gaya
# =============================================================================
st.set_page_config(page_title="Detektor Jenis Sampah", layout="wide")

# Gaya CSS kustom untuk tema lingkungan
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Anda bisa membuat file style.css atau langsung definisikan di sini
css = """
<style>
/* Latar belakang utama */
.stApp {
    background-color: #F0F8F0; /* Honeydew green */
}

/* Kustomisasi tombol */
.stButton>button {
    border: 2px solid #4CAF50;
    background-color: #4CAF50;
    color: white;
    padding: 0.5em 1em;
    border-radius: 8px;
    font-weight: bold;
}
.stButton>button:hover {
    background-color: #45a049;
    border-color: #45a049;
}

/* Kustomisasi file uploader */
.stFileUploader label {
    font-size: 1.1em;
    font-weight: bold;
    color: #2E8B57; /* SeaGreen */
}
</style>
"""
st.markdown(css, unsafe_allow_html=True)

# =============================================================================
# Memuat Model dan Inisialisasi
# =============================================================================
# Pastikan file model Anda berada di direktori yang sama atau berikan path lengkap
try:
    pipeline = joblib.load("svm_model_sobel_hog.pkl")
except FileNotFoundError:
    st.error("File model 'svm_model_sobel_hog.pkl' tidak ditemukan. Pastikan file berada di direktori yang sama.")
    st.stop()
    
# Parameter
IMG_SIZE = (128, 128)
label_map = {0: "Anorganik", 1: "Organik"}

# =============================================================================
# Fungsi Ekstraksi Fitur (Tidak ada perubahan di sini)
# =============================================================================
def extract_features(image_gray):
    """Menggabungkan ekstraksi Sobel dan HOG."""
    # Fitur Sobel
    edge_mag = sobel(image_gray)
    sobel_feats = [
        np.sum(edge_mag > 0.1) / edge_mag.size,  # Edge density
        np.mean(edge_mag),                       # Mean magnitude
        np.std(edge_mag)                         # Std deviation of magnitude
    ]
    
    # Fitur HOG
    hog_feats = hog(image_gray,
                    orientations=9,
                    pixels_per_cell=(8, 8),
                    cells_per_block=(2, 2),
                    block_norm='L2-Hys',
                    visualize=False) # Set visualize ke False karena kita hanya butuh fiturnya
                    
    # Gabungkan fitur
    combined_feats = np.hstack([sobel_feats, hog_feats])
    return combined_feats

def preprocess_image(uploaded_file):
    """Membaca, mengubah ukuran, dan mengekstrak fitur dari file yang diunggah."""
    # Baca gambar langsung dari buffer file yang diunggah
    img = imread(uploaded_file)
    
    # Preprocessing
    img_resized = resize(img, IMG_SIZE, anti_aliasing=True)
    
    # Konversi ke grayscale, pastikan formatnya benar
    if img_resized.ndim == 3 and img_resized.shape[2] in [3, 4]:
        img_gray = rgb2gray(img_resized)
    else:
        img_gray = img_resized # Asumsikan sudah grayscale jika tidak 3-channel

    img_denoised = cv2.GaussianBlur(img_gray, (3, 3), 0)
    
    # Ekstrak fitur
    features = extract_features(img_denoised)
    return features.reshape(1, -1)

# =============================================================================
# Antarmuka Streamlit
# =============================================================================
st.title("🌿 Detektor Jenis Sampah")
st.markdown("Unggah gambar sampah untuk mengetahui jenisnya dan cara mengelolanya. Mari bersama menjaga bumi!")
st.markdown("---")

# Area upload file
uploaded_file = st.file_uploader("Pilih atau seret gambar sampah ke sini", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Tampilkan spinner saat memproses
    with st.spinner('Menganalisis gambar...'):
        # Preprocess & prediksi
        features = preprocess_image(uploaded_file)
        pred_label = pipeline.predict(features)[0]
        proba = pipeline.predict_proba(features)[0]
        
        # Buat tata letak kolom
        col1, col2 = st.columns([2, 3])

        with col1:
            st.image(uploaded_file, caption="Gambar yang Diunggah", use_column_width=True)

        with col2:
            st.subheader("Hasil Analisis")
            
            # Tampilkan hasil prediksi dengan gaya
            hasil_prediksi = label_map[pred_label]
            if hasil_prediksi == "Organik":
                st.success(f"**Prediksi: {hasil_prediksi}**")
            else:
                st.warning(f"**Prediksi: {hasil_prediksi}**")

            st.subheader("Tingkat Keyakinan Model")
            
            # Buat DataFrame untuk grafik batang
            proba_df = pd.DataFrame({
                'Jenis Sampah': [label_map[0], label_map[1]],
                'Probabilitas': proba
            })
            
            # Tampilkan grafik batang
            st.bar_chart(proba_df.set_index('Jenis Sampah'))

            # Tampilkan Tips Lingkungan berdasarkan hasil
            st.subheader("💡 Tips Lingkungan")
            if hasil_prediksi == "Organik":
                st.info(
                    """
                    **Apa yang harus dilakukan?**
                    - Sampah organik seperti sisa makanan dan daun dapat diolah menjadi **kompos**.
                    - Kompos sangat baik untuk menyuburkan tanah dan mengurangi sampah di TPA.
                    - Anda bisa memulai komposter sederhana di rumah!
                    """
                )
            else:
                st.info(
                    """
                    **Apa yang harus dilakukan?**
                    - Sampah anorganik seperti plastik, kaleng, dan kertas sebaiknya **didaur ulang**.
                    - Pisahkan sampah ini dari sampah organik.
                    - Bersihkan sebelum diserahkan ke bank sampah atau petugas kebersihan.
                    """
                )

st.markdown("---")