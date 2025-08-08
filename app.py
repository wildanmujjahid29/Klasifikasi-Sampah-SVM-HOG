import streamlit as st
import joblib
import numpy as np
import cv2
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.transform import resize
from skimage.feature import hog
from skimage.filters import sobel_h, sobel_v, sobel

# ===== Load pipeline =====
pipeline = joblib.load("svm_model_sobel_hog.pkl")

# ===== Parameter =====
IMG_SIZE = (128, 128)
label_map = {0: "Anorganik", 1: "Organik"}

# ===== Fungsi ekstraksi fitur =====
def extract_sobel_features(image):
    edge_h = sobel_h(image)
    edge_v = sobel_v(image)
    edge_mag = sobel(image)
    threshold = 0.1
    edge_density = np.sum(edge_mag > threshold) / edge_mag.size
    mean_mag = np.mean(edge_mag)
    std_mag = np.std(edge_mag)
    return [edge_density, mean_mag, std_mag]

def extract_hog_features(image):
    features, _ = hog(image,
                      orientations=9,
                      pixels_per_cell=(8, 8),
                      cells_per_block=(2, 2),
                      block_norm='L2-Hys',
                      visualize=True)
    return features

def preprocess_image(img_path):
    img = imread(img_path)
    img_resized = resize(img, IMG_SIZE, anti_aliasing=True)
    img_gray = rgb2gray(img_resized)
    img_denoised = cv2.GaussianBlur(img_gray, (3, 3), 0)
    sobel_feats = extract_sobel_features(img_denoised)
    hog_feats = extract_hog_features(img_denoised)
    combined_feats = np.hstack([sobel_feats, hog_feats])
    return combined_feats

# ===== Streamlit App =====
st.title("Klasifikasi Sampah - Sobel + HOG + SVM")

uploaded_file = st.file_uploader("Upload gambar sampah", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Simpan file sementara
    with open("temp_image.png", "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Preprocess & prediksi
    features = preprocess_image("temp_image.png").reshape(1, -1)
    pred = pipeline.predict(features)[0]
    proba = pipeline.predict_proba(features)[0]

    # Tampilkan gambar & hasil
    st.image(uploaded_file, caption="Gambar yang diunggah", use_column_width=True)
    st.write(f"**Prediksi:** {label_map[pred]}")
    st.write(f"**Probabilitas:**")
    for i, label in label_map.items():
        st.write(f"{label}: {proba[i]:.4f}")
