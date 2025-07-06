"""
Utilitas untuk preprocessing gambar dan ekstraksi fitur HOG
"""

import numpy as np
import cv2
from PIL import Image
from skimage.feature import hog
from skimage import exposure
import matplotlib.pyplot as plt

def preprocess_image(image, target_size=(128, 128)):
    """
    Preprocessing gambar untuk klasifikasi
    
    Args:
        image: Input gambar (PIL Image atau numpy array)
        target_size: Ukuran target resize (width, height)
    
    Returns:
        numpy array: Gambar yang sudah dipreprocess
    """
    # Konversi PIL ke numpy array jika diperlukan
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Resize gambar
    image_resized = cv2.resize(image, target_size)
    
    # Konversi ke grayscale jika gambar berwarna
    if len(image_resized.shape) == 3:
        image_gray = cv2.cvtColor(image_resized, cv2.COLOR_RGB2GRAY)
    else:
        image_gray = image_resized
    
    # Normalisasi pixel ke [0,1]
    image_normalized = image_gray.astype(np.float32) / 255.0
    
    return image_normalized

def extract_hog_features(image, orientations=9, pixels_per_cell=(8, 8), 
                        cells_per_block=(2, 2), visualize=False):
    """
    Ekstraksi fitur HOG dari gambar
    
    Args:
        image: Input gambar (numpy array)
        orientations: Jumlah orientasi gradient
        pixels_per_cell: Ukuran pixel per cell
        cells_per_block: Ukuran cell per block
        visualize: Apakah menghasilkan visualisasi HOG
    
    Returns:
        numpy array: Fitur HOG
        numpy array: Gambar HOG (jika visualize=True)
    """
    if visualize:
        features, hog_image = hog(
            image,
            orientations=orientations,
            pixels_per_cell=pixels_per_cell,
            cells_per_block=cells_per_block,
            visualize=True,
            block_norm='L2-Hys'
        )
        
        # Rescale hog image untuk visualisasi
        hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
        
        return features, hog_image_rescaled
    else:
        features = hog(
            image,
            orientations=orientations,
            pixels_per_cell=pixels_per_cell,
            cells_per_block=cells_per_block,
            visualize=False,
            block_norm='L2-Hys'
        )
        
        return features

def predict_waste_class(model, features):
    """
    Prediksi kelas sampah menggunakan model SVM
    
    Args:
        model: Model SVM yang sudah dilatih
        features: Fitur HOG yang sudah diekstrak
    
    Returns:
        tuple: (label, confidence_score)
    """
    # Reshape features untuk prediksi
    features_reshaped = features.reshape(1, -1)
    
    # Prediksi
    prediction = model.predict(features_reshaped)[0]
    confidence = model.decision_function(features_reshaped)[0]
    
    # Konversi ke label
    if prediction == 1:
        label = "Organik"
    else:
        label = "Anorganik"
    
    return label, confidence

def create_hog_visualization(original_image, hog_image, title="HOG Visualization"):
    """
    Membuat visualisasi perbandingan gambar asli dan HOG
    
    Args:
        original_image: Gambar asli
        hog_image: Gambar HOG
        title: Judul visualisasi
    
    Returns:
        matplotlib.figure.Figure: Figure matplotlib
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Gambar asli
    ax1.imshow(original_image, cmap='gray')
    ax1.set_title('Gambar Asli (Preprocessing)')
    ax1.axis('off')
    
    # Gambar HOG
    ax2.imshow(hog_image, cmap='gray')
    ax2.set_title('HOG Features')
    ax2.axis('off')
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    return fig

def validate_image(image_file, max_size_mb=10):
    """
    Validasi file gambar
    
    Args:
        image_file: File gambar yang diupload
        max_size_mb: Ukuran maksimal file dalam MB
    
    Returns:
        tuple: (is_valid, error_message)
    """
    # Cek ukuran file
    if image_file.size > max_size_mb * 1024 * 1024:
        return False, f"Ukuran file terlalu besar! Maksimal {max_size_mb}MB"
    
    # Cek format file
    allowed_formats = ['jpg', 'jpeg', 'png']
    file_extension = image_file.name.split('.')[-1].lower()
    
    if file_extension not in allowed_formats:
        return False, f"Format file tidak didukung! Gunakan: {', '.join(allowed_formats)}"
    
    return True, "Valid"

def get_waste_info(prediction_label):
    """
    Mendapatkan informasi detail tentang jenis sampah
    
    Args:
        prediction_label: Label prediksi ('Organik' atau 'Anorganik')
    
    Returns:
        dict: Informasi detail tentang jenis sampah
    """
    if prediction_label == "Organik":
        return {
            "emoji": "🥬",
            "color": "#4CAF50",
            "description": "Sampah yang dapat terurai secara alami",
            "examples": ["Sisa makanan", "Daun kering", "Kulit buah", "Kertas"],
            "handling": "Dapat dijadikan kompos atau pupuk organik"
        }
    else:
        return {
            "emoji": "🔧",
            "color": "#FF6B6B",
            "description": "Sampah yang tidak dapat terurai secara alami",
            "examples": ["Plastik", "Logam", "Kaca", "Karet"],
            "handling": "Perlu daur ulang atau pemrosesan khusus"
        }