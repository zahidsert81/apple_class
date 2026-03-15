import streamlit as st
import os
import time
import numpy as np
import cv2
import pickle
import qrcode
from io import BytesIO
from PIL import Image
from rembg import remove
from skimage.feature import graycomatrix, graycoprops
from scipy.stats import skew, kurtosis

# 1. AYARLAR
st.set_page_config(page_title="Pestisit Analiz Laboratuvarı", page_icon="🧪", layout="wide")

SAVE_DIR = "analiz_havuzu"
MOBILE_UPLOAD_DIR = "mobil_aktarim"
MODEL_FILE = "rf_model.pkl"
SCALER_FILE = "scaler.pkl"

for d in [SAVE_DIR, MOBILE_UPLOAD_DIR]:
    if not os.path.exists(d): os.makedirs(d)

# 2. ÖZELLİK ÇIKARMA (Orijinal 11 Özellik)
def extract_features(img_gray):
    img_gray = cv2.medianBlur(img_gray, 5)
    img_8 = cv2.normalize(img_gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    glcm = graycomatrix(img_8, distances=[1], angles=[0], symmetric=True, normed=True)
    glcm_feats = [graycoprops(glcm, p)[0, 0] for p in ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']]
    flat = img_gray.flatten()
    stats = [np.mean(flat), np.std(flat), skew(flat), kurtosis(flat)]
    edges = cv2.Canny(img_8, 50, 150)
    density = np.sum(edges > 0) / edges.size
    return np.array(glcm_feats + stats + [density]).reshape(1, -1)

@st.cache_resource
def load_assets():
    if os.path.exists(MODEL_FILE) and os.path.exists(SCALER_FILE):
        with open(MODEL_FILE, "rb") as f: m = pickle.load(f)
        with open(SCALER_FILE, "rb") as f: s = pickle.load(f)
        return m, s
    return None

assets = load_assets()

# --- STREAMLIT CLOUD URL TESPİTİ ---
# Cloud ortamında çalışan uygulamanın gerçek URL'sini manuel girmek en garantisidir
# Buraya uygulamanın ".streamlit.app" ile biten linkini yapıştır
APP_URL = "https://pestisit-kontrol.streamlit.app/" 

query_params = st.query_params
is_mobile = query_params.get("mode") == "mobile"

if is_mobile:
    st.markdown("### 📱 Mobil Aktarım")
    m_file = st.file_uploader("Fotoğraf Gönder", type=["jpg", "png", "jpeg"])
    if m_file:
        img = Image.open(m_file)
        img.save(os.path.join(MOBILE_UPLOAD_DIR, "transfer.png"))
        st.success("✅ Gönderildi! Bilgisayardaki sayfayı yenileyin.")
else:
    # --- MASAÜSTÜ ARAYÜZÜ ---
    col_l, col_m, col_r = st.columns([1, 4, 1])
    with col_l:
        if os.path.exists("TÜBİTAK_logo.svg.png"): st.image("TÜBİTAK_logo.svg.png", width=110)
    with col_m:
        st.markdown("<h1 style='text-align: center; color: #1E3A8A;'>Pestisit Tespit Sistemi</h1>", unsafe_allow_html=True)
    with col_r:
        if os.path.exists("images.jpg"): st.image("images.jpg", width=110)

    st.divider()
    m_col, s_col = st.columns([3, 1])

    with s_col:
        st.subheader("📲 Mobil Bağlantı")
        # URL'ye modu ekliyoruz
        mobile_link = f"{APP_URL}/?mode=mobile"
        
        qr_img = qrcode.make(mobile_link)
        buf = BytesIO()
        qr_img.save(buf, format="PNG")
        st.image(buf, caption="Telefondan yüklemek için okutun")
        
        sens = st.slider("Hassasiyet", 0.1, 0.9, 0.45)

    with m_col:
        m_path = os.path.join(MOBILE_UPLOAD_DIR, "transfer.png")
        source = None
        
        if os.path.exists(m_path):
            st.info("📱 Mobilden veri geldi!")
            if st.button("Gelen Veriyi Analiz Et"):
                source = Image.open(m_path)
            if st.button("Temizle"):
                os.remove(m_path)
                st.rerun()

        up = st.file_uploader("Veya Manuel Yükle", type=["jpg", "png", "jpeg"])
        if up: source = Image.open(up)

        if source and assets:
            model, scaler = assets
            # ... (Analiz kodları - Yukarıdakiyle aynı)
            # Analiz kısmını buraya yapıştırabilirsin.
