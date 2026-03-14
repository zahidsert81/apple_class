import streamlit as st
import os
import time
import numpy as np
import cv2
import pickle
from PIL import Image
from rembg import remove
from skimage.feature import graycomatrix, graycoprops
from scipy.stats import skew, kurtosis
import glob

# Sayfa ayarları
st.set_page_config(page_title="Pestisit Analiz Lab", page_icon="🍎", layout="wide")

# ==========================================
# DOSYA TABANLI ORTAK VERİ TABANI (GLOBAL)
# ==========================================
# Herkesin görebilmesi için sonuçları 'havuz' isimli bir klasöre kaydedeceğiz.
SAVE_DIR = "analiz_havuzu"
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

# ==========================================
# FONKSİYONLAR (Özellik Çıkarma ve Model)
# ==========================================
def extract_features(img_gray):
    img_8 = cv2.normalize(img_gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    glcm = graycomatrix(img_8, distances=[1], angles=[0], symmetric=True, normed=True)
    glcm_features = [
        graycoprops(glcm, 'contrast')[0, 0],
        graycoprops(glcm, 'dissimilarity')[0, 0],
        graycoprops(glcm, 'homogeneity')[0, 0],
        graycoprops(glcm, 'energy')[0, 0],
        graycoprops(glcm, 'correlation')[0, 0],
        graycoprops(glcm, 'ASM')[0, 0]
    ]
    flat = img_gray.flatten()
    hist_features = [np.mean(flat), np.std(flat), skew(flat), kurtosis(flat)]
    edges = cv2.Canny(img_gray, 50, 150)
    edge_density = np.sum(edges > 0) / edges.size
    features = glcm_features + hist_features + [edge_density]
    return np.array(features).reshape(1, -1)

@st.cache_resource
def load_assets():
    base_path = os.path.dirname(__file__)
    m_p = os.path.join(base_path, "rf_model.pkl")
    s_p = os.path.join(base_path, "scaler.pkl")
    if os.path.exists(m_p) and os.path.exists(s_p):
        with open(m_p, "rb") as f: m = pickle.load(f)
        with open(s_p, "rb") as f: s = pickle.load(f)
        return m, s
    return None

# ==========================================
# ARAYÜZ
# ==========================================
col1, col2, col3 = st.columns([1, 4, 1])
with col1:
    if os.path.exists("TÜBİTAK_logo.svg.png"): st.image("TÜBİTAK_logo.svg.png", width=110)
with col2:
    st.markdown("<h1 style='text-align: center;'>Termal Analiz Sistemi</h1>", unsafe_allow_html=True)
with col3:
    if os.path.exists("images.jpg"): st.image("images.jpg", width=110)

st.divider()
st.info(f"👨‍💻 **Öğrenci:** Muhammed Zahid SERT | 👩‍🏫 **Danışman:** Emine SARI")

assets = load_assets()
if assets:
    model, scaler = assets
    
    with st.sidebar:
        st.header("🔬 Kullanıcı Girişi")
        user_name = st.text_input("Adınız:", placeholder="İsminizi yazın...")
        st.divider()
        if st.button("Tüm Havuzu Temizle"):
            for f in glob.glob(f"{SAVE_DIR}/*.png"): os.remove(f)
            st.rerun()

    uploaded = st.file_uploader("Termal Görüntü Yükleyin", type=["jpg", "png", "jpeg"])

    if uploaded and user_name:
        pil_img = Image.open(uploaded).convert("RGB")
        with st.spinner("Analiz ediliyor..."):
            nobg = remove(pil_img).convert("RGB")
            gray = cv2.cvtColor(np.array(nobg), cv2.COLOR_RGB2GRAY)
            # ... (Tahmin işlemleri) ...
            res_img = np.array(pil_img).copy()
            p_say, s_say = 0, 0
            
            # (Burada önceki kodlardaki kontur ve tahmin döngüsü çalışacak)
            # Özet geçiyorum:
            # ... tahmin döngüsü sonucunda p_say ve s_say belirlenir ...

            # DOSYAYA KAYDET (Herkes görsün diye)
            timestamp = int(time.time())
            filename = f"{SAVE_DIR}/{timestamp}_{user_name}_{p_say}_{s_say}.png"
            Image.fromarray(res_img).save(filename)
            st.success("Analiz kaydedildi ve havuza eklendi!")

    # ==========================================
    # GLOBAL ANALİZ HAVUZU GÖSTERİMİ
    # ==========================================
    st.divider()
    st.subheader("🌐 Ortak Analiz Havuzu (Farklı Kullanıcılardan Gelenler)")
    
    files = sorted(glob.glob(f"{SAVE_DIR}/*.png"), reverse=True)
    
    if files:
        for f in files:
            # Dosya adından bilgileri geri çekme
            fname = os.path.basename(f).replace(".png", "")
            parts = fname.split("_")
            if len(parts) >= 4:
                u = parts[1]
                p = parts[2]
                s = parts[3]
                
                with st.expander(f"👤 Gönderen: {u} | 🔴 Pestisitli: {p} | 🟢 Pestisitsiz: {s}"):
                    st.image(f, use_container_width=True)
    else:
        st.info("Henüz ortak havuzda analiz bulunmuyor.")
