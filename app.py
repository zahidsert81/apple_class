import streamlit as st
import numpy as np
import cv2
import pickle
import os
from PIL import Image
from rembg import remove
from skimage.feature import graycomatrix, graycoprops
from scipy.stats import skew, kurtosis
import io
import time

# ==========================================
# 1. ÖZELLİK ÇIKARMA FONKSİYONU
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

# ==========================================
# 2. MODEL VE SCALER OTOMATİK YÜKLEME
# ==========================================
@st.cache_resource
def load_assets():
    model_path = "rf_model.pkl"
    scaler_path = "scaler.pkl"
    
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        with open(model_path, "rb") as f:
            loaded_model = pickle.load(f)
        with open(scaler_path, "rb") as f:
            loaded_scaler = pickle.load(f)
        return loaded_model, loaded_scaler
    return None, None

# ==========================================
# 3. VERİ TABANI SİSTEMİ (SESSION STATE)
# ==========================================
if 'db_analiz' not in st.session_state:
    st.session_state.db_analiz = []

# ==========================================
# 4. ARAYÜZ YAPILANDIRMASI
# ==========================================
st.set_page_config(page_title="Termal Analiz Lab", layout="wide")

st.title("🌡️ Otomatik Termal Analiz & Veri Kayıt Sistemi")
st.markdown("---")

# Modelleri Otomatik Yükle
model, scaler = load_assets()

if model is None:
    st.error("❌ 'rf_model.pkl' veya 'scaler.pkl' dosyaları bulunamadı! Lütfen dosyaları ana dizine ekleyin.")
else:
    # Üst Kısım: Yeni Analiz Alanı
    st.subheader("📸 Yeni Termal Fotoğraf Analizi")
    uploaded_file = st.file_uploader("Bir termal görüntü seçin...", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        pil_img = Image.open(uploaded_file).convert("RGB")
        
        with st.spinner("Yapay zeka analiz ediyor..."):
            # Görüntü İşleme
            nobg = remove(pil_img).convert("RGB")
            gray = cv2.cvtColor(np.array(nobg), cv2.COLOR_RGB2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            _, th = cv2.threshold(blur, 10, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            res_img = np.array(pil_img).copy()
            p_sayisi = 0
            s_sayisi = 0
            
            for cnt in contours:
                if cv2.contourArea(cnt) < 1000: continue
                x, y, w, h = cv2.boundingRect(cnt)
                crop = gray[y:y+h, x:x+w]
                
                feats = extract_features(crop)
                feats_scaled = scaler.transform(feats)
                pred = model.predict(feats_scaled)[0]
                
                if pred == 1:
                    label, color = "PESTISITLI", (255, 0, 0)
                    p_sayisi += 1
                else:
                    label, color = "PESTISITSIZ", (0, 255, 0)
                    s_sayisi += 1
                
                cv2.rectangle(res_img, (x, y), (x+w, y+h), color, 12)
                cv2.putText(res_img, label, (x, y-15), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 4)

            # Veri Tabanına Ekle (Max 100)
            if len(st.session_state.db_analiz) >= 100:
                st.session_state.db_analiz.pop(-1) # En eskiyi sil
            
            st.session_state.db_analiz.insert(0, {
                "image": res_img,
                "p": p_sayisi,
                "s": s_sayisi,
                "t": time.strftime("%H:%M:%S")
            })

            # Anlık Sonuç Gösterimi
            col1, col2 = st.columns([2, 1])
            with col1:
                st.image(res_img, use_container_width=True)
            with col2:
                st.metric("🔴 Tespit Edilen Pestisitli", p_sayisi)
                st.metric("🟢 Tespit Edilen Pestisitsiz", s_sayisi)
                st.write(f"**Analiz Saati:** {time.strftime('%H:%M:%S')}")

    # Alt Kısım: Veri Tabanı Arşivi
    st.divider()
    
    # Toplam İstatistikler
    if st.session_state.db_analiz:
        total_p = sum(item['p'] for item in st.session_state.db_analiz)
        total_s = sum(item['s'] for item in st.session_state.db_analiz)
        
        st.subheader(f"📂 Analiz Arşivi ({len(st.session_state.db_analiz)} / 100)")
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Toplam Analiz Sayısı", len(st.session_state.db_analiz))
        c2.metric("Toplam Pestisitli", total_p)
        c3.metric("Toplam Pestisitsiz", total_s)

        # Galeri
        st.markdown("---")
        gal_cols = st.columns(4)
        for idx, item in enumerate(st.session_state.db_analiz):
            with gal_cols[idx % 4]:
                st.image(item["image"], use_container_width=True)
                st.caption(f"🕒 {item['t']} | 🔴 {item['p']} | 🟢 {item['s']}")
    else:
        st.info("Sistem hazır. Analiz yapmak için bir fotoğraf yükleyin.")

with st.sidebar:
    st.header("🔬 Sistem Durumu")
    st.success("✅ Model ve Scaler otomatik yüklendi.")
    if st.button("Veri Tabanını Sıfırla"):
        st.session_state.db_analiz = []
        st.rerun()
