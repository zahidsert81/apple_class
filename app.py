import streamlit as st
import os
import time
import numpy as np
import cv2
import pickle
import glob
import pandas as pd
from PIL import Image
from rembg import remove
from skimage.feature import graycomatrix, graycoprops
from scipy.stats import skew, kurtosis

# 1. SAYFA AYARLARI
st.set_page_config(page_title="Pestisit Tespit Sistemi", page_icon="🍎", layout="wide")

# 2. AYARLAR & DİZİNLER
ADMIN_PASSWORD = "1234" 
SAVE_DIR = "analiz_havuzu"
MODEL_FILE = "rf_model.pkl"
SCALER_FILE = "scaler.pkl"

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

# 3. ESKİ MODELLE UYUMLU ÖZELLİK ÇIKARMA (Tam 11 Özellik)
def extract_features(img_gray):
    # Gürültü giderme
    img_gray = cv2.medianBlur(img_gray, 5)
    img_8 = cv2.normalize(img_gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Doku Analizi (GLCM) - 6 Özellik
    glcm = graycomatrix(img_8, distances=[1], angles=[0], symmetric=True, normed=True)
    glcm_features = [
        graycoprops(glcm, 'contrast')[0, 0],
        graycoprops(glcm, 'dissimilarity')[0, 0],
        graycoprops(glcm, 'homogeneity')[0, 0],
        graycoprops(glcm, 'energy')[0, 0],
        graycoprops(glcm, 'correlation')[0, 0],
        graycoprops(glcm, 'ASM')[0, 0]
    ]
    
    # İstatistiksel Özellikler - 4 Özellik
    flat = img_gray.flatten()
    stats = [
        np.mean(flat), 
        np.std(flat), 
        skew(flat), 
        kurtosis(flat)
    ]
    
    # Kenar Yoğunluğu - 1 Özellik
    edges = cv2.Canny(img_8, 50, 150)
    edge_density = np.sum(edges > 0) / edges.size
    
    # Toplam: 11 Özellik
    full_features = glcm_features + stats + [edge_density]
    return np.array(full_features).reshape(1, -1)

# 4. ASSET YÜKLEME
@st.cache_resource
def load_assets():
    if os.path.exists(MODEL_FILE) and os.path.exists(SCALER_FILE):
        with open(MODEL_FILE, "rb") as f: m = pickle.load(f)
        with open(SCALER_FILE, "rb") as f: s = pickle.load(f)
        return m, s
    return None

# 5. KURUMSAL ÜST PANEL
col_l, col_c, col_r = st.columns([1, 4, 1])
with col_l:
    if os.path.exists("TÜBİTAK_logo.svg.png"): st.image("TÜBİTAK_logo.svg.png", width=120)
with col_c:
    st.markdown("<h1 style='text-align: center; color: #1E3A8A;'>PESTİSİT TESPİT SİSTEMİ</h1>", unsafe_allow_html=True)
    st.markdown(f"<p style='text-align: center; color: #64748B;'>Muhammed Zahid SERT | Analiz ve Raporlama</p>", unsafe_allow_html=True)
with col_r:
    if os.path.exists("images.jpg"): st.image("images.jpg", width=120)

assets = load_assets()

if not assets:
    st.error("⚠️ Sistem Hatası: Eski model dosyaları (.pkl) bulunamadı!")
else:
    model, scaler = assets
    
    # --- YAN PANEL ---
    with st.sidebar:
        st.header("⚙️ Sistem Ayarları")
        user_name = st.text_input("Operatör:", "Muhammed")
        st.divider()
        if 'admin' not in st.session_state: st.session_state.admin = False
        
        if not st.session_state.admin:
            pw = st.text_input("Yönetici Şifresi:", type="password")
            if st.button("Giriş"):
                if pw == ADMIN_PASSWORD: st.session_state.admin = True; st.rerun()
        else:
            st.success("Yönetici Yetkisi Aktif")
            if st.button("Çıkış Yap"): st.session_state.admin = False; st.rerun()

    # --- ANA ANALİZ AKIŞI ---
    uploaded = st.file_uploader("Termal Görüntü Yükleyin", type=["jpg", "png", "jpeg"])

    if uploaded:
        pil_img = Image.open(uploaded).convert("RGB")
        with st.spinner("Eski Model Verileri İşliyor..."):
            # Arka plan temizleme ve maskeleme
            nobg = remove(pil_img).convert("RGB")
            img_np = np.array(nobg)
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            
            # Maske oluştur (Isı haritası gürültüsünü önlemek için)
            _, mask = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)
            heatmap = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
            heatmap = cv2.bitwise_and(heatmap, heatmap, mask=mask)
            
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            res_img = np.array(pil_img).copy()
            detaylar = []
            
            for cnt in contours:
                if cv2.contourArea(cnt) < 1500: continue
                x, y, w, h = cv2.boundingRect(cnt)
                crop = gray[y:y+h, x:x+w]
                
                # Özellik Çıkarma ve Tahmin
                try:
                    feats = extract_features(crop)
                    f_scaled = scaler.transform(feats)
                    pred = model.predict(f_scaled)[0]
                    prob = model.predict_proba(f_scaled)[0]
                    
                    label = "PESTISITLI" if pred == 1 else "TEMIZ"
                    color = (0, 0, 255) if pred == 1 else (0, 255, 0)
                    
                    detaylar.append({
                        "Nesne": f"Elma {len(detaylar)+1}",
                        "Durum": label,
                        "Güven": f"%{max(prob)*100:.1f}",
                        "Renk": "#fee2e2" if pred == 1 else "#dcfce7",
                        "Border": "#991b1b" if pred == 1 else "#166534"
                    })
                    
                    # Görsel üzerine çizim (Türkçe karakter içermez - Soru işareti hatası çözümü)
                    cv2.rectangle(res_img, (x, y), (x+w, y+h), color, 10)
                    cv2.putText(res_img, f"#{len(detaylar)} {label}", (x, y-15), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 4)
                except Exception as e:
                    st.error(f"Tahmin hatası: {e}")

            # --- RAPORLAMA ---
            st.subheader("📋 Analiz Raporu")
            c1, c2, c3 = st.columns(3)
            c1.image(uploaded, caption="Ham Veri", use_container_width=True)
            c2.image(heatmap, caption="Maskelenmiş Isı Haritası", use_container_width=True)
            c3.image(res_img, caption="Eski Model Teşhisi", use_container_width=True)
            
            st.divider()
            
            if detaylar:
                cols = st.columns(len(detaylar))
                for i, d in enumerate(detaylar):
                    with cols[i]:
                        st.markdown(f"""
                            <div style="background-color:{d['Renk']}; padding:20px; border-radius:15px; border:3px solid {d['Border']}; text-align:center;">
                                <h3 style="color:{d['Border']};">{d['Nesne']}</h3>
                                <b style="font-size:22px;">{d['Durum']}</b><br>
                                <span>Güven Oranı: {d['Güven']}</span>
                            </div>
                        """, unsafe_allow_html=True)

            # Kayıt (Zaman damgası ile)
            p_say = sum(1 for d in detaylar if d["Durum"] == "PESTISITLI")
            s_say = len(detaylar) - p_say
            save_path = f"{SAVE_DIR}/{int(time.time())}_{user_name}_{p_say}_{s_say}.png"
            Image.fromarray(res_img).save(save_path)
