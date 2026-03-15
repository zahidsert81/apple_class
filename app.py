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

# 1. AYARLAR & URL
st.set_page_config(page_title="Pestisit Analiz Laboratuvarı", page_icon="🧪", layout="wide")

APP_URL = "https://pestisit-kontrol.streamlit.app" 
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

# --- SAYFA MODU ---
query_params = st.query_params
is_mobile = query_params.get("mode") == "mobile"

if is_mobile:
    # 📱 MOBİL PANEL (SADE VE HIZLI)
    st.markdown("### 📱 Mobil Veri Aktarımı")
    m_file = st.file_uploader("Termal Görüntü Gönder", type=["jpg", "png", "jpeg"])
    if m_file:
        img = Image.open(m_file)
        img.save(os.path.join(MOBILE_UPLOAD_DIR, "transfer.png"))
        st.success("✅ Görüntü aktarıldı! Bilgisayardan 'Analiz Et'e basabilirsiniz.")
        st.balloons()
else:
    # 💻 ANA MASAÜSTÜ PANELİ
    col_l, col_m, col_r = st.columns([1, 4, 1])
    with col_l:
        if os.path.exists("TÜBİTAK_logo.svg.png"): st.image("TÜBİTAK_logo.svg.png", width=110)
    with col_m:
        st.markdown("<h1 style='text-align: center; color: #1E3A8A;'>Pestisit Tespit Sistemi</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: #64748B; font-weight: bold;'>Muhammed Zahid SERT | Gıda Güvenliği Analiz Modülü</p>", unsafe_allow_html=True)
    with col_r:
        if os.path.exists("images.jpg"): st.image("images.jpg", width=110)

    st.divider()
    
    # --- YAN MENÜ (SIDEBAR) ---
    with st.sidebar:
        st.header("📲 Mobil Bağlantı")
        qr = qrcode.make(f"{APP_URL}/?mode=mobile")
        buf = BytesIO()
        qr.save(buf, format="PNG")
        st.image(buf, caption="Okut ve Fotoğraf Gönder")
        
        st.divider()
        st.header("⚙️ Kontrol")
        sens = st.slider("Hassasiyet Ayarı", 0.1, 0.9, 0.45)
        
        # MOBİL VERİ AKTARIM KONTROLÜ
        st.subheader("📥 Mobil Veri Havuzu")
        m_path = os.path.join(MOBILE_UPLOAD_DIR, "transfer.png")
        if os.path.exists(m_path):
            st.warning("Yeni veri bekliyor!")
            if st.button("📥 Veriyi Analiz Havuzuna Al"):
                # Mobilden geleni ana havuzun göreceği yere kopyala
                img_tmp = Image.open(m_path)
                img_tmp.save(os.path.join(SAVE_DIR, "mobile_current.png"))
                st.success("Veri alındı!")
                st.rerun()
        else:
            st.info("Mobil veri yok.")
            
        if st.button("🔄 Sistemi Yenile"):
            st.rerun()

    # --- ANA ANALİZ ALANI ---
    # Hem manuel yüklemeyi hem de mobilden gelen havuzu kontrol et
    up_file = st.file_uploader("Bilgisayardan Yükle", type=["jpg", "png", "jpeg"])
    
    # Analiz edilecek dosyayı seç
    source_img = None
    if up_file:
        source_img = Image.open(up_file)
    elif os.path.exists(os.path.join(SAVE_DIR, "mobile_current.png")):
        st.success("✅ Mobil havuzdaki görüntü analiz ediliyor.")
        source_img = Image.open(os.path.join(SAVE_DIR, "mobile_current.png"))

    if source_img and assets:
        model, scaler = assets
        with st.spinner("Analiz Yapılıyor..."):
            p_img = source_img.convert("RGB")
            nb = remove(p_img).convert("RGB")
            gray = cv2.cvtColor(np.array(nb), cv2.COLOR_RGB2GRAY)
            
            _, mask = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            res = np.array(p_img).copy()
            results = []

            for cnt in contours:
                if cv2.contourArea(cnt) < 1800: continue
                x,y,w,h = cv2.boundingRect(cnt)
                crop = gray[y:y+h, x:x+w]
                
                f_sc = scaler.transform(extract_features(crop))
                probs = model.predict_proba(f_sc)[0]
                
                pred = 1 if probs[1] >= sens else 0
                lbl = "PESTISITLI" if pred == 1 else "TEMIZ"
                clr = (255, 0, 0) if pred == 1 else (0, 255, 0)
                
                results.append({"Durum": lbl, "Güven": f"%{max(probs)*100:.1f}"})
                cv2.rectangle(res, (x,y), (x+w,y+h), clr, 12)
                cv2.putText(res, lbl, (x, y-15), cv2.FONT_HERSHEY_SIMPLEX, 1.4, clr, 4)

            st.image(res, use_container_width=True)
            if results:
                cols = st.columns(len(results))
                for i, r in enumerate(results):
                    cols[i].metric(f"Örnek {i+1}", r["Durum"], r["Güven"])
            
            # Analiz bitince mobil havuzu temizlemek istersen:
            if st.button("🗑️ Analizi Tamamla ve Havuzu Boşalt"):
                if os.path.exists(os.path.join(SAVE_DIR, "mobile_current.png")):
                    os.remove(os.path.join(SAVE_DIR, "mobile_current.png"))
                if os.path.exists(os.path.join(MOBILE_UPLOAD_DIR, "transfer.png")):
                    os.remove(os.path.join(MOBILE_UPLOAD_DIR, "transfer.png"))
                st.rerun()

    elif not assets:
        st.error("rf_model.pkl veya scaler.pkl bulunamadı!")
