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
import socket

# 1. SAYFA VE DİZİN AYARLARI
st.set_page_config(page_title="Pestisit Analiz Laboratuvarı", page_icon="🧪", layout="wide")

SAVE_DIR = "analiz_havuzu"
MOBILE_UPLOAD_DIR = "mobil_aktarim"
MODEL_FILE = "rf_model.pkl"
SCALER_FILE = "scaler.pkl"

for d in [SAVE_DIR, MOBILE_UPLOAD_DIR]:
    if not os.path.exists(d): os.makedirs(d)

# 2. YEREL IP ADRESİNİ BULMA (QR Kod İçin)
def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except: return "localhost"

# 3. ÖZELLİK ÇIKARMA (Eski Model Uyumlu 11 Özellik)
def extract_features(img_gray):
    img_gray = cv2.medianBlur(img_gray, 5)
    img_8 = cv2.normalize(img_gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    glcm = graycomatrix(img_8, distances=[1], angles=[0], symmetric=True, normed=True)
    glcm_features = [graycoprops(glcm, p)[0, 0] for p in ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']]
    flat = img_gray.flatten()
    stats = [np.mean(flat), np.std(flat), skew(flat), kurtosis(flat)]
    edges = cv2.Canny(img_8, 50, 150)
    edge_density = np.sum(edges > 0) / edges.size
    return np.array(glcm_features + stats + [edge_density]).reshape(1, -1)

# 4. VARLIKLARI YÜKLE
@st.cache_resource
def load_assets():
    if os.path.exists(MODEL_FILE) and os.path.exists(SCALER_FILE):
        with open(MODEL_FILE, "rb") as f: m = pickle.load(f)
        with open(SCALER_FILE, "rb") as f: s = pickle.load(f)
        return m, s
    return None

assets = load_assets()

# --- SAYFA YÖNLENDİRMESİ (MOBİL/MASAÜSTÜ) ---
query_params = st.query_params
is_mobile = query_params.get("mode") == "mobile"

if is_mobile:
    # 📱 MOBİL YÜKLEME ARAYÜZÜ
    st.markdown("<h2 style='text-align: center; color: #1E3A8A;'>📱 Mobil Aktarım Paneli</h2>", unsafe_allow_html=True)
    st.info("Kameranızı açın veya termal fotoğrafı seçin.")
    
    mobile_file = st.file_uploader("Fotoğraf Gönder", type=["jpg", "png", "jpeg"])
    if mobile_file:
        img = Image.open(mobile_file)
        img.save(os.path.join(MOBILE_UPLOAD_DIR, "transfer.png"))
        st.success("✅ Fotoğraf başarıyla bilgisayara aktarıldı!")
        st.balloons()

else:
    # 💻 ANA BİLGİSAYAR ARAYÜZÜ
    
    # ÜST PANEL (LOGOLAR)
    col_l, col_m, col_r = st.columns([1, 4, 1])
    with col_l:
        if os.path.exists("TÜBİTAK_logo.svg.png"): st.image("TÜBİTAK_logo.svg.png", width=120)
    with col_m:
        st.markdown("<h1 style='text-align: center; color: #1E3A8A; margin-bottom: 0;'>Pestisit Tespit Sistemi</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: #64748B; font-weight: bold;'>Muhammed Zahid SERT | Gıda Güvenliği Analiz Modülü</p>", unsafe_allow_html=True)
    with col_r:
        if os.path.exists("images.jpg"): st.image("images.jpg", width=120)

    st.divider()

    # ANA GÖVDE
    col_left, col_right = st.columns([3, 1])

    with col_right:
        st.markdown("### 📲 Telefon Bağlantısı")
        ip = get_local_ip()
        # Not: Streamlit varsayılan 8501 portunu kullanır
        mobile_url = f"http://{ip}:8501/?mode=mobile"
        
        qr = qrcode.make(mobile_url)
        buf = BytesIO()
        qr.save(buf, format="PNG")
        st.image(buf, caption="QR Kodu Okutun")
        st.caption(f"Manuel Erişim: {mobile_url}")
        
        st.divider()
        sens = st.slider("Hassasiyet Ayarı", 0.1, 0.9, 0.45)
        st.info("Eğer her elmaya temiz diyorsa bu ayarı düşürün.")

    with col_left:
        # Mobil dosya kontrolü
        mobile_path = os.path.join(MOBILE_UPLOAD_DIR, "transfer.png")
        
        source_img = None
        if os.path.exists(mobile_path):
            st.success("📱 Mobilden yeni bir veri alındı!")
            source_img = Image.open(mobile_path)
            if st.button("📱 Gelen Fotoğrafı Analiz Et"):
                # Analiz tetiklenecek
                pass 
            if st.button("🗑️ Mobilden Gelen Veriyi Sil"):
                os.remove(mobile_path)
                st.rerun()
        
        st.subheader("🖼️ Analiz Paneli")
        uploaded = st.file_uploader("Veya Bilgisayardan Dosya Seçin", type=["jpg", "png", "jpeg"])
        
        if uploaded: source_img = Image.open(uploaded)

        if source_img and assets:
            model, scaler = assets
            with st.spinner("YZ Termal Verileri İşliyor..."):
                # Görüntü İşleme
                pil_img = source_img.convert("RGB")
                nobg = remove(pil_img).convert("RGB")
                gray = cv2.cvtColor(np.array(nobg), cv2.COLOR_RGB2GRAY)
                
                _, mask = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                res_img = np.array(pil_img).copy()
                detaylar = []

                for cnt in contours:
                    if cv2.contourArea(cnt) < 1500: continue
                    x,y,w,h = cv2.boundingRect(cnt)
                    crop = gray[y:y+h, x:x+w]
                    
                    feats = extract_features(crop)
                    f_scaled = scaler.transform(feats)
                    probs = model.predict_proba(f_scaled)[0]
                    
                    pred = 1 if probs[1] >= sens else 0
                    label = "PESTISITLI" if pred == 1 else "PESISITSIZ"
                    color = (255, 0, 0) if pred == 1 else (0, 255, 0)
                    
                    detaylar.append({"Durum": label, "Güven": f"%{max(probs)*100:.1f}"})
                    cv2.rectangle(res_img, (x,y), (x+w,y+h), color, 12)
                    cv2.putText(res_img, label, (x, y-15), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 4)

                # SONUÇ GÖSTERİMİ
                st.image(res_img, use_container_width=True)
                
                if detaylar:
                    cols = st.columns(len(detaylar))
                    for i, d in enumerate(detaylar):
                        with cols[i]:
                            st.metric(f"Örnek {i+1}", d["Durum"], d["Güven"])

        elif not assets:
            st.error("⚠️ Model dosyaları eksik! rf_model.pkl ve scaler.pkl'yi yükleyin.")
