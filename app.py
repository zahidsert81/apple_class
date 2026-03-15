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

# 1. SAYFA YAPILANDIRMASI
st.set_page_config(page_title="Pestisit Analiz Laboratuvarı", page_icon="🍎", layout="wide")

# Klasörleri Hazırla
SAVE_DIR = "analiz_havuzu"
MOBILE_UPLOAD_DIR = "mobil_aktarim"
MODEL_FILE = "rf_model.pkl"
SCALER_FILE = "scaler.pkl"

for d in [SAVE_DIR, MOBILE_UPLOAD_DIR]:
    if not os.path.exists(d): 
        os.makedirs(d)

# 2. YEREL IP TESPİTİ (Telefonun bağlanabilmesi için)
def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except: 
        return "127.0.0.1"

# 3. ÖZELLİK ÇIKARICI (Senin Orijinal 11 Özellikli Yapın)
def extract_features(img_gray):
    img_gray = cv2.medianBlur(img_gray, 5)
    img_8 = cv2.normalize(img_gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # GLCM (6)
    glcm = graycomatrix(img_8, distances=[1], angles=[0], symmetric=True, normed=True)
    glcm_feats = [graycoprops(glcm, p)[0, 0] for p in ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']]
    
    # İstatistik (4)
    flat = img_gray.flatten()
    stats = [np.mean(flat), np.std(flat), skew(flat), kurtosis(flat)]
    
    # Kenar (1)
    edges = cv2.Canny(img_8, 50, 150)
    density = np.sum(edges > 0) / edges.size
    
    return np.array(glcm_feats + stats + [density]).reshape(1, -1)

# 4. VARLIKLARI YÜKLE (Cache ile hızlandırılmış)
@st.cache_resource
def load_assets():
    if os.path.exists(MODEL_FILE) and os.path.exists(SCALER_FILE):
        with open(MODEL_FILE, "rb") as f: m = pickle.load(f)
        with open(SCALER_FILE, "rb") as f: s = pickle.load(f)
        return m, s
    return None

assets = load_assets()

# --- MOD SEÇİMİ (Mobil mi, Masaüstü mü?) ---
query_params = st.query_params
is_mobile = query_params.get("mode") == "mobile"

if is_mobile:
    # 📱 TELEFONDA GÖRÜNECEK SAYFA
    st.markdown("<h2 style='text-align: center; color: #1E3A8A;'>📱 Mobil Veri Aktarımı</h2>", unsafe_allow_html=True)
    st.write("Termal kameradan aldığınız görüntüyü aşağıdan yükleyin.")
    
    m_file = st.file_uploader("Dosya Seç veya Fotoğraf Çek", type=["jpg", "png", "jpeg"])
    if m_file:
        img = Image.open(m_file)
        # Bilgisayara 'transfer.png' adıyla kaydet
        img.save(os.path.join(MOBILE_UPLOAD_DIR, "transfer.png"))
        st.success("✅ Görüntü başarıyla bilgisayara gönderildi. Masaüstü ekranını kontrol edin.")
        st.balloons()
else:
    # 💻 BİLGİSAYARDA GÖRÜNECEK ANA SAYFA
    
    # Üst Bilgi (Logolar ve Başlık)
    col_l, col_m, col_r = st.columns([1, 4, 1])
    with col_l:
        if os.path.exists("TÜBİTAK_logo.svg.png"): st.image("TÜBİTAK_logo.svg.png", width=110)
    with col_m:
        st.markdown("<h1 style='text-align: center; color: #1E3A8A; margin-bottom: 0;'>Pestisit Analiz Sistemi</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: #64748B;'>Muhammed Zahid SERT | Gıda Güvenliği Modülü</p>", unsafe_allow_html=True)
    with col_r:
        if os.path.exists("images.jpg"): st.image("images.jpg", width=110)

    st.divider()

    # Yan Panel ve Ana Alan
    main_col, side_col = st.columns([3, 1])

    with side_col:
        st.subheader("📲 Mobil Bağlantı")
        local_ip = get_local_ip()
        # Streamlit URL'sini oluştur
        m_url = f"http://{local_ip}:8501/?mode=mobile"
        
        qr_img = qrcode.make(m_url)
        buf = BytesIO()
        qr_img.save(buf, format="PNG")
        st.image(buf, caption="Bağlanmak için okutun")
        st.caption(f"Adres: {m_url}")
        
        st.divider()
        st.subheader("⚙️ Ayarlar")
        sens = st.slider("Analiz Hassasiyeti", 0.1, 0.9, 0.45)
        st.info("Not: Bu ayar modelin karar eşiğini belirler.")

    with main_col:
        m_path = os.path.join(MOBILE_UPLOAD_DIR, "transfer.png")
        
        # Mobil veri kontrolü
        if os.path.exists(m_path):
            st.info("📱 Telefondan yeni bir görüntü transfer edildi.")
            col_btn1, col_btn2 = st.columns(2)
            if col_btn1.button("✅ Gelen Veriyi Analiz Et"):
                source_img = Image.open(m_path)
                # Analiz tetiklenecek
            if col_btn2.button("🗑️ Aktarımı İptal Et / Sil"):
                os.remove(m_path)
                st.rerun()
        
        # Manuel Yükleme
        up_file = st.file_uploader("Veya Bilgisayardan Veri Yükleyin", type=["jpg", "png", "jpeg"])
        
        # Analiz Başlatma
        target = None
        if 'source_img' in locals() and source_img: target = source_img
        elif up_file: target = Image.open(up_file)

        if target and assets:
            model, scaler = assets
            with st.spinner("Analiz Yapılıyor..."):
                # Görüntü İşleme
                p_img = target.convert("RGB")
                nb = remove(p_img).convert("RGB")
                gray = cv2.cvtColor(np.array(nb), cv2.COLOR_RGB2GRAY)
                
                _, mask = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                res = np.array(p_img).copy()
                results_list = []

                for cnt in contours:
                    if cv2.contourArea(cnt) < 1800: continue
                    x,y,w,h = cv2.boundingRect(cnt)
                    crop = gray[y:y+h, x:x+w]
                    
                    # Tahmin
                    feats = extract_features(crop)
                    f_sc = scaler.transform(feats)
                    probs = model.predict_proba(f_sc)[0]
                    
                    pred = 1 if probs[1] >= sens else 0
                    lbl = "PESTISITLI" if pred == 1 else "TEMIZ"
                    clr = (255, 0, 0) if pred == 1 else (0, 255, 0)
                    
                    results_list.append({"Durum": lbl, "Güven": f"%{max(probs)*100:.1f}"})
                    cv2.rectangle(res, (x,y), (x+w,y+h), clr, 12)
                    cv2.putText(res, lbl, (x, y-15), cv2.FONT_HERSHEY_SIMPLEX, 1.4, clr, 4)

                # Raporlama
                st.image(res, caption="Laboratuvar Teşhisi", use_container_width=True)
                
                if results_list:
                    c = st.columns(len(results_list))
                    for i, r in enumerate(results_list):
                        c[i].metric(f"Örnek {i+1}", r["Durum"], r["Güven"])

        elif not assets:
            st.warning("Lütfen model dosyalarınızı (rf_model.pkl ve scaler.pkl) kontrol edin.")
