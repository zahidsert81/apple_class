import streamlit as st
import os
import time
import numpy as np
import cv2
import pickle
import qrcode
import zipfile
from io import BytesIO
from PIL import Image
from rembg import remove
from skimage.feature import graycomatrix, graycoprops
from scipy.stats import skew, kurtosis
from datetime import datetime

# 1. TEMEL AYARLAR & GÜVENLİK
st.set_page_config(page_title="Pestisit Analiz Laboratuvarı", page_icon="🧪", layout="wide")

APP_URL = "https://pestisit-kontrol.streamlit.app" 
SAVE_DIR = "analiz_havuzu"
MOBILE_UPLOAD_DIR = "mobil_aktarim"
MODEL_FILE = "rf_model.pkl"
SCALER_FILE = "scaler.pkl"
ADMIN_PASSWORD = "3681" # Belirlediğin şifre

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

# --- SAYFA MODU (MOBİL) ---
query_params = st.query_params
if query_params.get("mode") == "mobile":
    st.markdown("### 📱 Mobil Veri Aktarımı")
    m_file = st.file_uploader("Fotoğraf Seç veya Çek", type=["jpg", "png", "jpeg"])
    if m_file:
        img = Image.open(m_file)
        img.save(os.path.join(MOBILE_UPLOAD_DIR, "transfer.png"))
        st.success("✅ Görüntü aktarıldı!")
    st.stop()

# --- MASAÜSTÜ ARAYÜZÜ ---
col_l, col_m, col_r = st.columns([1, 4, 1])
with col_l:
    if os.path.exists("TÜBİTAK_logo.svg.png"): st.image("TÜBİTAK_logo.svg.png", width=110)
with col_m:
    st.markdown("<h1 style='text-align: center; color: #1E3A8A;'>Pestisit Tespit Sistemi</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #64748B; font-weight: bold;'>Muhammed Zahid SERT | Gıda Güvenliği Modülü</p>", unsafe_allow_html=True)
with col_r:
    if os.path.exists("images.jpg"): st.image("images.jpg", width=110)

st.divider()

# --- YAN MENÜ (SIDEBAR) ---
with st.sidebar:
    st.header("🛂 Kontrol Paneli")
    
    # "Veri Gönder" butonu ile yükleme alanını aç/kapat (Session State kullanarak yenilemeden yapıyoruz)
    if "show_upload" not in st.session_state:
        st.session_state.show_upload = False

    if st.button("📤 Veri Gönder / Yükle"):
        st.session_state.show_upload = not st.session_state.show_upload

    st.divider()
    st.subheader("📲 Mobil Bağlantı")
    qr = qrcode.make(f"{APP_URL}/?mode=mobile")
    buf = BytesIO()
    qr.save(buf, format="PNG")
    st.image(buf, caption="Okut ve Fotoğraf Gönder")
    
    st.divider()
    sens = st.slider("Hassasiyet Ayarı", 0.1, 0.9, 0.45)

# --- ANA ALAN ---
source_img = None

# Eğer yan menüden "Veri Gönder" seçildiyse yükleme alanını göster
if st.session_state.show_upload:
    st.info("💡 Buradan bilgisayarınızdan veya mobilden gelen veriyi seçebilirsiniz.")
    up_file = st.file_uploader("Bilgisayardan Dosya Seç", type=["jpg", "png", "jpeg"])
    
    m_path = os.path.join(MOBILE_UPLOAD_DIR, "transfer.png")
    if os.path.exists(m_path):
        if st.button("📥 Mobilden Gelen Veriyi Kullan"):
            source_img = Image.open(m_path)
            # Analiz edilecek geçici bir yere kaydet
            source_img.save(os.path.join(SAVE_DIR, "temp_current.png"))
    
    if up_file:
        source_img = Image.open(up_file)
        source_img.save(os.path.join(SAVE_DIR, "temp_current.png"))

# Analiz Süreci
if os.path.exists(os.path.join(SAVE_DIR, "temp_current.png")):
    source_img = Image.open(os.path.join(SAVE_DIR, "temp_current.png"))
    
    if st.button("🔍 Analizi Başlat"):
        if assets:
            model, scaler = assets
            with st.spinner("YZ Analizi Yapılıyor..."):
                p_img = source_img.convert("RGB")
                nb = remove(p_img).convert("RGB")
                gray = cv2.cvtColor(np.array(nb), cv2.COLOR_RGB2GRAY)
                _, mask = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                res = np.array(p_img).copy()
                p_count, s_count = 0, 0
                
                for cnt in contours:
                    if cv2.contourArea(cnt) < 1800: continue
                    x,y,w,h = cv2.boundingRect(cnt)
                    f_sc = scaler.transform(extract_features(gray[y:y+h, x:x+w]))
                    prob = model.predict_proba(f_sc)[0][1]
                    pred = 1 if prob >= sens else 0
                    
                    lbl = "PESTISITLI" if pred == 1 else "TEMIZ"
                    clr = (255, 0, 0) if pred == 1 else (0, 255, 0)
                    if pred == 1: p_count += 1
                    else: s_count += 1
                    cv2.rectangle(res, (x,y), (x+w,y+h), clr, 12)
                    cv2.putText(res, lbl, (x, y-15), cv2.FONT_HERSHEY_SIMPLEX, 1.4, clr, 4)

                st.image(res, caption="Analiz Sonucu", use_container_width=True)
                
                # ARŞİVE KAYDET (Tarih ve Saat ile)
                now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                save_name = f"ARŞİV_{now}_P{p_count}_S{s_count}.png"
                Image.fromarray(res).save(os.path.join(SAVE_DIR, save_name))
                st.success(f"Analiz başarıyla arşive eklendi: {now}")
                
                # Temizlik
                os.remove(os.path.join(SAVE_DIR, "temp_current.png"))
                if os.path.exists(m_path): os.remove(m_path)

st.divider()

# --- ŞİFRELİ ARŞİV BÖLÜMÜ ---
st.subheader("📂 Veri Arşivi (Şifreli Erişim)")
with st.expander("Arşivi Görüntülemek İçin Tıklayın"):
    pwd = st.text_input("Yönetici Şifresi:", type="password")
    
    if pwd == ADMIN_PASSWORD:
        st.success("Erişim Onaylandı.")
        files = sorted([f for f in os.listdir(SAVE_DIR) if f.startswith("ARŞİV_")], reverse=True)
        
        if files:
            # ZIP OLUŞTURMA
            zip_buffer = BytesIO()
            with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
                for f in files:
                    zip_file.write(os.path.join(SAVE_DIR, f), f)
            
            st.download_button(
                label="📥 Tüm Analizleri ZIP Olarak İndir",
                data=zip_buffer.getvalue(),
                file_name="pestisit_rapor_arsivi.zip",
                mime="application/zip"
            )

            # Liste Görünümü
            for f in files:
                # Dosya isminden tarih ve sonucu ayıkla
                parts = f.replace(".png", "").split("_")
                tarih = parts[1]
                saat = parts[2].replace("-", ":")
                sonuc = f"Pestisitli: {parts[3][1:]}, Sağlıklı: {parts[4][1:]}"
                
                col1, col2, col3 = st.columns([2, 3, 1])
                col1.write(f"📅 {tarih} | 🕒 {saat}")
                col2.write(f"📊 {sonuc}")
                with col3:
                    with open(os.path.join(SAVE_DIR, f), "rb") as file:
                        st.download_button("💾 İndir", file, file_name=f, key=f)
                st.divider()
        else:
            st.info("Henüz kaydedilmiş bir analiz bulunmamaktadır.")
    elif pwd != "":
        st.error("Hatalı Şifre!")
