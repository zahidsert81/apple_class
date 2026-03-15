import streamlit as st
import os
import time
import qrcode
import zipfile
from io import BytesIO
from PIL import Image
from rembg import remove
import cv2
import numpy as np
import pickle
from datetime import datetime
from scipy.stats import skew, kurtosis
from skimage.feature import graycomatrix, graycoprops

# 1. TEMEL AYARLAR
st.set_page_config(page_title="Pestisit Analiz Laboratuvarı", page_icon="🧪", layout="wide")
APP_URL = "https://pestisit-kontrol.streamlit.app" 
SAVE_DIR = "analiz_havuzu"
MOBILE_UPLOAD_DIR = "mobil_aktarim"
MODEL_FILE = "rf_model.pkl"
SCALER_FILE = "scaler.pkl"
ADMIN_PASSWORD = "3681"

for d in [SAVE_DIR, MOBILE_UPLOAD_DIR]:
    if not os.path.exists(d): os.makedirs(d)

# 2. ÖZELLİK ÇIKARMA
def extract_features(img_gray):
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

# --- MOBİL AKTARIM SAYFASI (LocalSend Mantığı) ---
if st.query_params.get("mode") == "mobile":
    st.markdown("### 📤 Hızlı Veri Aktarımı (LocalSend Mode)")
    st.info("Kameranızı açın veya galeriden termal görüntüyü seçin.")
    f = st.file_uploader("Dosya Seç", type=["jpg","png","jpeg"], label_visibility="collapsed")
    if f:
        with st.spinner("Aktarılıyor..."):
            Image.open(f).save(os.path.join(MOBILE_UPLOAD_DIR, "transfer.png"))
            st.success("✅ Veri saniyeler içinde ana sisteme iletildi!")
            st.balloons()
    st.stop()

# --- MASAÜSTÜ ANA PANEL ---
col_l, col_m, col_r = st.columns([1, 3, 1])
with col_l:
    if os.path.exists("TÜBİTAK_logo.svg.png"): st.image("TÜBİTAK_logo.svg.png", width=100)
with col_m:
    st.markdown("<h1 style='text-align: center; color: #1E3A8A;'>Pestisit Tespit Sistemi</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #64748B; font-weight: bold;'>Muhammed Zahid SERT | Gıda Güvenliği Modülü</p>", unsafe_allow_html=True)
with col_r:
    if os.path.exists("images.jpg"): st.image("images.jpg", width=100)

st.divider()

# --- YAN MENÜ (Dinamik Kontrol) ---
with st.sidebar:
    st.header("🔌 Bağlantı & Kontrol")
    
    # Bluetooth Simülasyonu
    if "bt_connected" not in st.session_state: st.session_state.bt_connected = False
    if not st.session_state.bt_connected:
        if st.button("🔵 Cihaza Bağlan (Bluetooth)"):
            with st.status("Cihaz aranıyor...", expanded=False) as s:
                time.sleep(1); s.write("🔍 Pestisit_Scanner bulundu..."); time.sleep(1)
                st.session_state.bt_connected = True
                st.rerun()
    else:
        st.success("📱 Cihaz Bağlı: Bluetooth V1")
        if st.button("🔴 Bağlantıyı Kes"):
            st.session_state.bt_connected = False
            st.rerun()

    st.divider()
    
    # LocalSend Butonu
    if st.button("🚀 Veri Gönder / Yükle"):
        st.session_state.show_up = not st.session_state.get("show_up", False)

    st.divider()
    st.subheader("📲 Hızlı Bağlantı QR")
    qr = qrcode.make(f"{APP_URL}/?mode=mobile")
    b = BytesIO(); qr.save(b, format="PNG")
    st.image(b, caption="Telefonundan Dosya Gönder")
    
    sens = st.slider("Hassasiyet Ayarı", 0.1, 0.9, 0.45)

# --- ANA ANALİZ AKIŞI ---
m_path = os.path.join(MOBILE_UPLOAD_DIR, "transfer.png")

if st.session_state.get("show_up"):
    c1, c2 = st.columns(2)
    with c1:
        local_up = st.file_uploader("Bilgisayardan Yükle", type=["jpg","png","jpeg"])
        if local_up: 
            Image.open(local_up).save(os.path.join(SAVE_DIR, "current.png"))
            st.rerun()
    with c2:
        if os.path.exists(m_path):
            st.markdown("#### 📱 Telefondan Veri Geldi!")
            st.image(m_path, width=150)
            if st.button("📥 Veriyi Analize Al"):
                os.rename(m_path, os.path.join(SAVE_DIR, "current.png"))
                st.rerun()

# Analiz Ekranı
cur_path = os.path.join(SAVE_DIR, "current.png")
if os.path.exists(cur_path):
    st.subheader("🖼️ Analiz Hazır")
    img_ready = Image.open(cur_path)
    st.image(img_ready, width=400)
    
    if st.button("🔍 ANALİZİ BAŞLAT"):
        if assets:
            model, scaler = assets
            with st.spinner("Yapay Zeka İnceliyor..."):
                p_img = img_ready.convert("RGB")
                nb = remove(p_img).convert("RGB")
                gray = cv2.cvtColor(np.array(nb), cv2.COLOR_RGB2GRAY)
                _, mask = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                res_img = np.array(p_img).copy()
                p_count, s_count = 0, 0
                for cnt in contours:
                    if cv2.contourArea(cnt) < 1800: continue
                    x,y,w,h = cv2.boundingRect(cnt)
                    f_sc = scaler.transform(extract_features(gray[y:y+h, x:x+w]))
                    pred = 1 if model.predict_proba(f_sc)[0][1] >= sens else 0
                    
                    lbl, clr = ("PESTISITLI", (255,0,0)) if pred == 1 else ("TEMIZ", (0,255,0))
                    if pred == 1: p_count += 1
                    else: s_count += 1
                    cv2.rectangle(res_img, (x,y), (x+w,y+h), clr, 12)
                    cv2.putText(res_img, lbl, (x, y-15), cv2.FONT_HERSHEY_SIMPLEX, 1.4, clr, 4)

                st.image(res_img, use_container_width=True)
                
                # ARŞİVE AT
                now = datetime.now().strftime("%H-%M-%S")
                save_name = f"ARŞİV_{now}_P{p_count}_S{s_count}.png"
                Image.fromarray(res_img).save(os.path.join(SAVE_DIR, save_name))
                os.remove(cur_path)
                st.success(f"Analiz tamamlandı. Arşive eklendi: {now}")

# --- ŞİFRELİ ARŞİV ---
st.divider()
with st.expander("📂 Kayıtlı Verilere Bak (Şifre: 3681)"):
    pw = st.text_input("Giriş Yap:", type="password")
    if pw == ADMIN_PASSWORD:
        files = sorted([f for f in os.listdir(SAVE_DIR) if f.startswith("ARŞİV_")], reverse=True)
        if files:
            z_buf = BytesIO()
            with zipfile.ZipFile(z_buf, "w") as zf:
                for f in files: zf.write(os.path.join(SAVE_DIR, f), f)
            st.download_button("📥 Tüm Arşivi İndir (ZIP)", z_buf.getvalue(), "analiz_arsivi.zip")
            
            for f in files:
                col1, col2 = st.columns([4, 1])
                col1.write(f"📁 {f}")
                with col2:
                    with open(os.path.join(SAVE_DIR, f), "rb") as file:
                        st.download_button("💾", file, file_name=f, key=f)
        else: st.info("Arşiv boş.")
