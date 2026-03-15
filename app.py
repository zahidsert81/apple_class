import streamlit as st
import os
import time
import numpy as np
import cv2
import pickle
import glob
import pandas as pd
import zipfile
from io import BytesIO
from PIL import Image
from rembg import remove
from skimage.feature import graycomatrix, graycoprops
from scipy.stats import skew, kurtosis

# Grafik kontrolü
try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# 1. SAYFA AYARLARI
st.set_page_config(page_title="Pestisit Analiz Lab V2", page_icon="🧪", layout="wide")

# 2. AYARLAR
ADMIN_PASSWORD = "1234" 
SAVE_DIR = "analiz_havuzu"
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

# 3. ÖZELLİK ÇIKARMA
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

# 4. MODEL YÜKLEME
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

# 5. ÜST PANEL
col1, col2, col3 = st.columns([1, 4, 1])
with col1:
    if os.path.exists("TÜBİTAK_logo.svg.png"): st.image("TÜBİTAK_logo.svg.png", width=110)
with col2:
    st.markdown("<h1 style='text-align: center; color: #1E3A8A;'>Pestisit Tespit ve Raporlama Sistemi</h1>", unsafe_allow_html=True)
    st.markdown(f"<p style='text-align: center; color: #64748B;'>Muhammed Zahid SERT | Laboratuvar Arşiv Sistemi</p>", unsafe_allow_html=True)
with col3:
    if os.path.exists("images.jpg"): st.image("images.jpg", width=110)

assets = load_assets()

if assets:
    model, scaler = assets
    
    with st.sidebar:
        st.header("⚙️ Kontrol Paneli")
        user_name = st.text_input("Analiz Sorumlusu:", placeholder="İsim yazınız...")
        
        st.divider()
        if 'logged_in' not in st.session_state: st.session_state.logged_in = False

        if not st.session_state.logged_in:
            admin_pw = st.text_input("Yönetici Şifresi:", type="password")
            if st.button("Sistemi Aç"):
                if admin_pw == ADMIN_PASSWORD:
                    st.session_state.logged_in = True
                    st.rerun()
        else:
            st.success("Yönetici Yetkisi Aktif")
            if st.button("Çıkış Yap"):
                st.session_state.logged_in = False
                st.rerun()
            if st.button("⚠️ Arşivi Sıfırla"):
                for f in glob.glob(f"{SAVE_DIR}/*.png"): os.remove(f)
                st.rerun()

    # --- ANALİZ MODÜLÜ ---
    uploaded = st.file_uploader("Termal Veri Yükle", type=["jpg", "png", "jpeg"])

    if uploaded and user_name:
        pil_img = Image.open(uploaded).convert("RGB")
        with st.spinner("YZ Analizi Yapılıyor..."):
            nobg = remove(pil_img).convert("RGB")
            gray = cv2.cvtColor(np.array(nobg), cv2.COLOR_RGB2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            _, th = cv2.threshold(blur, 10, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            res_img = np.array(pil_img).copy()
            p_say, s_say = 0, 0
            
            for cnt in contours:
                if cv2.contourArea(cnt) < 1000: continue
                x, y, w, h = cv2.boundingRect(cnt)
                crop = gray[y:y+h, x:x+w]
                f_scaled = scaler.transform(extract_features(crop))
                pred = model.predict(f_scaled)[0]
                
                label = "PESTISIT" if pred == 1 else "TEMIZ"
                color = (255, 0, 0) if pred == 1 else (0, 255, 0)
                p_say += (1 if pred == 1 else 0)
                s_say += (1 if pred == 0 else 0)
                
                cv2.rectangle(res_img, (x, y), (x+w, y+h), color, 8)
                cv2.putText(res_img, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

            # Otomatik Kayıt
            clean_name = "".join(x for x in user_name if x.isalnum())
            timestamp = int(time.time())
            save_path = f"{SAVE_DIR}/{timestamp}_{clean_name}_{p_say}_{s_say}.png"
            Image.fromarray(res_img).save(save_path)

            # Görsel Rapor
            st.subheader("📝 Analiz Bulguları")
            col_a, col_b, col_c = st.columns([1.5, 1.5, 1])
            with col_a: st.image(uploaded, caption="Girdi", use_container_width=True)
            with col_b: st.image(res_img, caption="Tespit", use_container_width=True)
            with col_c:
                total = p_say + s_say
                oran = (p_say / total * 100) if total > 0 else 0
                if oran > 50:
                    st.error(f"🚨 YÜKSEK RİSK: %{oran:.1f}")
                elif oran > 0:
                    st.warning(f"⚠️ DÜŞÜK RİSK: %{oran:.1f}")
                else:
                    st.success("✅ GÜVENLİ ÜRÜN")
                
                if PLOTLY_AVAILABLE and total > 0:
                    fig = go.Figure(data=[go.Pie(labels=['Pestisit', 'Temiz'], values=[p_say, s_say], hole=.4)])
                    fig.update_layout(height=250, margin=dict(l=0,r=0,b=0,t=0), showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)

    # --- YÖNETİCİ ARŞİVİ ---
    if st.session_state.logged_in:
        st.divider()
        st.header("📂 Laboratuvar Arşivi")
        
        files = sorted(glob.glob(f"{SAVE_DIR}/*.png"), key=os.path.getmtime, reverse=True)
        
        if files:
            # 1. TOPLU İŞLEMLER
            col_arch1, col_arch2 = st.columns(2)
            with col_arch1:
                # Arşivi ZIP olarak indirme
                zip_buffer = BytesIO()
                with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
                    for f in files:
                        zip_file.write(f, os.path.basename(f))
                st.download_button("📦 Tüm Arşivi ZIP Olarak İndir", data=zip_buffer.getvalue(), file_name="pestisit_arsiv.zip", mime="application/zip")
            
            # 2. VERİ TABLOSU
            data_list = []
            for f in files:
                p = os.path.basename(f).replace(".png", "").split("_")
                if len(p) >= 4:
                    data_list.append({"Tarih": time.ctime(int(p[0])), "Sorumlu": p[1], "Pestisitli": p[2], "Temiz": p[3]})
            
            st.dataframe(pd.DataFrame(data_list), use_container_width=True)

            # 3. TEKLİ GÖRÜNTÜLEME
            st.write("🔍 Detaylı Görüntü İnceleme")
            for f in files:
                try:
                    fname = os.path.basename(f)
                    with st.expander(f"Kayıt: {fname}"):
                        st.image(f, use_container_width=True)
                        with open(f, "rb") as file:
                            st.download_button("💾 Görseli İndir", data=file, file_name=fname, mime="image/png", key=f"dl_{fname}")
                except: continue
        else:
            st.info("Arşivde veri bulunamadı.")
else:
    st.error("Sistem başlatılamadı (.pkl dosyaları eksik).")
