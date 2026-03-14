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

# Grafik için Plotly kontrolü
try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# 1. SAYFA AYARLARI
st.set_page_config(page_title="Pestisit Analiz Lab", page_icon="🍎", layout="wide")

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

# 5. ARAYÜZ ÜST KISIM
col1, col2, col3 = st.columns([1, 4, 1])
with col1:
    if os.path.exists("TÜBİTAK_logo.svg.png"): st.image("TÜBİTAK_logo.svg.png", width=110)
with col2:
    st.markdown("<h1 style='text-align: center; margin-bottom: 0;'>Termal Analiz Sistemi</h1>", unsafe_allow_html=True)
    st.markdown(f"<p style='text-align: center; color: gray;'>Muhammed Zahid SERT | Danışman: Emine SARI</p>", unsafe_allow_html=True)
with col3:
    if os.path.exists("images.jpg"): st.image("images.jpg", width=110)

assets = load_assets()

if assets is None:
    st.error("Model dosyaları yüklenemedi. Lütfen .pkl dosyalarını kontrol edin.")
else:
    model, scaler = assets
    
    with st.sidebar:
        st.header("🔬 Kontrol")
        user_name = st.text_input("Analiz Yapan:", placeholder="İsim giriniz...")
        
        st.divider()
        if 'logged_in' not in st.session_state: st.session_state.logged_in = False

        if not st.session_state.logged_in:
            admin_pw = st.text_input("Yönetici Şifresi:", type="password")
            if st.button("Giriş"):
                if admin_pw == ADMIN_PASSWORD:
                    st.session_state.logged_in = True
                    st.rerun()
                else:
                    st.error("Hatalı şifre!")
        else:
            st.success("Yönetici Modu")
            if st.button("Çıkış Yap"):
                st.session_state.logged_in = False
                st.rerun()
            if st.button("Havuzu Boşalt"):
                for f in glob.glob(f"{SAVE_DIR}/*.png"): os.remove(f)
                st.rerun()

    # --- ANALİZ BÖLÜMÜ ---
    uploaded = st.file_uploader("Termal Görüntü Seçin", type=["jpg", "png", "jpeg"])

    if uploaded and user_name:
        pil_img = Image.open(uploaded).convert("RGB")
        
        with st.spinner("Analiz ediliyor..."):
            # İşleme
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
                feats = extract_features(crop)
                f_scaled = scaler.transform(feats)
                pred = model.predict(f_scaled)[0]
                
                label = "PESTISITLI" if pred == 1 else "PESTISITSIZ"
                color = (255, 0, 0) if pred == 1 else (0, 255, 0)
                if pred == 1: p_say += 1
                else: s_say += 1
                cv2.rectangle(res_img, (x, y), (x+w, y+h), color, 8)
                cv2.putText(res_img, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

            # Kayıt
            clean_name = "".join(x for x in user_name if x.isalnum())
            save_path = f"{SAVE_DIR}/{int(time.time())}_{clean_name}_{p_say}_{s_say}.png"
            Image.fromarray(res_img).save(save_path)

            # EKRAN ÇIKTISI
            st.subheader("📊 Analiz Raporu")
            c1, c2, c3 = st.columns([1, 1, 1])
            
            with c1: st.image(uploaded, caption="Orijinal", use_container_width=True)
            with c2: st.image(res_img, caption="Tespit", use_container_width=True)
            with c3:
                if PLOTLY_AVAILABLE and (p_say + s_say > 0):
                    fig = go.Figure(data=[go.Pie(labels=['Pestisit', 'Temiz'], values=[p_say, s_say], hole=.3, marker=dict(colors=['#FF4B4B', '#00CC96']))])
                    fig.update_layout(height=300, margin=dict(l=0,r=0,b=0,t=0))
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.metric("🔴 Pestisitli", p_say)
                    st.metric("🟢 Temiz", s_say)

    # --- YÖNETİCİ HAVUZU (GÜNCELLENMİŞ İNDİRME ÖZELLİĞİ) ---
    if st.session_state.logged_in:
        st.divider()
        st.subheader("📁 Arşivlenmiş Analizler ve İndirme")
        files = sorted(glob.glob(f"{SAVE_DIR}/*.png"), key=os.path.getmtime, reverse=True)
        
        if not files:
            st.info("Arşiv henüz boş.")
        
        for f in files:
            try:
                fname = os.path.basename(f)
                # Dosya adından bilgileri ayıkla (zaman_isim_p_s.png)
                parts = fname.replace(".png", "").split("_")
                
                display_label = f"📁 Kayıt: {fname}"
                if len(parts) >= 4:
                    display_label = f"👤 {parts[1]} | 🔴 {parts[2]} | 🟢 {parts[3]}"

                with st.expander(display_label):
                    img = Image.open(f)
                    st.image(img, use_container_width=True)
                    
                    # İndirme Butonu
                    with open(f, "rb") as file:
                        btn = st.download_button(
                            label="📥 Analizi İndir",
                            data=file,
                            file_name=f"indirilen_{fname}",
                            mime="image/png",
                            key=f"btn_{fname}" # Her buton için benzersiz anahtar
                        )
            except Exception as e:
                continue

        
                
