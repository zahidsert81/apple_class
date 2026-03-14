import streamlit as st
import os
import time

# Sayfa ayarlarını en başta yapalım (Beyaz ekranı önlemek için)
st.set_page_config(page_title="Pestisit Analiz Lab", page_icon="🍎", layout="wide")

# Kütüphane kontrolü ve Importlar
try:
    import numpy as np
    import cv2
    import pickle
    from PIL import Image
    from rembg import remove
    from skimage.feature import graycomatrix, graycoprops
    from scipy.stats import skew, kurtosis
    import sklearn
except ImportError as e:
    st.error(f"Kütüphane yükleme hatası: {e}")
    st.info("Lütfen requirements.txt dosyasını kontrol edin ve uygulamanın sağ altından 'Reboot' yapın.")
    st.stop()

# ==========================================
# ÖZELLİK ÇIKARMA
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
# MODEL YÜKLEME
# ==========================================
@st.cache_resource
def load_assets():
    base_path = os.path.dirname(__file__)
    m_p = os.path.join(base_path, "rf_model.pkl")
    s_p = os.path.join(base_path, "scaler.pkl")
    
    if os.path.exists(m_p) and os.path.exists(s_p):
        try:
            with open(m_p, "rb") as f:
                m = pickle.load(f)
            with open(s_p, "rb") as f:
                s = pickle.load(f)
            return m, s
        except Exception as e:
            return f"Yükleme Hatası: {e}"
    return "Dosya bulunamadı."

# ==========================================
# ANA UYGULAMA
# ==========================================
if 'db' not in st.session_state:
    st.session_state.db = []

st.title("🌡️ Termal Analiz ve Pestisit Tespit Sistemi")

# Model Kontrolü
assets = load_assets()

if isinstance(assets, str):
    st.error(f"Sistem Hazır Değil: {assets}")
    st.info("GitHub'da 'rf_model.pkl' ve 'scaler.pkl' dosyalarının olduğundan emin olun.")
else:
    model, scaler = assets
    st.sidebar.success("✅ Sistem Aktif")
    
    # Haber ve Başarı Kısmı (Sidebar)
    st.sidebar.markdown("### 🥉 Başarılarımız")
    st.sidebar.write("TÜBİTAK 2204-B Türkiye 3.lüğü")
    
    uploaded = st.file_uploader("Analiz için termal fotoğraf yükleyin", type=["jpg", "png", "jpeg"])

    if uploaded:
        pil_img = Image.open(uploaded).convert("RGB")
        
        with st.spinner("Analiz ediliyor..."):
            # Görüntü İşleme
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
                feats_scaled = scaler.transform(feats)
                pred = model.predict(feats_scaled)[0]
                
                label = "PESTISITLI" if pred == 1 else "PESTISITSIZ"
                color = (255, 0, 0) if pred == 1 else (0, 255, 0)
                if pred == 1: p_say += 1
                else: s_say += 1
                
                cv2.rectangle(res_img, (x, y), (x+w, y+h), color, 12)
                cv2.putText(res_img, label, (x, y-15), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 4)

            # Veritabanına Ekle
            st.session_state.db.insert(0, {"img": res_img, "p": p_say, "s": s_say, "t": time.strftime("%H:%M:%S")})
            if len(st.session_state.db) > 100: st.session_state.db.pop()

            # Gösterim
            c1, c2 = st.columns([2, 1])
            with c1:
                st.image(res_img, use_container_width=True)
            with c2:
                st.metric("🔴 PESTİSİTLİ", p_say)
                st.metric("🟢 PESTİSİTSİZ", s_say)

    # Arşiv
    if st.session_state.db:
        st.divider()
        st.subheader(f"📂 Analiz Arşivi ({len(st.session_state.db)} / 100)")
        cols = st.columns(4)
        for idx, item in enumerate(st.session_state.db):
            with cols[idx % 4]:
                st.image(item["img"], use_container_width=True)
                st.caption(f"🕒 {item['t']} | 🔴 {item['p']} | 🟢 {item['s']}")
