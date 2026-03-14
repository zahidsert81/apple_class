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

# 1. SAYFA AYARLARI
st.set_page_config(page_title="Pestisit Analiz Lab", page_icon="🍎", layout="wide")

# 2. AYARLAR VE ŞİFRE
# Şifreyi sunumdan önce buradan değiştirebilirsin.
ADMIN_PASSWORD = "ZAHID" 
SAVE_DIR = "analiz_havuzu"
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

# 3. ÖZELLİK ÇIKARMA FONKSİYONU
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
        try:
            with open(m_p, "rb") as f: m = pickle.load(f)
            with open(s_p, "rb") as f: s = pickle.load(f)
            return m, s
        except Exception as e: return f"Model Hatası: {e}"
    return "Model dosyaları (.pkl) bulunamadı."

# 5. ÜST BİLGİ VE LOGOLAR
def display_header():
    col1, col2, col3 = st.columns([1, 4, 1])
    with col1:
        if os.path.exists("TÜBİTAK_logo.svg.png"): st.image("TÜBİTAK_logo.svg.png", width=110)
    with col2:
        st.markdown("<h1 style='text-align: center; margin-bottom: 0;'>Termal Analiz Sistemi</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: gray;'>Muhammed Zahid SERT | Danışman: Emine SARI</p>", unsafe_allow_html=True)
    with col3:
        if os.path.exists("images.jpg"): st.image("images.jpg", width=110)
    st.divider()

display_header()
assets = load_assets()

if isinstance(assets, str):
    st.error(f"Sistem Hazır Değil: {assets}")
else:
    model, scaler = assets
    
    # --- YAN PANEL (SIDEBAR) ---
    with st.sidebar:
        st.header("🔬 Giriş Paneli")
        user_name = st.text_input("Analiz Yapanın Adı:", placeholder="İsminizi yazın...")
        
        st.divider()
        st.header("🔐 Yetkili Erişimi")
        admin_auth = st.text_input("Yönetici Şifresi:", type="password")
        
        if admin_auth == ADMIN_PASSWORD:
            st.success("Yönetici girişi yapıldı.")
            if st.button("Tüm Havuzu Temizle"):
                for f in glob.glob(f"{SAVE_DIR}/*.png"): os.remove(f)
                st.rerun()
        elif admin_auth != "":
            st.error("Hatalı şifre!")

    # --- ANA ANALİZ EKRANI ---
    uploaded = st.file_uploader("Termal Görüntü Yükleyin", type=["jpg", "png", "jpeg"])

    if uploaded:
        if not user_name:
            st.warning("Lütfen önce sol menüden adınızı girin!")
        else:
            pil_img = Image.open(uploaded).convert("RGB")
            with st.spinner("Yapay zeka analiz ediyor..."):
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

                # Ortak Havuza Kaydet
                timestamp = int(time.time())
                clean_name = "".join(x for x in user_name if x.isalnum())
                save_path = f"{SAVE_DIR}/{timestamp}_{clean_name}_{p_say}_{s_say}.png"
                Image.fromarray(res_img).save(save_path)

                # --- SONUÇLARI GÖSTER (GÜNCELLENDİ) ---
                st.subheader("📊 Analiz Sonuçları")
                
                # Sürümleri yan yana göstermek için 2 sütun oluşturuyoruz
                col_img1, col_img2 = st.columns(2)
                
                with col_img1:
                    # Orijinal termal görüntüyü yüklendiği gibi göster
                    st.image(uploaded, caption="Yüklenen Görüntü", use_container_width=True)
                    
                with col_img2:
                    # Analiz edilmiş, kare içine alınmış görüntüyü göster
                    st.image(res_img, caption="Analiz Edilmiş Görüntü", use_container_width=True)
                
                # Sayısal sonuçları metrik olarak göster
                st.divider()
                col_metric1, col_metric2 = st.columns(2)
                with col_metric1:
                    st.metric("🔴 PESTİSİTLİ Örnek Sayısı", p_say)
                with col_metric2:
                    st.metric("🟢 PESTİSİTSİZ Örnek Sayısı", s_say)

    # --- ŞİFRELİ ORTAK HAVUZ GÖRÜNÜMÜ ---
    if admin_auth == ADMIN_PASSWORD:
        st.divider()
        st.subheader("🌐 Ortak Analiz Havuzu (Yönetici Paneli)")
        
        all_files = sorted(glob.glob(f"{SAVE_DIR}/*.png"), key=os.path.getmtime, reverse=True)
        
        if all_files:
            for f in all_files:
                fname = os.path.basename(f).replace(".png", "")
                parts = fname.split("_")
                if len(parts) >= 4:
                    u_name, p_val, s_val = parts[1], parts[2], parts[3]
                    # Havuzda da genişliği sınırla
                    with st.expander(f"👤 Gönderen: {u_name} | 🔴 {p_val} | 🟢 {s_val}"):
                        st.image(f, use_container_width=True)
        else:
            st.info("Havuzda kayıtlı analiz bulunmuyor.")
    else:
        # Şifre girilmediyse havuz gizli kalır
        st.divider()
        st.caption("Ortak analiz kayıtlarını görmek için yönetici girişi yapmalısınız.")
