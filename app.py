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
import qrcode

# 1. SAYFA AYARLARI
st.set_page_config(page_title="Pestisit Analiz Lab", page_icon="🧪", layout="wide")

# 2. AYARLAR
ADMIN_PASSWORD = "1234" 
SAVE_DIR = "analiz_havuzu"
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

# --- MOBİL GÖNDERİM KONTROLÜ ---
query_params = st.query_params
is_mobile = query_params.get("mode") == "mobile"

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

# 4. MODEL VE SCALER YÜKLEME
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

# --- MOBİL ARAYÜZ (GPS EKLENDİ) ---
if is_mobile:
    st.markdown("<h2 style='text-align: center;'>📲 Mobil Veri Gönderimi</h2>", unsafe_allow_html=True)
    
    # Konum bilgisini almak için HTML5 Geolocation API kullanıyoruz
    st.markdown("""
        <script>
        function getLocation() {
          if (navigator.geolocation) {
            navigator.geolocation.getCurrentPosition(showPosition);
          }
        }
        function showPosition(position) {
          const lat = position.coords.latitude;
          const lon = position.coords.longitude;
          window.parent.postMessage({type: 'location', lat: lat, lon: lon}, '*');
        }
        getLocation();
        </script>
    """, unsafe_allow_html=True)

    mobile_user = st.text_input("Analiz Sorumlusu Adı:", key="mob_user")
    # Kullanıcıdan koordinatları manuel onay gibi almak için gizli olmayan bir alan
    coords = st.text_input("Konum (Opsiyonel - örn: 40.84, 31.15):", placeholder="Harita verisi için boş bırakılabilir")
    mob_file = st.file_uploader("Termal Fotoğrafı Çek veya Seç", type=["jpg", "png", "jpeg"], key="mob_file")
    
    if mob_file and mobile_user:
        with st.spinner("Dosya ve konum sunucuya aktarılıyor..."):
            img = Image.open(mob_file)
            # Dosya ismine koordinatları da gömüyoruz
            loc_tag = coords.replace(",", "-").replace(" ", "") if coords else "no-gps"
            temp_path = f"{SAVE_DIR}/TEMP_{int(time.time())}_{mobile_user}_{loc_tag}.png"
            img.save(temp_path)
            st.success("✅ Veri başarıyla gönderildi!")
            st.balloons()
    st.stop()

# --- ANA BİLGİSAYAR ARAYÜZÜ ---
col_l, col_m, col_r = st.columns([1, 4, 1])
with col_l:
    if os.path.exists("TÜBİTAK_logo.svg.png"): st.image("TÜBİTAK_logo.svg.png", width=110)
with col_m:
    st.markdown("<h1 style='text-align: center; color: #1E3A8A; margin-bottom: 0;'>Pestisit Tespit Sistemi</h1>", unsafe_allow_html=True)
    st.markdown(f"<p style='text-align: center; color: #64748B;'>Muhammed Zahid SERT | Gıda Güvenliği Analiz Modülü</p>", unsafe_allow_html=True)
with col_r:
    if os.path.exists("images.jpg"): st.image("images.jpg", width=110)

assets = load_assets()

if not assets:
    st.error("⚠️ Kritik Hata: Model dosyaları (.pkl) bulunamadı.")
else:
    model, scaler = assets
    
    with st.sidebar:
        st.header("🛂 Giriş Paneli")
        user_name = st.text_input("Analiz Sorumlusu:", placeholder="Adınız...")
        
        st.divider()
        st.subheader("📲 Telefondan Fotoğraf Gönder")
        target_url = "https://pestisit-kontrol.streamlit.app/?mode=mobile"
        qr = qrcode.make(target_url)
        buf = BytesIO()
        qr.save(buf, format="PNG")
        st.image(buf, caption="Telefonla Tara & Yükle")
        st.divider()

        if 'logged_in' not in st.session_state: st.session_state.logged_in = False
        if not st.session_state.logged_in:
            admin_pw = st.text_input("Yönetici Şifresi:", type="password")
            if st.button("Yönetici Girişi"):
                if admin_pw == ADMIN_PASSWORD:
                    st.session_state.logged_in = True
                    st.rerun()
        else:
            st.success("Yönetici Modu Aktif")
            if st.button("Oturumu Kapat"):
                st.session_state.logged_in = False
                st.rerun()
            if st.button("🗑️ Arşivi Temizle"):
                for f in glob.glob(f"{SAVE_DIR}/*.png"): os.remove(f)
                st.rerun()

    # MOBİL'DEN GELEN DOSYALARI YAKALAMA VE KONUMU AYRIŞTIRMA
    temp_files = glob.glob(f"{SAVE_DIR}/TEMP_*.png")
    uploaded = st.file_uploader("Bilgisayardan Fotoğraf Yükleyin", type=["jpg", "png", "jpeg"])
    current_gps = "Bilinmiyor"
    
    source_file = None
    if temp_files:
        source_file = temp_files[0]
        fname_parts = os.path.basename(source_file).split("_")
        if len(fname_parts) >= 4:
            current_gps = fname_parts[3].replace(".png", "")
        
        st.warning(f"🔔 Mobil Veri Geldi! Sorumlu: {fname_parts[2]} | Konum: {current_gps}")
        if st.button("Mobil Veriyi İşle"):
            uploaded = source_file
    
    if uploaded and user_name:
        pil_img = Image.open(uploaded).convert("RGB")
        with st.spinner("Analiz ediliyor..."):
            nobg = remove(pil_img).convert("RGB")
            gray = cv2.cvtColor(np.array(nobg), cv2.COLOR_RGB2GRAY)
            # ... (Özellik çıkarma ve tahmin bölümleri aynı kalıyor)
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            _, th = cv2.threshold(blur, 10, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            res_img = np.array(pil_img).copy()
            detaylar = []
            for cnt in contours:
                if cv2.contourArea(cnt) < 1500: continue
                x, y, w, h = cv2.boundingRect(cnt)
                crop = gray[y:y+h, x:x+w]
                feats = extract_features(crop)
                f_scaled = scaler.transform(feats)
                pred = model.predict(f_scaled)[0]
                prob = model.predict_proba(f_scaled)[0]
                
                label = "PESTISITLI" if pred == 1 else "TEMIZ"
                color = (255, 0, 0) if pred == 1 else (0, 255, 0)
                detaylar.append({"Durum": label, "Güven": f"%{max(prob)*100:.1f}", "Renk": "#fee2e2" if pred == 1 else "#dcfce7", "Border": "#991b1b" if pred == 1 else "#166534"})
                cv2.rectangle(res_img, (x, y), (x+w, y+h), color, 12)

            # --- SONUÇ PANELİ VE KONUM GÖSTERİMİ ---
            st.subheader("📊 Analitik Raporlama")
            if current_gps != "Bilinmiyor" and "no-gps" not in current_gps:
                st.info(f"📍 Analiz Edilen Bölge Koordinatı: {current_gps}")
                # Google Maps linki ekleyelim
                map_url = f"https://www.google.com/maps/search/?api=1&query={current_gps.replace('-', ',')}"
                st.markdown(f"[🌍 Haritada Görüntüle]({map_url})")

            col_res1, col_res2 = st.columns(2)
            with col_res1: st.image(uploaded, use_container_width=True)
            with col_res2: st.image(res_img, use_container_width=True)

            # Kayıt (Dosya ismine konumu da ekliyoruz)
            p_say = sum(1 for d in detaylar if d["Durum"] == "PESTISITLI")
            s_say = len(detaylar) - p_say
            clean_gps = current_gps.replace("-", "_")
            save_path = f"{SAVE_DIR}/{int(time.time())}_{user_name}_{p_say}_{s_say}_{clean_gps}.png"
            Image.fromarray(res_img).save(save_path)
            
            if isinstance(uploaded, str) and "TEMP_" in uploaded:
                os.remove(uploaded)

    # --- ARŞİV TABLOSU (KONUM SÜTUNU EKLENDİ) ---
    if st.session_state.logged_in:
        st.divider()
        st.header("📂 Laboratuvar Arşiv Havuzu")
        files = sorted(glob.glob(f"{SAVE_DIR}/*.png"), key=os.path.getmtime, reverse=True)
        files = [f for f in files if "TEMP_" not in f]
        
        if files:
            data = []
            for f in files:
                p = os.path.basename(f).replace(".png", "").split("_")
                if len(p) >= 5:
                    data.append({
                        "Zaman": time.ctime(int(p[0])), 
                        "Sorumlu": p[1], 
                        "Pestisitli": p[2], 
                        "Temiz": p[3],
                        "Konum (GPS)": p[4] if p[4] != "no-gps" else "Belirtilmedi"
                    })
            st.table(pd.DataFrame(data))
