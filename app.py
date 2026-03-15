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
import qrcode # Kütüphane eklendi

# 1. SAYFA AYARLARI
st.set_page_config(page_title="Pestisit Analiz Lab", page_icon="🧪", layout="wide")

# 2. AYARLAR
ADMIN_PASSWORD = "1234" 
SAVE_DIR = "analiz_havuzu"
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

# 3. ÖZELLİK ÇIKARMA (GLCM & İstatistiksel)
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

# 5. ÜST BİLGİ PANELİ
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
    st.error("⚠️ Kritik Hata: Model dosyaları (.pkl) bulunamadı. Lütfen dosyaları yükleyin.")
else:
    model, scaler = assets
    
    # --- YAN PANEL (KONTROL) ---
    with st.sidebar:
        st.header("🛂 Giriş Paneli")
        user_name = st.text_input("Analiz Sorumlusu:", placeholder="Adınız...")
        
        # --- QR KOD BÖLÜMÜ (YENİ) ---
        st.divider()
        st.subheader("📲 Mobil Hızlı Erişim")
        # Uygulamanın çalıştığı URL'yi otomatik al veya manuel gir
        app_url = "https://pestisit-kontrol.streamlit.app/" # Buraya gerçek URL'nizi yazın
        qr = qrcode.make(app_url)
        buf = BytesIO()
        qr.save(buf, format="PNG")
        st.image(buf, caption="Veri Göndermek İçin Taratın", width=150)
        # --------------------------
        
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

    # --- ANA ANALİZ BÖLÜMÜ ---
    uploaded = st.file_uploader("Termal Görüntü Yükleyin", type=["jpg", "png", "jpeg"])

    if uploaded and user_name:
        pil_img = Image.open(uploaded).convert("RGB")
        with st.spinner("Yapay Zeka Mikroskobik Verileri İnceliyor..."):
            # Arka plan temizleme ve ön işleme
            nobg = remove(pil_img).convert("RGB")
            gray = cv2.cvtColor(np.array(nobg), cv2.COLOR_RGB2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            _, th = cv2.threshold(blur, 10, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            res_img = np.array(pil_img).copy()
            detaylar = []
            
            for cnt in contours:
                if cv2.contourArea(cnt) < 1500: continue
                x, y, w, h = cv2.boundingRect(cnt)
                crop = gray[y:y+h, x:x+w]
                
                # Model Tahmini
                feats = extract_features(crop)
                f_scaled = scaler.transform(feats)
                pred = model.predict(f_scaled)[0]
                prob = model.predict_proba(f_scaled)[0] # Güven oranı
                
                label = "PESTISITLI" if pred == 1 else "TEMIZ"
                color = (255, 0, 0) if pred == 1 else (0, 255, 0)
                
                detaylar.append({
                    "Nesne": f"Örnek {len(detaylar)+1}",
                    "Durum": label,
                    "Güven": f"%{max(prob)*100:.1f}",
                    "Doku": f"{feats[0][0]:.2f}",
                    "Renk": "#fee2e2" if pred == 1 else "#dcfce7",
                    "Border": "#991b1b" if pred == 1 else "#166534"
                })
                
                cv2.rectangle(res_img, (x, y), (x+w, y+h), color, 12)
                cv2.putText(res_img, f"#{len(detaylar)} {label}", (x, y-15), cv2.FONT_HERSHEY_SIMPLEX, 1.6, color, 4)

            # --- SONUÇLARI GÖSTER ---
            st.subheader("📊 Analitik Raporlama")
            col_res1, col_res2 = st.columns(2)
            with col_res1: st.image(uploaded, caption="Giriş Verisi", use_container_width=True)
            with col_res2: st.image(res_img, caption="Analiz Sonucu", use_container_width=True)
            
            st.divider()
            
            # Nesne Kartları (2 elma için ideal görünüm)
            if detaylar:
                card_cols = st.columns(len(detaylar))
                for i, d in enumerate(detaylar):
                    with card_cols[i]:
                        st.markdown(f"""
                            <div style="background-color:{d['Renk']}; padding:25px; border-radius:15px; border:3px solid {d['Border']}; text-align:center;">
                                <h2 style="color:{d['Border']}; margin:0;">{d['Nesne']}</h2>
                                <p style="font-size:24px; font-weight:bold; color:{d['Border']}">{d['Durum']}</p>
                                <p style="margin:5px 0; color:#333;">🎯 Güven Skoru: <b>{d['Güven']}</b></p>
                                <p style="margin:0; color:#555; font-size:14px;">Mikroskobik Doku Pürüzlülüğü: {d['Doku']}</p>
                            </div>
                        """, unsafe_allow_html=True)

            # Kayıt işlemi
            p_say = sum(1 for d in detaylar if d["Durum"] == "PESTISITLI")
            s_say = len(detaylar) - p_say
            clean_name = "".join(x for x in user_name if x.isalnum())
            save_path = f"{SAVE_DIR}/{int(time.time())}_{clean_name}_{p_say}_{s_say}.png"
            Image.fromarray(res_img).save(save_path)

    # --- YÖNETİCİ ARŞİVİ ---
    if st.session_state.logged_in:
        st.divider()
        st.header("📂 Laboratuvar Arşiv Havuzu")
        
        files = sorted(glob.glob(f"{SAVE_DIR}/*.png"), key=os.path.getmtime, reverse=True)
        
        if files:
            # Toplu ZIP İndirme
            zip_buf = BytesIO()
            with zipfile.ZipFile(zip_buf, "a", zipfile.ZIP_DEFLATED, False) as zip_f:
                for f in files: zip_f.write(f, os.path.basename(f))
            st.download_button("📦 Tüm Arşivi ZIP Olarak İndir", data=zip_buf.getvalue(), file_name="analiz_arsivi.zip", mime="application/zip")
            
            # Tablo Görünümü
            data = []
            for f in files:
                p = os.path.basename(f).replace(".png", "").split("_")
                if len(p) >= 4:
                    data.append({"Zaman": time.ctime(int(p[0])), "Sorumlu": p[1], "Pestisitli": p[2], "Temiz": p[3]})
            st.table(pd.DataFrame(data))

            # Detaylı Görsel İnceleme
            for f in files:
                fname = os.path.basename(f)
                with st.expander(f"İncele: {fname}"):
                    st.image(f, use_container_width=True)
                    with open(f, "rb") as fb:
                        st.download_button("💾 Görseli İndir", data=fb, file_name=fname, mime="image/png", key=f"btn_{fname}")
        else:
            st.info("Arşiv henüz boş.")
