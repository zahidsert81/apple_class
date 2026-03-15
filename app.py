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

# 1. SAYFA AYARLARI
st.set_page_config(page_title="Pestisit Tespit Sistemi", page_icon="🍎", layout="wide")

# 2. AYARLAR & DİZİNLER
ADMIN_PASSWORD = "1234" 
SAVE_DIR = "analiz_havuzu"
MODEL_FILE = "rf_model.pkl"
SCALER_FILE = "scaler.pkl"

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

# 3. ÖZELLİK ÇIKARMA (Modelin Anlayacağı Teknik Veri)
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

# 4. MODELİ YENİDEN EĞİTME (Kendi Kendini Geliştirme)
def retrain_system(model, scaler):
    files = glob.glob(f"{SAVE_DIR}/*.png")
    if len(files) < 5:
        return False, "Eğitim için havuzda en az 5 yeni analiz olmalı."
    
    X_new, y_new = [], []
    for f in files:
        try:
            parts = os.path.basename(f).replace(".png", "").split("_")
            label = 1 if int(parts[2]) > int(parts[3]) else 0
            img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                X_new.append(extract_features(img)[0])
                y_new.append(label)
        except: continue
    
    if X_new:
        model.fit(X_new, y_new)
        with open(MODEL_FILE, "wb") as f: pickle.dump(model, f)
        return True, f"Model {len(X_new)} yeni veri ile güncellendi."
    return False, "Veri işleme hatası."

# 5. ASSET YÜKLEME
@st.cache_resource
def load_assets():
    if os.path.exists(MODEL_FILE) and os.path.exists(SCALER_FILE):
        with open(MODEL_FILE, "rb") as f: m = pickle.load(f)
        with open(SCALER_FILE, "rb") as f: s = pickle.load(f)
        return m, s
    return None

# 6. KURUMSAL ÜST PANEL (LOGOLAR)
col_logo1, col_head, col_logo2 = st.columns([1, 4, 1])

with col_logo1:
    # Sol tarafa TÜBİTAK veya kurum logosu
    if os.path.exists("TÜBİTAK_logo.svg.png"): 
        st.image("TÜBİTAK_logo.svg.png", width=120)

with col_head:
    st.markdown("<h1 style='text-align: center; color: #1E3A8A; margin-top: 0;'>PESTİSİT TESPİT SİSTEMİ</h1>", unsafe_allow_html=True)
    st.markdown(f"<p style='text-align: center; color: #64748B; font-weight: bold;'>Muhammed Zahid SERT | Analiz ve Raporlama Modülü</p>", unsafe_allow_html=True)

with col_logo2:
    # Sağ tarafa okul veya diğer proje logosu
    if os.path.exists("images.jpg"): 
        st.image("images.jpg", width=120)

assets = load_assets()

if not assets:
    st.error("⚠️ Sistem hatası: Model dosyaları (.pkl) bulunamadı.")
else:
    model, scaler = assets
    
    # --- YAN PANEL ---
    with st.sidebar:
        st.header("🛂 Sistem Kontrolü")
        user_name = st.text_input("Operatör Adı:", placeholder="Giriş yapın...")
        st.divider()
        
        if 'logged_in' not in st.session_state: st.session_state.logged_in = False
        
        if not st.session_state.logged_in:
            admin_pw = st.text_input("Yönetici Şifresi:", type="password")
            if st.button("Yönetici Paneline Giriş"):
                if admin_pw == ADMIN_PASSWORD:
                    st.session_state.logged_in = True
                    st.rerun()
        else:
            st.success("Yönetici Modu Aktif")
            if st.button("🚀 MODELİ EĞİT (ÖĞRENME MODU)"):
                success, msg = retrain_system(model, scaler)
                if success: st.success(msg); st.balloons()
                else: st.warning(msg)
            
            if st.button("🗑️ Arşivi Sıfırla"):
                for f in glob.glob(f"{SAVE_DIR}/*.png"): os.remove(f)
                st.rerun()
            if st.button("Çıkış Yap"):
                st.session_state.logged_in = False
                st.rerun()

    # --- ANA ANALİZ AKIŞI ---
    uploaded = st.file_uploader("Analiz Edilecek Termal Görüntüyü Seçin", type=["jpg", "png", "jpeg"])

    if uploaded and user_name:
        pil_img = Image.open(uploaded).convert("RGB")
        with st.spinner("Yapay Zeka Termal Veriyi Analiz Ediyor..."):
            # Heatmap ve Ön İşleme
            nobg = remove(pil_img).convert("RGB")
            img_np = np.array(nobg)
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            heatmap = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
            
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            _, th = cv2.threshold(blur, 10, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            res_img = img_np.copy()
            detaylar = []
            
            for cnt in contours:
                if cv2.contourArea(cnt) < 1500: continue
                x, y, w, h = cv2.boundingRect(cnt)
                crop = gray[y:y+h, x:x+w]
                
                feats = extract_features(crop)
                f_scaled = scaler.transform(feats)
                pred = model.predict(f_scaled)[0]
                prob = model.predict_proba(f_scaled)[0]
                
                label = "PESTİSİTLİ" if pred == 1 else "TEMİZ"
                color = (255, 0, 0) if pred == 1 else (0, 255, 0)
                
                detaylar.append({
                    "Nesne": f"Örnek {len(detaylar)+1}",
                    "Durum": label,
                    "Güven": f"%{max(prob)*100:.1f}",
                    "Renk": "#fee2e2" if pred == 1 else "#dcfce7",
                    "Border": "#991b1b" if pred == 1 else "#166534"
                })
                
                cv2.rectangle(res_img, (x, y), (x+w, y+h), color, 12)
                cv2.putText(res_img, f"#{len(detaylar)} {label}", (x, y-20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.8, color, 4)

            # --- ANALİZ RAPORU ---
            st.subheader("📋 Teknik Analiz Raporu")
            c1, c2, c3 = st.columns(3)
            with c1: st.image(uploaded, caption="Girdi Verisi", use_container_width=True)
            with c2: st.image(heatmap, caption="Isı Haritası (Heatmap)", use_container_width=True)
            with c3: st.image(res_img, caption="Teşhis ve Sınıflandırma", use_container_width=True)
            
            st.divider()
            
            # Kartlar (Az sayıda elma için büyük görünüm)
            if detaylar:
                cols = st.columns(len(detaylar))
                for i, d in enumerate(detaylar):
                    with cols[i]:
                        st.markdown(f"""
                            <div style="background-color:{d['Renk']}; padding:25px; border-radius:15px; border:4px solid {d['Border']}; text-align:center;">
                                <h2 style="color:{d['Border']}; margin:0;">{d['Nesne']}</h2>
                                <b style="font-size:26px; color:{d['Border']}">{d['Durum']}</b><br>
                                <span style="color:#333;">Teşhis Güveni: <b>{d['Güven']}</b></span>
                            </div>
                        """, unsafe_allow_html=True)

            # Kayıt (Zaman damgası ve sonuçlarla)
            p_say = sum(1 for d in detaylar if d["Durum"] == "PESTİSİTLİ")
            s_say = len(detaylar) - p_say
            clean_name = "".join(x for x in user_name if x.isalnum())
            save_path = f"{SAVE_DIR}/{int(time.time())}_{clean_name}_{p_say}_{s_say}.png"
            Image.fromarray(res_img).save(save_path)

    # --- YÖNETİCİ ARŞİVİ ---
    if st.session_state.logged_in:
        st.divider()
        st.header("📊 Geçmiş Analiz Kayıtları")
        files = sorted(glob.glob(f"{SAVE_DIR}/*.png"), key=os.path.getmtime, reverse=True)
        if files:
            df_data = []
            for f in files:
                p = os.path.basename(f).replace(".png", "").split("_")
                if len(p) >= 4:
                    df_data.append({
                        "Tarih": time.strftime('%Y-%m-%d %H:%M', time.localtime(int(p[0]))),
                        "Operatör": p[1],
                        "Pestisitli": p[2],
                        "Temiz": p[3]
                    })
            st.table(pd.DataFrame(df_data))
        else:
            st.info("Sistem henüz bir kayıt üretmedi.")
