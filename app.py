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
st.set_page_config(page_title="Pestisit Analiz Lab", page_icon="🧪", layout="wide")

# 2. AYARLAR & DİZİNLER
ADMIN_PASSWORD = "1234" 
SAVE_DIR = "analiz_havuzu"
MODEL_FILE = "rf_model.pkl"
SCALER_FILE = "scaler.pkl"

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

# 3. ÖZELLİK ÇIKARMA (Modelin Anlayacağı Veri)
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

# 4. MODELİ YENİDEN EĞİTME (Active Learning)
def retrain_system(model, scaler):
    files = glob.glob(f"{SAVE_DIR}/*.png")
    if len(files) < 5:
        return False, "Eğitim için havuzda en az 5 yeni analiz olmalı."
    
    X_new, y_new = [], []
    for f in files:
        try:
            parts = os.path.basename(f).replace(".png", "").split("_")
            label = 1 if int(parts[2]) > int(parts[3]) else 0 # Pestisit baskınsa 1
            img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                X_new.append(extract_features(img)[0])
                y_new.append(label)
        except: continue
    
    if X_new:
        model.fit(X_new, y_new) # Modeli yeni verilerle güncelle
        with open(MODEL_FILE, "wb") as f: pickle.dump(model, f)
        return True, f"Başarılı! {len(X_new)} yeni veri YZ belleğine eklendi."
    return False, "Veri işleme hatası."

# 5. MODEL YÜKLEME
@st.cache_resource
def load_assets():
    if os.path.exists(MODEL_FILE) and os.path.exists(SCALER_FILE):
        with open(MODEL_FILE, "rb") as f: m = pickle.load(f)
        with open(SCALER_FILE, "rb") as f: s = pickle.load(f)
        return m, s
    return None

# 6. TASARIM VE ÜST PANEL
st.markdown("<h1 style='text-align: center; color: #1E3A8A;'>Derin Odak: Kendi Kendini Eğiten Analiz Sistemi</h1>", unsafe_allow_html=True)

assets = load_assets()
if not assets:
    st.error("⚠️ Model dosyaları eksik! Lütfen rf_model.pkl ve scaler.pkl yükleyin.")
else:
    model, scaler = assets
    
    # --- SİDEBAR (YÖNETİCİ VE EĞİTİM) ---
    with st.sidebar:
        st.header("🛂 Kontrol Merkezi")
        user_name = st.text_input("Analiz Sorumlusu:", placeholder="Adınız...")
        st.divider()
        
        if 'logged_in' not in st.session_state: st.session_state.logged_in = False
        
        if not st.session_state.logged_in:
            admin_pw = st.text_input("Yönetici Şifresi:", type="password")
            if st.button("Sistemi Yönet"):
                if admin_pw == ADMIN_PASSWORD:
                    st.session_state.logged_in = True
                    st.rerun()
        else:
            st.success("Yönetici Modu")
            if st.button("🚀 MODELİ YENİ VERİLERLE EĞİT"):
                success, msg = retrain_system(model, scaler)
                if success: st.success(msg); st.balloons()
                else: st.warning(msg)
            
            if st.button("🗑️ Arşivi Boşalt"):
                for f in glob.glob(f"{SAVE_DIR}/*.png"): os.remove(f)
                st.rerun()
            if st.button("Çıkış Yap"):
                st.session_state.logged_in = False
                st.rerun()

    # --- ANALİZ AKIŞI ---
    uploaded = st.file_uploader("Analiz için Termal Görüntü Yükleyin", type=["jpg", "png", "jpeg"])

    if uploaded and user_name:
        pil_img = Image.open(uploaded).convert("RGB")
        with st.spinner("Yapay Zeka Belleğini Kullanarak Analiz Yapıyor..."):
            # İşleme ve Isı Haritası (Heatmap)
            nobg = remove(pil_img).convert("RGB")
            img_np = np.array(nobg)
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            heatmap = cv2.applyColorMap(gray, cv2.COLORMAP_JET) # Bilimsel renk paleti
            
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
                
                label = "PESTISITLI" if pred == 1 else "TEMIZ"
                color = (255, 0, 0) if pred == 1 else (0, 255, 0)
                
                detaylar.append({
                    "Nesne": f"Elma {len(detaylar)+1}",
                    "Durum": label,
                    "Güven": f"%{max(prob)*100:.1f}",
                    "Renk": "#fee2e2" if pred == 1 else "#dcfce7",
                    "Border": "#991b1b" if pred == 1 else "#166534"
                })
                
                cv2.rectangle(res_img, (x, y), (x+w, y+h), color, 12)
                cv2.putText(res_img, label, (x, y-15), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 4)

            # Raporlama Ekranı
            st.subheader("📝 Aktif Analiz Raporu")
            c1, c2, c3 = st.columns(3)
            with c1: st.image(uploaded, caption="Orijinal Veri", use_container_width=True)
            with c2: st.image(heatmap, caption="Isı Haritası (Heatmap)", use_container_width=True)
            with c3: st.image(res_img, caption="YZ Teşhis", use_container_width=True)
            
            st.divider()
            
            # Kartlar
            if detaylar:
                cols = st.columns(len(detaylar))
                for i, d in enumerate(detaylar):
                    with cols[i]:
                        st.markdown(f"""
                            <div style="background-color:{d['Renk']}; padding:20px; border-radius:15px; border:3px solid {d['Border']}; text-align:center;">
                                <h3 style="color:{d['Border']};">{d['Nesne']}</h3>
                                <b style="font-size:22px;">{d['Durum']}</b><br>
                                <span>Güven Oranı: {d['Güven']}</span>
                            </div>
                        """, unsafe_allow_html=True)

            # Otomatik Kayıt (Öğrenme için veri biriktirir)
            p_say = sum(1 for d in detaylar if d["Durum"] == "PESTISITLI")
            s_say = len(detaylar) - p_say
            save_path = f"{SAVE_DIR}/{int(time.time())}_{user_name}_{p_say}_{s_say}.png"
            Image.fromarray(res_img).save(save_path)

    # --- ARŞİV VE TABLO ---
    if st.session_state.logged_in:
        st.divider()
        st.header("📂 Veri Havuzu ve Eğitim Kayıtları")
        files = sorted(glob.glob(f"{SAVE_DIR}/*.png"), key=os.path.getmtime, reverse=True)
        if files:
            df = []
            for f in files:
                p = os.path.basename(f).replace(".png", "").split("_")
                if len(p) >= 4:
                    df.append({"Tarih": time.ctime(int(p[0])), "Sorumlu": p[1], "Pestisitli": p[2], "Temiz": p[3]})
            st.table(pd.DataFrame(df))
        else:
            st.info("Havuzda henüz veri yok.")
