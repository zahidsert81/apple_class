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

# 3. ÖZELLİK ÇIKARMA (Eski Model ve Scaler ile Tam Uyumlu 11 Özellik)
def extract_features(img_gray):
    # Gürültü giderme
    img_gray = cv2.medianBlur(img_gray, 5)
    img_8 = cv2.normalize(img_gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Doku Analizi (GLCM) - 6 Özellik
    glcm = graycomatrix(img_8, distances=[1], angles=[0], symmetric=True, normed=True)
    glcm_features = [
        graycoprops(glcm, 'contrast')[0, 0],
        graycoprops(glcm, 'dissimilarity')[0, 0],
        graycoprops(glcm, 'homogeneity')[0, 0],
        graycoprops(glcm, 'energy')[0, 0],
        graycoprops(glcm, 'correlation')[0, 0],
        graycoprops(glcm, 'ASM')[0, 0]
    ]
    
    # İstatistiksel Özellikler - 4 Özellik
    flat = img_gray.flatten()
    stats = [np.mean(flat), np.std(flat), skew(flat), kurtosis(flat)]
    
    # Kenar Yoğunluğu - 1 Özellik
    edges = cv2.Canny(img_8, 50, 150)
    edge_density = np.sum(edges > 0) / edges.size
    
    return np.array(glcm_features + stats + [edge_density]).reshape(1, -1)

# 4. AKTİF ÖĞRENME MODÜLÜ (Modeli Yeni Verilerle Eğitme)
def train_with_collected_data(model, scaler):
    all_files = glob.glob(f"{SAVE_DIR}/*.png")
    if len(all_files) < 3:
        return False, "Eğitim için havuzda yeterli (en az 3) yeni veri yok."
    
    X_new, y_new = [], []
    for f in all_files:
        try:
            # Dosya adından etiket çözümü: timestamp_name_pestisit_temiz.png
            parts = os.path.basename(f).replace(".png", "").split("_")
            # Eğer pestisitli sayısı temizden fazlaysa etiket 1, değilse 0
            label = 1 if int(parts[2]) >= int(parts[3]) else 0
            
            img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                feats = extract_features(img)
                X_new.append(feats[0])
                y_new.append(label)
        except: continue

    if len(X_new) > 0:
        # Modeli yeni verilerle "partial_fit" değilse bile tekrar eğitip güncelleme
        model.fit(X_new, y_new) 
        with open(MODEL_FILE, "wb") as f_model:
            pickle.dump(model, f_model)
        return True, f"Model başarıyla güncellendi! {len(X_new)} yeni örnek öğrenildi."
    return False, "Özellik çıkarma sırasında hata oluştu."

# 5. MODEL VE SCALER YÜKLEME
@st.cache_resource
def load_assets():
    if os.path.exists(MODEL_FILE) and os.path.exists(SCALER_FILE):
        with open(MODEL_FILE, "rb") as f: m = pickle.load(f)
        with open(SCALER_FILE, "rb") as f: s = pickle.load(f)
        return m, s
    return None

# 6. KURUMSAL ÜST PANEL
col_l, col_m, col_r = st.columns([1, 4, 1])
with col_l:
    if os.path.exists("TÜBİTAK_logo.svg.png"): st.image("TÜBİTAK_logo.svg.png", width=110)
with col_m:
    st.markdown("<h1 style='text-align: center; color: #1E3A8A; margin-bottom: 0;'>Pestisit Tespit Sistemi</h1>", unsafe_allow_html=True)
    st.markdown(f"<p style='text-align: center; color: #64748B; font-weight: bold;'>Muhammed Zahid SERT | Gıda Güvenliği Analiz Modülü</p>", unsafe_allow_html=True)
with col_r:
    if os.path.exists("images.jpg"): st.image("images.jpg", width=110)

assets = load_assets()

if not assets:
    st.error("⚠️ Model dosyaları bulunamadı! Lütfen rf_model.pkl ve scaler.pkl yükleyin.")
else:
    model, scaler = assets
    
    # --- YAN PANEL (KONTROL) ---
    with st.sidebar:
        st.header("🛂 Kontrol Paneli")
        user_name = st.text_input("Operatör Adı:", value="Muhammed")
        st.divider()
        
        if 'logged_in' not in st.session_state: st.session_state.logged_in = False
        
        if not st.session_state.logged_in:
            admin_pw = st.text_input("Yönetici Şifresi:", type="password")
            if st.button("Sistemi Yönet"):
                if admin_pw == ADMIN_PASSWORD:
                    st.session_state.logged_in = True
                    st.rerun()
        else:
            st.success("Yönetici Yetkisi Aktif")
            
            # --- ÖĞRENME BUTONU ---
            st.subheader("🤖 YZ Sürekli Öğrenme")
            if st.button("🚀 MODELİ YENİ VERİLERLE GÜNCELLE"):
                with st.spinner("Yapay Zeka Yeni Bilgileri Hafızasına Alıyor..."):
                    success, msg = train_with_collected_data(model, scaler)
                    if success: st.success(msg); st.balloons()
                    else: st.warning(msg)
            
            st.divider()
            if st.button("Oturumu Kapat"):
                st.session_state.logged_in = False
                st.rerun()
            if st.button("🗑️ Arşivi Temizle"):
                for f in glob.glob(f"{SAVE_DIR}/*.png"): os.remove(f)
                st.rerun()

    # --- ANA ANALİZ BÖLÜMÜ ---
    uploaded = st.file_uploader("Analiz İçin Termal Görüntü Yükleyin", type=["jpg", "png", "jpeg"])

    if uploaded and user_name:
        pil_img = Image.open(uploaded).convert("RGB")
        with st.spinner("Analiz Yapılıyor..."):
            # Arka plan temizleme ve maskeleme
            nobg = remove(pil_img).convert("RGB")
            img_np = np.array(nobg)
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            
            # Maske ve Isı Haritası
            _, mask = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)
            heatmap = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
            heatmap = cv2.bitwise_and(heatmap, heatmap, mask=mask)
            
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            res_img = np.array(pil_img).copy()
            detaylar = []
            
            for cnt in contours:
                if cv2.contourArea(cnt) < 1500: continue
                x, y, w, h = cv2.boundingRect(cnt)
                crop = gray[y:y+h, x:x+w]
                
                # Tahmin
                feats = extract_features(crop)
                f_scaled = scaler.transform(feats)
                pred = model.predict(f_scaled)[0]
                prob = model.predict_proba(f_scaled)[0]
                
                label = "PESTISITLI" if pred == 1 else "TEMIZ"
                color = (0, 0, 255) if pred == 1 else (0, 255, 0)
                
                detaylar.append({
                    "Nesne": f"Örnek {len(detaylar)+1}",
                    "Durum": label,
                    "Güven": f"%{max(prob)*100:.1f}",
                    "Renk": "#fee2e2" if pred == 1 else "#dcfce7",
                    "Border": "#991b1b" if pred == 1 else "#166534"
                })
                
                cv2.rectangle(res_img, (x, y), (x+w, y+h), color, 12)
                cv2.putText(res_img, label, (x, y-15), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 4)

            # --- RAPORLAMA ---
            st.subheader("📝 Teknik Analiz Raporu")
            c1, c2, c3 = st.columns(3)
            with c1: st.image(uploaded, caption="Ham Giriş", use_container_width=True)
            with c2: st.image(heatmap, caption="Maskeli Heatmap", use_container_width=True)
            with c3: st.image(res_img, caption="Teşhis", use_container_width=True)
            
            st.divider()
            
            if detaylar:
                cols = st.columns(len(detaylar))
                for i, d in enumerate(detaylar):
                    with cols[i]:
                        st.markdown(f"""
                            <div style="background-color:{d['Renk']}; padding:20px; border-radius:15px; border:3px solid {d['Border']}; text-align:center;">
                                <h2 style="color:{d['Border']}; margin:0;">{d['Nesne']}</h2>
                                <b style="font-size:24px; color:{d['Border']}">{d['Durum']}</b><br>
                                <span style="color:#333;">Güven Oranı: <b>{d['Güven']}</b></span>
                            </div>
                        """, unsafe_allow_html=True)

            # OTOMATİK KAYIT (Gelecekteki eğitimler için)
            p_say = sum(1 for d in detaylar if d["Durum"] == "PESTISITLI")
            s_say = len(detaylar) - p_say
            clean_name = "".join(x for x in user_name if x.isalnum())
            save_path = f"{SAVE_DIR}/{int(time.time())}_{clean_name}_{p_say}_{s_say}.png"
            Image.fromarray(res_img).save(save_path)
