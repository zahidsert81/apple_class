import streamlit as st
import os
import time
import numpy as np
import cv2
import pickle
import glob
import pandas as pd
from PIL import Image
from rembg import remove
from skimage.feature import graycomatrix, graycoprops
from scipy.stats import skew, kurtosis

# 1. AYARLAR
st.set_page_config(page_title="Pestisit Tespit Sistemi", page_icon="🍎", layout="wide")
SAVE_DIR = "analiz_havuzu"
MODEL_FILE = "rf_model.pkl"
SCALER_FILE = "scaler.pkl"
ADMIN_PASSWORD = "1234"

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

# 2. TEKNİK ÖZELLİK ÇIKARMA (Modelin başarısı buraya bağlı)
def extract_features(img_gray):
    # Gürültü giderme (Median Filter)
    img_gray = cv2.medianBlur(img_gray, 5)
    img_8 = cv2.normalize(img_gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Doku Analizi (GLCM)
    glcm = graycomatrix(img_8, distances=[1], angles=[0], symmetric=True, normed=True)
    glcm_features = [
        graycoprops(glcm, 'contrast')[0, 0],
        graycoprops(glcm, 'dissimilarity')[0, 0],
        graycoprops(glcm, 'homogeneity')[0, 0],
        graycoprops(glcm, 'energy')[0, 0]
    ]
    
    # İstatistiksel Özellikler
    flat = img_gray.flatten()
    stats = [np.mean(flat), np.std(flat), skew(flat)]
    
    return np.array(glcm_features + stats).reshape(1, -1)

# 3. MODELİ YENİDEN EĞİTME FONKSİYONU
def train_model_with_new_data(model, scaler):
    all_files = glob.glob(f"{SAVE_DIR}/*.png")
    if len(all_files) < 3:
        return False, "Eğitim için havuzda yeterli (en az 3) fotoğraf yok."
    
    X_train, y_train = [], []
    for f in all_files:
        try:
            # Dosya isminden etiketi çöz (Örn: 171000_Muhammed_1_0.png -> Pestisitli)
            parts = os.path.basename(f).split("_")
            label = 1 if int(parts[2]) > int(parts[3]) else 0
            
            img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                # Fotoğrafın içinden sadece elma olan bölgeleri bul ve özellik çıkar
                feats = extract_features(img)
                X_train.append(feats[0])
                y_train.append(label)
        except: continue
    
    if len(X_train) > 0:
        model.fit(X_train, y_train) # Modeli eğit
        with open(MODEL_FILE, "wb") as f: pickle.dump(model, f)
        return True, f"Model {len(X_train)} yeni örnekle güncellendi!"
    return False, "Özellik çıkarma başarısız oldu."

# 4. YÜKLEME VE ARAYÜZ
@st.cache_resource
def load_assets():
    if os.path.exists(MODEL_FILE) and os.path.exists(SCALER_FILE):
        with open(MODEL_FILE, "rb") as f: m = pickle.load(f)
        with open(SCALER_FILE, "rb") as f: s = pickle.load(f)
        return m, s
    return None

# --- ÜST LOGOLAR ---
col_l, col_c, col_r = st.columns([1,3,1])
with col_l:
    if os.path.exists("TÜBİTAK_logo.svg.png"): st.image("TÜBİTAK_logo.svg.png", width=100)
with col_c:
    st.markdown("<h1 style='text-align: center;'>PESTİSİT TESPİT SİSTEMİ</h1>", unsafe_allow_html=True)
with col_r:
    if os.path.exists("images.jpg"): st.image("images.jpg", width=100)

assets = load_assets()
if assets:
    model, scaler = assets
    
    with st.sidebar:
        st.header("⚙️ Ayarlar")
        user = st.text_input("Operatör:", "Muhammed")
        if 'admin' not in st.session_state: st.session_state.admin = False
        
        if not st.session_state.admin:
            pw = st.text_input("Şifre:", type="password")
            if st.button("Yönetici Giriş"):
                if pw == ADMIN_PASSWORD: st.session_state.admin = True; st.rerun()
        else:
            st.success("Yönetici Yetkisi")
            if st.button("🚀 MODELİ GÜNCELLE"):
                ok, msg = train_model_with_new_data(model, scaler)
                st.info(msg)
            if st.button("Çıkış"): st.session_state.admin = False; st.rerun()

    # --- ANALİZ ---
    up = st.file_uploader("Termal Görsel Seç", type=["jpg","jpeg","png"])
    if up:
        img_orig = Image.open(up).convert("RGB")
        # Arka planı sil ve maske oluştur
        nobg = remove(img_orig).convert("RGB")
        gray = cv2.cvtColor(np.array(nobg), cv2.COLOR_RGB2GRAY)
        
        # Maskeleme: Isı haritası sadece elmaların üstünde görünsün
        _, mask = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)
        heatmap = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
        heatmap = cv2.bitwise_and(heatmap, heatmap, mask=mask) # Arka planı siyah yap
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        res_img = np.array(img_orig).copy()
        p_count, s_count = 0, 0
        
        for cnt in contours:
            if cv2.contourArea(cnt) < 1500: continue
            x,y,w,h = cv2.boundingRect(cnt)
            crop = gray[y:y+h, x:x+w]
            
            f = extract_features(crop)
            f_s = scaler.transform(f)
            pred = model.predict(f_s)[0]
            
            label = "PESTISITLI" if pred == 1 else "TEMIZ"
            color = (0,0,255) if pred == 1 else (0,255,0)
            if pred == 1: p_count += 1
            else: s_count += 1
            
            cv2.rectangle(res_img, (x,y), (x+w,y+h), color, 8)
            cv2.putText(res_img, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

        # Görselleri Yan Yana Göster
        c1, c2, c3 = st.columns(3)
        c1.image(img_orig, caption="Girdi")
        c2.image(heatmap, caption="Temiz Heatmap")
        c3.image(res_img, caption="Teşhis")
        
        # Sonucu Kaydet (Eğitim için)
        save_name = f"{SAVE_DIR}/{int(time.time())}_{user}_{p_count}_{s_count}.png"
        Image.fromarray(res_img).save(save_name)
