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
st.set_page_config(page_title="Pestisit Analiz & Eğitim", layout="wide")
SAVE_DIR = "analiz_havuzu"
MODEL_FILE = "rf_model.pkl"
SCALER_FILE = "scaler.pkl"

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

# 2. ÖZELLİK ÇIKARMA (Eski Model Uyumlu - 11 Parametre)
def extract_features(img_gray):
    img_gray = cv2.medianBlur(img_gray, 5)
    img_8 = cv2.normalize(img_gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # GLCM Özellikleri (6 adet)
    glcm = graycomatrix(img_8, distances=[1], angles=[0], symmetric=True, normed=True)
    glcm_feats = [graycoprops(glcm, p)[0, 0] for p in ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']]
    
    # İstatistiksel Özellikler (4 adet)
    flat = img_gray.flatten()
    stats = [np.mean(flat), np.std(flat), skew(flat), kurtosis(flat)]
    
    # Kenar Yoğunluğu (1 adet)
    edges = cv2.Canny(img_8, 50, 150)
    density = np.sum(edges > 0) / edges.size
    
    return np.array(glcm_feats + stats + [density]).reshape(1, -1)

# 3. MEVCUT MODELE EKLEME YAPMA FONKSİYONU
def update_existing_model(model, scaler):
    files = glob.glob(f"{SAVE_DIR}/*.png")
    if len(files) < 3:
        return False, "Eğitim için havuzda en az 3 fotoğraf olmalı."
    
    X_new, y_new = [], []
    for f in files:
        try:
            # Dosya adından etiketi çöz (Örn: timestamp_isim_1_0.png -> 1=Pestisitli)
            parts = os.path.basename(f).split("_")
            label = 1 if int(parts[2]) > int(parts[3]) else 0
            
            img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                feats = extract_features(img)
                X_new.append(feats[0])
                y_new.append(label)
        except: continue
    
    if len(X_new) > 0:
        # Mevcut modelin üzerine yeni verilerle fit (eğitim) yapıyoruz
        model.fit(X_new, y_new) 
        with open(MODEL_FILE, "wb") as f_m: pickle.dump(model, f_m)
        return True, f"Model {len(X_new)} yeni veriyle güncellendi!"
    return False, "Veri işlenemedi."

# 4. YÜKLEME
@st.cache_resource
def load_assets():
    if os.path.exists(MODEL_FILE) and os.path.exists(SCALER_FILE):
        with open(MODEL_FILE, "rb") as f: m = pickle.load(f)
        with open(SCALER_FILE, "rb") as f: s = pickle.load(f)
        return m, s
    return None

assets = load_assets()
if assets:
    model, scaler = assets
    
    with st.sidebar:
        st.header("⚙️ Yönetim")
        user = st.text_input("Operatör:", "Muhammed")
        # Hassasiyet Ayarı: Hep temiz diyorsa 0.3'e çekin
        sens = st.slider("Tespit Hassasiyeti", 0.1, 0.9, 0.45)
        
        st.divider()
        if st.button("🚀 Mevcut Modeli Güncelle"):
            ok, msg = update_existing_model(model, scaler)
            st.info(msg)

    # --- ANALİZ ---
    up = st.file_uploader("Fotoğraf Yükle", type=["jpg", "png", "jpeg"])
    if up:
        img_orig = Image.open(up).convert("RGB")
        nobg = remove(img_orig).convert("RGB")
        gray = cv2.cvtColor(np.array(nobg), cv2.COLOR_RGB2GRAY)
        
        _, mask = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        res_img = np.array(img_orig).copy()
        p_count, s_count = 0, 0
        
        for cnt in contours:
            if cv2.contourArea(cnt) < 1500: continue
            x,y,w,h = cv2.boundingRect(cnt)
            crop = gray[y:y+h, x:x+w]
            
            f_scaled = scaler.transform(extract_features(crop))
            probs = model.predict_proba(f_scaled)[0]
            
            # Sürgüden gelen hassasiyete göre karar ver
            pred = 1 if probs[1] >= sens else 0
            
            label = "PESTISITLI" if pred == 1 else "TEMIZ"
            color = (0,0,255) if pred == 1 else (0,255,0)
            if pred == 1: p_count += 1
            else: s_count += 1
            
            cv2.rectangle(res_img, (x,y), (x+w,y+h), color, 10)
            cv2.putText(res_img, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

        st.image(res_img, caption="Analiz Sonucu")
        
        # Gelecekteki eğitimler için otomatik kaydet
        save_path = f"{SAVE_DIR}/{int(time.time())}_{user}_{p_count}_{s_count}.png"
        Image.fromarray(res_img).save(save_path)
