import streamlit as st
import numpy as np
import cv2
import pickle
import os
from PIL import Image
from rembg import remove
from skimage.feature import graycomatrix, graycoprops
from scipy.stats import skew, kurtosis
import io
import time

# ==========================================
# 1. ÖZELLİK ÇIKARMA (PROJENİN FİZİKSEL TEMELİ)
# ==========================================
def extract_features(img_gray):
    # Görüntüyü normalize et (0-255 arası)
    img_8 = cv2.normalize(img_gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # GLCM (Gri Seviye Eş Oluşum Matrisi) ile Doku Analizi
    glcm = graycomatrix(img_8, distances=[1], angles=[0], symmetric=True, normed=True)
    
    glcm_features = [
        graycoprops(glcm, 'contrast')[0, 0],
        graycoprops(glcm, 'dissimilarity')[0, 0],
        graycoprops(glcm, 'homogeneity')[0, 0],
        graycoprops(glcm, 'energy')[0, 0],
        graycoprops(glcm, 'correlation')[0, 0],
        graycoprops(glcm, 'ASM')[0, 0]
    ]
    
    # İstatistiksel Özellikler (Histogram)
    flat = img_gray.flatten()
    hist_features = [np.mean(flat), np.std(flat), skew(flat), kurtosis(flat)]
    
    # Kenar Yoğunluğu (Canny)
    edges = cv2.Canny(img_gray, 50, 150)
    edge_density = np.sum(edges > 0) / edges.size
    
    # Tüm özellikleri birleştir (Modelin beklediği 11 özellik)
    features = glcm_features + hist_features + [edge_density]
    return np.array(features).reshape(1, -1)

# ==========================================
# 2. OTOMATİK MODEL YÜKLEME (İNTERNET UYUMLU)
# ==========================================
@st.cache_resource
def load_trained_assets():
    base_path = os.path.dirname(__file__)
    m_path = os.path.join(base_path, "rf_model.pkl")
    s_path = os.path.join(base_path, "scaler.pkl")
    
    if os.path.exists(m_path) and os.path.exists(s_path):
        try:
            with open(m_path, "rb") as f:
                model = pickle.load(f)
            with open(s_path, "rb") as f:
                scaler = pickle.load(f)
            return model, scaler
        except Exception as e:
            st.error(f"Model yükleme hatası: {e}")
    return None, None

# ==========================================
# 3. VERİ TABANI SİSTEMİ (SESSION STATE)
# ==========================================
if 'analiz_db' not in st.session_state:
    st.session_state.analiz_db = []

# ==========================================
# 4. SAYFA TASARIMI (MENÜ VE İÇERİK)
# ==========================================
st.set_page_config(page_title="Pestisit Analiz Lab", page_icon="🍎", layout="wide")

with st.sidebar:
    st.title("🍀 Proje Menüsü")
    sayfa = st.radio("Bölüm Seçiniz:", ["🏠 Hakkımızda & Başarılar", "🔬 Termal Analiz Sistemi"])
    st.divider()
    st.info("📍 Fizik Alanı / Gıda Güvenliği Çalışması")
    if st.button("Veri Tabanını Sıfırla"):
        st.session_state.analiz_db = []
        st.rerun()

# --- SAYFA 1: HAKKIMIZDA ---
if sayfa == "🏠 Hakkımızda & Başarılar":
    st.title("🏆 Başarı Yolculuğumuz ve Proje Tanıtımı")
    
    col_img, col_txt = st.columns([1, 2])
    with col_img:
        # Ödül belgesi veya ekip fotoğrafı için yer tutucu
        st.image("https://81duzcehaber.com/resimler/haberler/duzceden-tubitakta-turkiye-derecesi-13619.jpg", 
                 caption="TÜBİTAK 2204-B Ödül Töreni")
    
    with col_txt:
        st.subheader("🥉 Geçmiş Başarılarımız")
        st.write("""
        Projemiz daha önce **TÜBİTAK 2204-B** yarışmasında büyük bir başarı elde etmiştir:
        - **Bölge Birinciliği**
        - **Türkiye Üçüncülüğü** (Coğrafya alanında)
        
        Şu an ise projemizi **Fizik** prensipleriyle (Termal Bariyer Etkisi) geliştirerek daha ileriye taşıyoruz. 
        Yeni dönemin sonuçları heyecanla beklenmektedir!
        """)
        st.link_button("📰 Haber Detayı İçin Tıklayın", "https://81duzcehaber.com/duzceden-tubitakta-turkiye-derecesi-13619")

    st.divider()
    st.subheader("🧪 Fiziksel Temel: Termal Bariyer Etkisi")
    st.write("""
    Pestisitler meyve yüzeyinde ince bir film oluşturarak ısının dışarı atılmasını zorlaştırır. 
    Bu 'ısı bariyeri', termal kamerada doku farklılığı olarak tespit edilir.
    """)
    

# --- SAYFA 2: ANALİZ SİSTEMİ ---
else:
    st.title("🚀 Termal Analiz ve Laboratuvar Kayıtları")
    
    model, scaler = load_trained_assets()
    
    if model is None:
        st.warning("⚠️ 'rf_model.pkl' ve 'scaler.pkl' dosyaları sunucuda bulunamadı. Lütfen ana dizine ekleyip GitHub'a yükleyin.")
    else:
        st.success("✅ Yapay Zeka Modeli Aktif. Analize Hazır!")
        
        uploaded_file = st.file_uploader("Termal Görüntü Yükleyin", type=["jpg", "png", "jpeg"])
        
        if uploaded_file:
            pil_img = Image.open(uploaded_file).convert("RGB")
            
            with st.spinner("Termal imza analiz ediliyor..."):
                # Görüntü İşleme: Arka Plan Temizleme ve Kontur Bulma
                nobg = remove(pil_img).convert("RGB")
                gray = cv2.cvtColor(np.array(nobg), cv2.COLOR_RGB2GRAY)
                blur = cv2.GaussianBlur(gray, (5, 5), 0)
                _, th = cv2.threshold(blur, 10, 255, cv2.THRESH_BINARY)
                contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                res_img = np.array(pil_img).copy()
                p_sayac, s_sayac = 0, 0
                
                for cnt in contours:
                    if cv2.contourArea(cnt) < 1000: continue
                    x, y, w, h = cv2.boundingRect(cnt)
                    crop = gray[y:y+h, x:x+w]
                    
                    # Özellik Çıkarımı ve Tahmin
                    feats = extract_features(crop)
                    feats_scaled = scaler.transform(feats)
                    prediction = model.predict(feats_scaled)[0]
                    
                    if prediction == 1:
                        p_sayac += 1
                        label, color = "PESTISITLI", (255, 0, 0)
                    else:
                        s_sayac += 1
                        label, color = "PESTISITSIZ", (0, 255, 0)
                    
                    cv2.rectangle(res_img, (x, y), (x+w, y+h), color, 12)
                    cv2.putText(res_img, label, (x, y-15), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 4)

                # Veri Tabanına Kaydet (Maks 100)
                if len(st.session_state.analiz_db) >= 100:
                    st.session_state.analiz_db.pop(-1)
                
                st.session_state.analiz_db.insert(0, {
                    "resim": res_img,
                    "p": p_sayac,
                    "s": s_sayac,
                    "saat": time.strftime("%H:%M:%S")
                })

                # Sonuç Paneli
                col_res, col_metric = st.columns([2, 1])
                with col_res:
                    st.image(res_img, caption="Anlık Analiz Sonucu", use_container_width=True)
                with col_metric:
                    st.metric("🔴 PESTİSİTLİ ÖRNEK", p_sayac)
                    st.metric("🟢 PESTİSİTSİZ ÖRNEK", s_sayac)
                    st.info(f"Analiz başarıyla tamamlandı. Kayıt veri tabanına eklendi.")

        # --- ARŞİV BÖLÜMÜ ---
        st.divider()
        if st.session_state.analiz_db:
            total_p = sum(i['p'] for i in st.session_state.analiz_db)
            total_s = sum(i['s'] for i in st.session_state.analiz_db)
            
            st.subheader(f"📂 Analiz Veri Tabanı ({len(st.session_state.analiz_db)} / 100)")
            m1, m2, m3 = st.columns(3)
            m1.metric("Toplam Analiz", len(st.session_state.analiz_db))
            m2.metric("Toplam Pestisitli", total_p)
            m3.metric("Toplam Pestisitsiz", total_s)

            st.markdown("---")
            gal_cols = st.columns(4)
            for idx, item in enumerate(st.session_state.analiz_db):
                with gal_cols[idx % 4]:
                    st.image(item["resim"], use_container_width=True)
                    st.caption(f"🕒 {item['saat']} | 🔴 {item['p']} | 🟢 {item['s']}")
        else:
            st.info("Sistem hazır. Analiz yapmak için termal bir fotoğraf yükleyin.")
