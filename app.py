import streamlit as st
import os
import time

# Sayfa ayarları
st.set_page_config(page_title="Pestisit Analiz Lab", page_icon="🍎", layout="wide")

# Kütüphane kontrolü
try:
    import numpy as np
    import cv2
    import pickle
    from PIL import Image
    from rembg import remove
    from skimage.feature import graycomatrix, graycoprops
    from scipy.stats import skew, kurtosis
except ImportError as e:
    st.error(f"Kütüphane yükleme hatası: {e}")
    st.stop()

# ==========================================
# PAYLAŞILAN VERİ HAVUZU (Tüm Kullanıcılar İçin)
# ==========================================
# Not: Streamlit Cloud'da uygulama uykuya dalana kadar bu liste bellekte kalır.
if 'global_db' not in st.session_state:
    st.session_state.global_db = []

def display_header():
    col1, col2, col3 = st.columns([1, 4, 1])
    with col1:
        if os.path.exists("TÜBİTAK_logo.svg.png"):
            st.image("TÜBİTAK_logo.svg.png", width=110)
    with col2:
        st.markdown("<h1 style='text-align: center; margin-bottom: 0;'>Termal Analiz ve Pestisit Tespit Sistemi</h1>", unsafe_allow_html=True)
    with col3:
        if os.path.exists("images.jpg"):
            st.image("images.jpg", width=110)
    st.divider()
    k1, k2 = st.columns(2)
    with k1: st.markdown(f"**👨‍💻 Öğrenci:** Muhammed Zahid SERT")
    with k2: st.markdown(f"**👩‍🏫 Danışman:** Emine SARI")

# Özellik çıkarma ve Model yükleme kısımları aynı kalıyor...
# (extract_features ve load_assets fonksiyonlarını buraya ekleyin)

display_header()
# ... Model yükleme kontrolleri ...

# ==========================================
# ANALİZ VE PAYLAŞIM ALANI
# ==========================================

# Kullanıcı adı girişi
user_name = st.text_input("Analiz yapan kişinin adı/rumuzu:", placeholder="Örn: Zahid")

uploaded = st.file_uploader("Analiz için fotoğraf yükleyin", type=["jpg", "png", "jpeg"])

if uploaded and user_name:
    pil_img = Image.open(uploaded).convert("RGB")
    with st.spinner("Analiz ediliyor..."):
        # Görüntü işleme adımları...
        # [Önceki kodlardaki analiz döngüsü burada çalışır]
        
        # Analiz bittiğinde veriyi GLOBAL havuza ekle
        analiz_verisi = {
            "kullanici": user_name,
            "orijinal": pil_img,
            "sonuc": res_img, # Analiz edilmiş resim
            "p": p_say,
            "s": s_say,
            "saat": time.strftime("%H:%M:%S")
        }
        st.session_state.global_db.insert(0, analiz_verisi)
        
        # Listeyi 50 kayıtla sınırla (Belleği şişirmemek için)
        if len(st.session_state.global_db) > 50:
            st.session_state.global_db.pop()

# ==========================================
# TÜM KULLANICILARIN ANALİZLERİ (HERKES GÖREBİLİR)
# ==========================================
st.divider()
st.subheader("🌐 Topluluk Analiz Havuzu (Son 50 Analiz)")

if st.session_state.global_db:
    for item in st.session_state.global_db:
        with st.container():
            st.markdown(f"### 👤 Kullanıcı: {item['kullanici']} | 🕒 {item['saat']}")
            col_a, col_b, col_c = st.columns([2, 2, 1])
            
            with col_a:
                st.image(item["orijinal"], caption="Orijinal Görüntü", use_container_width=True)
            with col_b:
                st.image(item["sonuc"], caption="Analiz Sonucu", use_container_width=True)
            with col_c:
                st.write("**Detaylar:**")
                st.metric("🔴 Pestisitli", item["p"])
                st.metric("🟢 Pestisitsiz", item["s"])
            st.divider()
else:
    st.info("Henüz kimse analiz yapmadı. İlk analizi siz yapın!")
