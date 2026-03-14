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
import plotly.graph_objects as go # Grafik için yeni kütüphane

# 1. SAYFA AYARLARI
st.set_page_config(page_title="Pestisit Analiz Lab", page_icon="🍎", layout="wide")

# ... (extract_features ve load_assets fonksiyonları aynı kalıyor) ...

# 5. SİSTEM ÇALIŞMA ŞEMASI (Yeni Fonksiyon)
def display_workflow():
    st.info("⚙️ **Sistem Çalışma Algoritması**")
    cols = st.columns(5)
    steps = [
        "📸 1. Görüntü Alımı\n(Termal Kamera)",
        "✂️ 2. Arka Plan Silme\n(Rembg-AI)",
        "🧪 3. Özellik Çıkarma\n(GLCM & Histogram)",
        "🤖 4. YZ Sınıflandırma\n(Random Forest)",
        "✅ 5. Karar & Rapor\n(Pestisit Tespiti)"
    ]
    for i, step in enumerate(steps):
        cols[i].markdown(f"<div style='border:1px solid #ddd; padding:10px; border-radius:5px; text-align:center; background:#f9f9f9; height:100px;'>{step}</div>", unsafe_allow_html=True)

# ... (Giriş ve Sidebar kısımları aynı) ...

    # --- ANA ANALİZ ---
    if uploaded:
        if not user_name:
            st.warning("Lütfen adınızı girin.")
        else:
            # ... (Görüntü işleme ve tahmin döngüsü aynı) ...
            
            # --- SONUÇLAR VE GRAFİKLER ---
            st.subheader("📊 Analiz ve İstatistikler")
            
            c_img1, c_img2, c_graph = st.columns([1, 1, 1.2]) # Grafik için 3. sütun
            
            with c_img1: st.image(uploaded, caption="Orijinal", use_container_width=True)
            with c_img2: st.image(res_img, caption="Analiz", use_container_width=True)
            
            with c_graph:
                # Plotly Pasta Grafiği
                if (p_say + s_say) > 0:
                    fig = go.Figure(data=[go.Pie(
                        labels=['Pestisitli', 'Pestisitsiz'],
                        values=[p_say, s_say],
                        hole=.4,
                        marker_colors=['#FF4B4B', '#00CC96']
                    )])
                    fig.update_layout(margin=dict(t=0, b=0, l=0, r=0), height=250, showlegend=True)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.write("Grafik oluşturulacak veri bulunamadı.")

            # Sistem Çalışma Akış Şeması
            st.divider()
            display_workflow()
            
            # Metrikler
            st.divider()
            m1, m2, m3 = st.columns(3)
            m1.metric("🔴 PESTİSİTLİ", f"{p_say} Adet")
            m2.metric("🟢 PESTİSİTSİZ", f"{s_say} Adet")
            m3.metric("⚠️ TESPİT ORANI", f"%{(p_say/(p_say+s_say)*100) if (p_say+s_say)>0 else 0:.1f}")
