import streamlit as st
from streamlit_autorefresh import st_autorefresh # Yeni kütüphane

# ... (Önceki kütüphane ve model yükleme kısımları) ...

# --- ANIMASYONLU SUNUM AYARLARI ---
# 3 dakika (180.000 ms) boyunca etkileşim olmazsa sayfayı yenileyip modu değiştirir
if 'last_action' not in st.session_state:
    st.session_state.last_action = time.time()
if 'show_presentation' not in st.session_state:
    st.session_state.show_presentation = False

# JavaScript ile 3 dakika hareketsizliği kontrol etme (Basit Mantık)
components_code = """
<script>
    var timeout;
    function resetTimer() {
        clearTimeout(timeout);
        timeout = setTimeout(function() {
            window.parent.postMessage({type: 'timeout'}, '*');
        }, 180000); // 180 saniye
    }
    window.onload = resetTimer;
    window.onmousemove = resetTimer;
    window.onmousedown = resetTimer; 
    window.ontouchstart = resetTimer;
    window.onclick = resetTimer;
    window.onkeydown = resetTimer;
</script>
"""
st.components.v1.html(components_code, height=0)

# --- SUNUM MODU (ANIMASYON) ---
if st.session_state.show_presentation:
    st.markdown("""
        <style>
        .presentation-box {
            background-color: #0e1117;
            color: white;
            padding: 50px;
            border-radius: 15px;
            text-align: center;
            border: 2px solid #FF4B4B;
            animation: fadeIn 2s;
        }
        @keyframes fadeIn { from {opacity: 0;} to {opacity: 1;} }
        </style>
    """, unsafe_allow_html=True)

    # Otomatik slayt geçişi (Her 5 saniyede bir)
    count = st_autorefresh(interval=5000, key="prescounter")
    
    slides = [
        {"t": "🎯 Proje Amacı", "c": "Tarımsal ürünlerdeki pestisit kalıntılarını termal görüntüleme ve YZ ile hızlıca tespit etmek."},
        {"t": "🔍 Yöntem", "c": "Termal kameralardan alınan veriler, Random Forest algoritması ile analiz edilir."},
        {"t": "📊 Veri İşleme", "c": "Görüntüdeki arka plan AI ile temizlenir ve elma dokusundaki sıcaklık farkları incelenir."},
        {"t": "✅ Sonuç", "c": "Geleneksel yöntemlere göre %90 daha hızlı ve düşük maliyetli analiz imkanı sağlar."}
    ]
    
    current_slide = slides[count % len(slides)]
    
    st.markdown(f"""
        <div class='presentation-box'>
            <h1>{current_slide['t']}</h1>
            <hr>
            <p style='font-size: 24px;'>{current_slide['c']}</p>
            <br><br>
            <small>Devam etmek için ekrana dokunun veya fareyi hareket ettirin.</small>
        </div>
    """, unsafe_allow_html=True)
    
    if st.button("Analiz Moduna Dön"):
        st.session_state.show_presentation = False
        st.rerun()
    st.stop() # Sayfanın geri kalanını yükleme

# --- NORMAL ANALİZ MODU BURADAN DEVAM EDER ---
# ...
