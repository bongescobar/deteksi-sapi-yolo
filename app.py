import streamlit as st
import cv2
from PIL import Image
import numpy as np
from ultralytics import YOLO
import base64
from io import BytesIO

# =========================================================
# KONFIGURASI HALAMAN
# =========================================================
st.set_page_config(
    page_title="SapiDeteksi",
    page_icon="🐄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================================
# CUSTOM CSS
# =========================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=Inter:wght@300;400;500;600&display=swap');

/* ---- ROOT VARIABLES ---- */
:root {
    --green-dark:   #1A3C34;
    --green-mid:    #2D6A4F;
    --green-light:  #52B788;
    --cream:        #F7F3ED;
    --cream-dark:   #EDE8DF;
    --gold:         #C8972B;
    --gold-light:   #F0C96B;
    --brown:        #3D1C02;
    --text-main:    #1C1C1C;
    --text-muted:   #6B6B6B;
    --white:        #FFFFFF;
    --radius:       12px;
    --shadow:       0 4px 20px rgba(0,0,0,0.08);
}

/* ---- GLOBAL ---- */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: var(--cream) !important;
    color: var(--text-main);
}

/* ---- SIDEBAR ---- */
[data-testid="stSidebar"] {
    background: var(--green-dark) !important;
    border-right: none;
}
[data-testid="stSidebar"] * {
    color: var(--white) !important;
}
[data-testid="stSidebar"] .stRadio label {
    font-size: 15px !important;
    padding: 8px 0 !important;
    letter-spacing: 0.02em;
}
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] .stSelectbox label {
    font-size: 13px !important;
    color: var(--gold-light) !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
[data-testid="stSidebar"] .stSlider [data-baseweb="slider"] div[role="slider"] {
    background: var(--gold) !important;
}
.sidebar-logo {
    text-align: center;
    padding: 24px 0 16px 0;
    border-bottom: 1px solid rgba(255,255,255,0.12);
    margin-bottom: 20px;
}
.sidebar-logo .logo-text {
    font-family: 'Playfair Display', serif;
    font-size: 22px;
    font-weight: 700;
    color: var(--white);
    letter-spacing: 0.03em;
    line-height: 1.2;
}
.sidebar-logo .logo-sub {
    font-size: 11px;
    color: rgba(255,255,255,0.55);
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-top: 4px;
}

/* ---- HERO SECTION (BERANDA) ---- */
.hero-section {
    background: linear-gradient(135deg, var(--green-dark) 0%, var(--green-mid) 100%);
    border-radius: var(--radius);
    padding: 56px 48px;
    color: var(--white);
    margin-bottom: 32px;
    position: relative;
    overflow: hidden;
}
.hero-section::before {
    content: "🐄";
    position: absolute;
    right: 40px;
    top: 50%;
    transform: translateY(-50%);
    font-size: 120px;
    opacity: 0.12;
    pointer-events: none;
}
.hero-section .hero-eyebrow {
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.18em;
    color: var(--gold-light);
    margin-bottom: 12px;
    font-weight: 500;
}
.hero-section h1 {
    font-family: 'Playfair Display', serif;
    font-size: 42px;
    font-weight: 700;
    line-height: 1.2;
    margin: 0 0 16px 0;
    color: var(--white);
}
.hero-section p {
    font-size: 16px;
    color: rgba(255,255,255,0.75);
    line-height: 1.7;
    max-width: 520px;
    margin: 0;
}

/* ---- BREED QUICK CARDS (BERANDA) ---- */
.breed-chips {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    margin: 28px 0;
}
.breed-chip {
    background: var(--white);
    border: 1.5px solid var(--cream-dark);
    border-radius: 100px;
    padding: 8px 18px;
    font-size: 13px;
    font-weight: 500;
    color: var(--green-dark);
    display: inline-flex;
    align-items: center;
    gap: 6px;
}

/* ---- STEP CARDS (BERANDA) ---- */
.steps-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 16px;
    margin-top: 8px;
}
.step-card {
    background: var(--white);
    border-radius: var(--radius);
    padding: 24px 20px;
    box-shadow: var(--shadow);
    border-top: 3px solid var(--gold);
}
.step-card .step-num {
    font-family: 'Playfair Display', serif;
    font-size: 32px;
    color: var(--cream-dark);
    font-weight: 700;
    line-height: 1;
    margin-bottom: 8px;
}
.step-card .step-title {
    font-size: 14px;
    font-weight: 600;
    color: var(--green-dark);
    margin-bottom: 6px;
}
.step-card .step-desc {
    font-size: 13px;
    color: var(--text-muted);
    line-height: 1.6;
}

/* ---- PAGE TITLE ---- */
.page-title {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 28px;
    padding-bottom: 16px;
    border-bottom: 2px solid var(--cream-dark);
}
.page-title h2 {
    font-family: 'Playfair Display', serif;
    font-size: 28px;
    font-weight: 700;
    color: var(--green-dark);
    margin: 0;
}
.page-title .page-icon {
    font-size: 28px;
}

/* ---- DETEKSI - UPLOAD ZONE ---- */
.upload-hint {
    background: var(--white);
    border: 2px dashed var(--green-light);
    border-radius: var(--radius);
    padding: 20px;
    text-align: center;
    color: var(--text-muted);
    font-size: 14px;
    margin-bottom: 12px;
}

/* ---- DETEKSI - RESULT BOX ---- */
.result-header {
    background: var(--green-dark);
    color: var(--white);
    border-radius: var(--radius) var(--radius) 0 0;
    padding: 14px 20px;
    font-size: 14px;
    font-weight: 600;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}
.breed-result-card {
    background: var(--white);
    border-radius: 0 0 var(--radius) var(--radius);
    border: 1px solid var(--cream-dark);
    border-top: none;
    padding: 20px;
    margin-bottom: 16px;
}

/* ---- INFO BREED CARDS ---- */
.breed-card {
    background: var(--white);
    border-radius: var(--radius);
    box-shadow: var(--shadow);
    overflow: hidden;
    margin-bottom: 24px;
    border: 1px solid var(--cream-dark);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.breed-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 30px rgba(0,0,0,0.12);
}
.breed-card-header {
    background: linear-gradient(90deg, var(--green-dark), var(--green-mid));
    padding: 16px 20px;
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.breed-card-header .breed-name {
    font-family: 'Playfair Display', serif;
    font-size: 22px;
    font-weight: 700;
    color: var(--white);
}
.breed-card-header .breed-type-badge {
    background: rgba(255,255,255,0.15);
    color: var(--gold-light);
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    padding: 4px 12px;
    border-radius: 100px;
    border: 1px solid rgba(255,255,255,0.2);
}
.breed-card-img {
    width: 100%;
    height: 220px;
    object-fit: cover;
    display: block;
}
.breed-card-img-placeholder {
    width: 100%;
    height: 220px;
    background: linear-gradient(135deg, var(--cream-dark), var(--cream));
    display: flex;
    align-items: center;
    justify-content: center;
    flex-direction: column;
    color: var(--text-muted);
    font-size: 13px;
    gap: 8px;
}
.breed-card-body {
    padding: 20px;
}
.info-row {
    display: flex;
    gap: 8px;
    margin-bottom: 12px;
    align-items: flex-start;
    font-size: 14px;
    line-height: 1.6;
}
.info-label {
    min-width: 120px;
    font-weight: 600;
    color: var(--green-dark);
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    padding-top: 2px;
}
.info-value {
    color: var(--text-main);
    flex: 1;
}

/* ---- CONFIDENCE BADGE ---- */
.conf-badge {
    display: inline-block;
    background: var(--green-light);
    color: var(--white);
    font-size: 12px;
    font-weight: 600;
    padding: 3px 10px;
    border-radius: 100px;
    margin-left: 8px;
}

/* ---- SECTION LABEL ---- */
.section-label {
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: var(--gold);
    margin-bottom: 4px;
}
.section-heading {
    font-family: 'Playfair Display', serif;
    font-size: 22px;
    font-weight: 700;
    color: var(--green-dark);
    margin-bottom: 16px;
}

/* ---- HIDE STREAMLIT DEFAULTS ---- */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2rem; padding-bottom: 2rem; }
</style>
""", unsafe_allow_html=True)

# =========================================================
# MODEL
# =========================================================
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# =========================================================
# DATA JENIS SAPI
# =========================================================
INFO_SAPI = {
    "Brangus": {
        "emoji": "🐂",
        "tipe": "Sapi Potong",
        "asal": "Amerika Serikat (persilangan Angus × Brahman)",
        "ciri_fisik": "Tubuh hitam pekat, kompak dan berotot, gelambir tidak terlalu besar, telinga sedang.",
        "keunggulan": "Tahan iklim panas dan tropis, daya tahan tubuh kuat, kualitas daging baik.",
        "pemanfaatan": "Penghasil daging (sapi potong).",
    },
    "Friesian Holstein": {
        "emoji": "🐄",
        "tipe": "Sapi Perah",
        "asal": "Belanda (wilayah Friesland dan Holstein)",
        "ciri_fisik": "Corak belang hitam-putih khas, tubuh besar, ambing susu berkembang baik, kaki panjang.",
        "keunggulan": "Produksi susu tertinggi di antara semua breed, sangat populer di peternakan susu.",
        "pemanfaatan": "Penghasil susu (sapi perah).",
    },
    "Limousin": {
        "emoji": "🐂",
        "tipe": "Sapi Potong",
        "asal": "Prancis (wilayah Limousin)",
        "ciri_fisik": "Warna cokelat keemasan hingga cokelat tua, otot tubuh sangat besar terutama di paha dan punggung, tanduk melengkung.",
        "keunggulan": "Pertumbuhan otot cepat, rendemen daging tinggi, kandungan lemak relatif rendah.",
        "pemanfaatan": "Penghasil daging premium (sapi potong).",
    },
    "Simmental": {
        "emoji": "🐄",
        "tipe": "Dwiguna",
        "asal": "Swiss (lembah Sungai Simme)",
        "ciri_fisik": "Tubuh cokelat kemerahan dengan wajah, perut, dan kaki berwarna putih, postur besar dan tinggi.",
        "keunggulan": "Pertumbuhan badan cepat, adaptasi baik di berbagai iklim, dapat dimanfaatkan ganda.",
        "pemanfaatan": "Daging dan susu (sapi dwiguna).",
    },
}

# =========================================================
# HELPER
# =========================================================
def pil_to_b64(img: Image.Image, fmt="JPEG") -> str:
    buf = BytesIO()
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode()

# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">
        <div class="logo-text">🐄 SapiDeteksi</div>
        <div class="logo-sub">Identifikasi Jenis Sapi</div>
    </div>
    """, unsafe_allow_html=True)

    halaman = st.radio(
        "Navigasi",
        ("🏠  Beranda", "🔍  Deteksi Sapi", "📖  Informasi Jenis Sapi"),
        label_visibility="hidden",
    )

# =========================================================
# FUNGSI DETEKSI
# =========================================================
def prediksi(image, conf_threshold):
    results = model.predict(source=image, conf=conf_threshold)
    annotated_img = results[0].plot()
    annotated_img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
    st.image(annotated_img_rgb, caption="Hasil Deteksi", use_container_width=True)

    boxes = results[0].boxes
    jumlah_objek = len(boxes)

    if jumlah_objek == 0:
        st.warning("⚠️ Tidak ada sapi yang terdeteksi. Coba turunkan confidence threshold atau gunakan foto yang lebih jelas.")
        return

    st.success(f"✅ Terdeteksi **{jumlah_objek}** objek sapi!")

    kelas_terdeteksi = {}
    for box in boxes:
        cls_id = int(box.cls[0])
        cls_name = model.names[cls_id]
        conf = float(box.conf[0])
        if cls_name not in kelas_terdeteksi or conf > kelas_terdeteksi[cls_name]:
            kelas_terdeteksi[cls_name] = conf

    st.markdown('<div class="section-label" style="margin-top:24px;">Informasi Jenis Terdeteksi</div>', unsafe_allow_html=True)

    for cls_name, conf in kelas_terdeteksi.items():
        info = INFO_SAPI.get(cls_name, {})
        badge_color = "#2D6A4F" if conf >= 0.7 else "#C8972B" if conf >= 0.5 else "#999"

        st.markdown(f"""
        <div class="breed-card" style="margin-bottom:16px;">
            <div class="breed-card-header">
                <div class="breed-name">{info.get('emoji','🐄')} {cls_name}</div>
                <div class="breed-type-badge">{info.get('tipe','—')}</div>
            </div>
            <div class="breed-card-body">
                <div style="display:flex;align-items:center;margin-bottom:14px;">
                    <span style="font-size:13px;color:#6B6B6B;">Confidence</span>
                    <span style="background:{badge_color};color:white;font-size:12px;font-weight:600;
                    padding:3px 12px;border-radius:100px;margin-left:10px;">{conf:.0%}</span>
                </div>
                <div class="info-row"><div class="info-label">Asal</div><div class="info-value">{info.get('asal','—')}</div></div>
                <div class="info-row"><div class="info-label">Ciri Fisik</div><div class="info-value">{info.get('ciri_fisik','—')}</div></div>
                <div class="info-row"><div class="info-label">Keunggulan</div><div class="info-value">{info.get('keunggulan','—')}</div></div>
                <div class="info-row"><div class="info-label">Pemanfaatan</div><div class="info-value">{info.get('pemanfaatan','—')}</div></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# =========================================================
# HALAMAN: BERANDA
# =========================================================
if "🏠" in halaman:
    st.markdown("""
    <div class="hero-section">
        <div class="hero-eyebrow">Teknologi Deteksi Berbasis AI</div>
        <h1>Identifikasi Jenis Sapi<br>Secara Otomatis</h1>
        <p>Unggah foto atau gunakan kamera untuk mendeteksi dan mengklasifikasikan jenis sapi secara instan menggunakan model YOLOv8 yang telah dilatih khusus.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-label">Jenis Sapi yang Dapat Dikenali</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="breed-chips">
        <div class="breed-chip">🐂 Brangus</div>
        <div class="breed-chip">🐄 Friesian Holstein</div>
        <div class="breed-chip">🐂 Limousin</div>
        <div class="breed-chip">🐄 Simmental</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-label" style="margin-top:32px;">Cara Menggunakan</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="steps-grid">
        <div class="step-card">
            <div class="step-num">01</div>
            <div class="step-title">Buka Halaman Deteksi</div>
            <div class="step-desc">Pilih menu "Deteksi Sapi" pada sidebar navigasi di sebelah kiri.</div>
        </div>
        <div class="step-card">
            <div class="step-num">02</div>
            <div class="step-title">Upload atau Foto Langsung</div>
            <div class="step-desc">Pilih metode input — unggah foto dari perangkat atau ambil foto langsung via kamera.</div>
        </div>
        <div class="step-card">
            <div class="step-num">03</div>
            <div class="step-title">Lihat Hasil Deteksi</div>
            <div class="step-desc">Model akan menampilkan jenis sapi beserta informasi karakteristiknya secara otomatis.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# =========================================================
# HALAMAN: DETEKSI
# =========================================================
elif "🔍" in halaman:
    st.markdown("""
    <div class="page-title">
        <div class="page-icon">🔍</div>
        <h2>Deteksi Jenis Sapi</h2>
    </div>
    """, unsafe_allow_html=True)

    col_ctrl1, col_ctrl2 = st.columns([1, 1])
    with col_ctrl1:
        metode = st.selectbox("Metode Input", ("📁  Upload Foto", "📷  Kamera (Snapshot)"))
    with col_ctrl2:
        conf_threshold = st.slider("Confidence Threshold", 0.1, 1.0, 0.5, 0.05)

    st.markdown("---")

    if "Upload" in metode:
        file = st.file_uploader("Pilih foto sapi (JPG / PNG)", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
        if file is not None:
            img = Image.open(file).convert("RGB")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown('<div class="section-label">Foto Asli</div>', unsafe_allow_html=True)
                st.image(img, use_container_width=True)
            with col2:
                st.markdown('<div class="section-label">Hasil Deteksi</div>', unsafe_allow_html=True)
                if st.button("🚀 Mulai Deteksi", use_container_width=True, type="primary"):
                    with st.spinner("Mendeteksi..."):
                        results = model.predict(source=img, conf=conf_threshold)
                        annotated_img = results[0].plot()
                        annotated_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
                        st.image(annotated_rgb, use_container_width=True)

            if st.session_state.get("run_detect"):
                pass

            if st.button("🚀 Mulai Deteksi & Tampilkan Info", use_container_width=True):
                with st.spinner("Mendeteksi..."):
                    prediksi(img, conf_threshold)
        else:
            st.markdown("""
            <div class="upload-hint">
                📂 Seret foto ke sini atau klik "Browse files" di atas<br>
                <span style="font-size:12px;">Format: JPG, JPEG, PNG</span>
            </div>
            """, unsafe_allow_html=True)

    elif "Kamera" in metode:
        kamera_file = st.camera_input("Arahkan kamera ke sapi, lalu ambil foto")
        if kamera_file is not None:
            img = Image.open(kamera_file).convert("RGB")
            with st.spinner("Mendeteksi..."):
                prediksi(img, conf_threshold)

# =========================================================
# HALAMAN: INFORMASI JENIS SAPI
# =========================================================
elif "📖" in halaman:
    st.markdown("""
    <div class="page-title">
        <div class="page-icon">📖</div>
        <h2>Informasi Jenis Sapi</h2>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<p style="color:#6B6B6B;margin-bottom:28px;font-size:14px;">Klik <b>Ganti Foto</b> pada tiap kartu untuk mengunggah foto sapi koleksi Anda sendiri.</p>', unsafe_allow_html=True)

    # Inisialisasi session state untuk foto custom
    for breed in INFO_SAPI:
        key = f"img_{breed.replace(' ', '_')}"
        if key not in st.session_state:
            st.session_state[key] = None

    # Tampilkan 2 kolom kartu breed
    breed_list = list(INFO_SAPI.items())
    for i in range(0, len(breed_list), 2):
        cols = st.columns(2, gap="large")
        for j, col in enumerate(cols):
            if i + j >= len(breed_list):
                break
            nama, info = breed_list[i + j]
            img_key = f"img_{nama.replace(' ', '_')}"

            with col:
                # Header kartu
                st.markdown(f"""
                <div class="breed-card-header" style="border-radius:12px 12px 0 0;">
                    <div class="breed-name">{info['emoji']} {nama}</div>
                    <div class="breed-type-badge">{info['tipe']}</div>
                </div>
                """, unsafe_allow_html=True)

                # Foto: custom atau placeholder
                if st.session_state[img_key] is not None:
                    st.image(st.session_state[img_key], use_container_width=True)
                else:
                    st.markdown(f"""
                    <div class="breed-card-img-placeholder">
                        <span style="font-size:40px;">{info['emoji']}</span>
                        <span>Belum ada foto — unggah di bawah</span>
                    </div>
                    """, unsafe_allow_html=True)

                # Upload foto
                uploaded = st.file_uploader(
                    f"Ganti foto {nama}",
                    type=["jpg", "jpeg", "png"],
                    key=f"upload_{img_key}",
                    label_visibility="collapsed",
                )
                if uploaded:
                    st.session_state[img_key] = Image.open(uploaded).convert("RGB")
                    st.rerun()

                col_btn1, col_btn2 = st.columns(2)
                with col_btn1:
                    if st.button(f"📷 Ganti Foto", key=f"btn_upload_{nama}", use_container_width=True):
                        pass  # trigger file uploader di atas
                with col_btn2:
                    if st.session_state[img_key] and st.button("🗑️ Hapus Foto", key=f"btn_del_{nama}", use_container_width=True):
                        st.session_state[img_key] = None
                        st.rerun()

                # Info kartu
                st.markdown(f"""
                <div style="background:white;border:1px solid #EDE8DF;border-top:none;
                    border-radius:0 0 12px 12px;padding:20px;margin-bottom:8px;">
                    <div class="info-row"><div class="info-label">Asal</div><div class="info-value">{info['asal']}</div></div>
                    <div class="info-row"><div class="info-label">Ciri Fisik</div><div class="info-value">{info['ciri_fisik']}</div></div>
                    <div class="info-row"><div class="info-label">Keunggulan</div><div class="info-value">{info['keunggulan']}</div></div>
                    <div class="info-row"><div class="info-label">Pemanfaatan</div><div class="info-value">{info['pemanfaatan']}</div></div>
                </div>
                """, unsafe_allow_html=True)