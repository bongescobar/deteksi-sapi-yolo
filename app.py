import streamlit as st
import cv2
from PIL import Image
import numpy as np
from ultralytics import YOLO
import os
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av
import threading
import time

# =========================================================
# KONFIGURASI HALAMAN
# =========================================================
st.set_page_config(
    page_title="SapiDeteksi",
    page_icon="🐄",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# =========================================================
# CUSTOM CSS
# =========================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=Inter:wght@300;400;500;600&display=swap');

:root {
    --green-dark:  #1A3C34;
    --green-mid:   #2D6A4F;
    --green-light: #52B788;
    --cream:       #F7F3ED;
    --cream-dark:  #EDE8DF;
    --gold:        #C8972B;
    --gold-light:  #F0C96B;
    --text-main:   #1C1C1C;
    --text-muted:  #6B6B6B;
    --white:       #FFFFFF;
    --radius:      12px;
    --shadow:      0 4px 20px rgba(0,0,0,0.08);
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: var(--cream) !important;
    color: var(--text-main);
}

/* ===== TOP NAV ===== */
.top-nav {
    background: var(--green-dark);
    border-radius: var(--radius);
    padding: 14px 24px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 28px;
}
.top-nav .brand {
    font-family: 'Playfair Display', serif;
    font-size: 20px;
    font-weight: 700;
    color: var(--white);
    letter-spacing: 0.02em;
}
.top-nav .brand span {
    color: var(--gold-light);
}

/* ===== TABS ===== */
[data-testid="stTabs"] {
    margin-bottom: 0 !important;
}
[data-testid="stTabs"] > div:first-child {
    background: var(--white);
    border-radius: var(--radius) var(--radius) 0 0;
    border-bottom: 2px solid var(--cream-dark);
    padding: 0 8px;
}
button[data-baseweb="tab"] {
    font-family: 'Inter', sans-serif !important;
    font-size: 14px !important;
    font-weight: 500 !important;
    color: var(--text-muted) !important;
    padding: 14px 20px !important;
    border-bottom: 3px solid transparent !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: var(--green-dark) !important;
    border-bottom: 3px solid var(--green-dark) !important;
    font-weight: 600 !important;
}
[data-testid="stTabsContent"] {
    background: var(--white);
    border-radius: 0 0 var(--radius) var(--radius);
    padding: 28px 24px !important;
    border: 1px solid var(--cream-dark);
    border-top: none;
}

/* ===== HERO ===== */
.hero-section {
    background: linear-gradient(135deg, var(--green-dark) 0%, var(--green-mid) 100%);
    border-radius: var(--radius);
    padding: 52px 48px;
    color: var(--white);
    margin-bottom: 28px;
    position: relative;
    overflow: hidden;
}
.hero-section::after {
    content: "🐄";
    position: absolute;
    right: 48px; top: 50%;
    transform: translateY(-50%);
    font-size: 110px;
    opacity: 0.10;
    pointer-events: none;
}
.hero-eyebrow {
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.18em;
    color: var(--gold-light);
    margin-bottom: 12px;
    font-weight: 500;
}
.hero-section h1 {
    font-family: 'Playfair Display', serif;
    font-size: 40px;
    font-weight: 700;
    line-height: 1.2;
    margin: 0 0 14px 0;
    color: var(--white);
}
.hero-section p {
    font-size: 15px;
    color: rgba(255,255,255,0.72);
    line-height: 1.75;
    max-width: 500px;
    margin: 0;
}

/* ===== BREED CHIPS ===== */
.breed-chips {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    margin: 24px 0 32px;
}
.breed-chip {
    background: var(--white);
    border: 1.5px solid var(--cream-dark);
    border-radius: 100px;
    padding: 7px 18px;
    font-size: 13px;
    font-weight: 500;
    color: var(--green-dark);
    display: inline-flex;
    align-items: center;
    gap: 6px;
}

/* ===== STEPS ===== */
.steps-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 16px;
}
.step-card {
    background: var(--cream);
    border-radius: var(--radius);
    padding: 22px 18px;
    border-top: 3px solid var(--gold);
}
.step-num {
    font-family: 'Playfair Display', serif;
    font-size: 30px;
    color: var(--cream-dark);
    font-weight: 700;
    line-height: 1;
    margin-bottom: 8px;
}
.step-title {
    font-size: 14px;
    font-weight: 600;
    color: var(--green-dark);
    margin-bottom: 5px;
}
.step-desc {
    font-size: 13px;
    color: var(--text-muted);
    line-height: 1.6;
}

/* ===== SECTION LABEL ===== */
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
    font-size: 20px;
    font-weight: 700;
    color: var(--green-dark);
    margin-bottom: 16px;
}

/* ===== DETEKSI ===== */
.detect-settings {
    background: var(--cream);
    border-radius: var(--radius);
    padding: 16px 20px;
    margin-bottom: 20px;
    display: flex;
    align-items: center;
    gap: 24px;
}
.upload-hint {
    background: var(--cream);
    border: 2px dashed var(--green-light);
    border-radius: var(--radius);
    padding: 32px;
    text-align: center;
    color: var(--text-muted);
    font-size: 14px;
    margin-bottom: 12px;
}

/* ===== RESULT CARD ===== */
.breed-result-card {
    background: var(--cream);
    border-radius: var(--radius);
    border-left: 4px solid var(--green-mid);
    padding: 18px 20px;
    margin-bottom: 12px;
}
.breed-result-name {
    font-family: 'Playfair Display', serif;
    font-size: 18px;
    font-weight: 700;
    color: var(--green-dark);
    margin-bottom: 12px;
    display: flex;
    align-items: center;
    gap: 10px;
}
.info-row {
    display: flex;
    gap: 8px;
    margin-bottom: 8px;
    align-items: flex-start;
    font-size: 13px;
    line-height: 1.6;
}
.info-label {
    min-width: 110px;
    font-weight: 600;
    color: var(--green-dark);
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    padding-top: 2px;
    flex-shrink: 0;
}
.info-value { color: var(--text-main); flex: 1; }

/* ===== BREED INFO CARD ===== */
.breed-card {
    background: var(--white);
    border-radius: var(--radius);
    box-shadow: var(--shadow);
    overflow: hidden;
    margin-bottom: 24px;
    border: 1px solid var(--cream-dark);
    transition: transform 0.2s, box-shadow 0.2s;
}
.breed-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 28px rgba(0,0,0,0.12);
}
.breed-card-header {
    background: linear-gradient(90deg, var(--green-dark), var(--green-mid));
    padding: 14px 18px;
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.breed-card-header .bname {
    font-family: 'Playfair Display', serif;
    font-size: 20px;
    font-weight: 700;
    color: var(--white);
}
.breed-type-badge {
    background: rgba(255,255,255,0.15);
    color: var(--gold-light);
    font-size: 10px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    padding: 4px 12px;
    border-radius: 100px;
    border: 1px solid rgba(255,255,255,0.2);
}
.img-placeholder {
    width: 100%;
    height: 210px;
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
    padding: 18px;
}

/* ===== BADGE ===== */
.badge-high  { background:#2D6A4F; color:white; font-size:12px; font-weight:600; padding:3px 12px; border-radius:100px; }
.badge-mid   { background:#C8972B; color:white; font-size:12px; font-weight:600; padding:3px 12px; border-radius:100px; }
.badge-low   { background:#999;    color:white; font-size:12px; font-weight:600; padding:3px 12px; border-radius:100px; }

/* ===== HIDE DEFAULTS ===== */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="collapsedControl"] { display: none !important; }
.block-container { padding-top: 1.5rem; padding-bottom: 2rem; max-width: 1100px; }
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
# Ganti path foto sesuai nama file di folder images/ di repo
# =========================================================
INFO_SAPI = {
    "Brangus": {
        "emoji": "🐂",
        "tipe": "Sapi Potong",
        "foto": "Brangus.jpg",
        "asal": "Amerika Serikat (persilangan Angus × Brahman)",
        "ciri_fisik": "Tubuh hitam pekat, kompak dan berotot, gelambir tidak terlalu besar, telinga sedang.",
        "keunggulan": "Tahan iklim panas dan tropis, daya tahan tubuh kuat, kualitas daging baik.",
        "pemanfaatan": "Penghasil daging (sapi potong).",
    },
    "Friesian Holstein": {
        "emoji": "🐄",
        "tipe": "Sapi Perah",
        "foto": "Fiesian Holstein.jpg",
        "asal": "Belanda (wilayah Friesland dan Holstein)",
        "ciri_fisik": "Corak belang hitam-putih khas, tubuh besar, ambing susu berkembang baik, kaki panjang.",
        "keunggulan": "Produksi susu tertinggi di antara semua breed, sangat populer di peternakan susu.",
        "pemanfaatan": "Penghasil susu (sapi perah).",
    },
    "Limousin": {
        "emoji": "🐂",
        "tipe": "Sapi Potong",
        "foto": "Limousin.jpg",
        "asal": "Prancis (wilayah Limousin)",
        "ciri_fisik": "Warna cokelat keemasan hingga cokelat tua, otot tubuh sangat besar terutama di paha dan punggung, tanduk melengkung.",
        "keunggulan": "Pertumbuhan otot cepat, rendemen daging tinggi, kandungan lemak relatif rendah.",
        "pemanfaatan": "Penghasil daging premium (sapi potong).",
    },
    "Simmental": {
        "emoji": "🐄",
        "tipe": "Dwiguna",
        "foto": "Simmental.jpg",
        "asal": "Swiss (lembah Sungai Simme)",
        "ciri_fisik": "Tubuh cokelat kemerahan dengan wajah, perut, dan kaki berwarna putih, postur besar dan tinggi.",
        "keunggulan": "Pertumbuhan badan cepat, adaptasi baik di berbagai iklim, dapat dimanfaatkan ganda.",
        "pemanfaatan": "Daging dan susu (sapi dwiguna).",
    },
}

# =========================================================
# HELPER: load foto dari folder images/
# =========================================================
def load_breed_image(path: str):
    """Return PIL Image jika file ada di repo, else None."""
    if path and os.path.exists(path):
        try:
            return Image.open(path).convert("RGB")
        except Exception:
            return None
    return None

# =========================================================
# TOP NAVBAR
# =========================================================
st.markdown("""
<div class="top-nav">
    <div class="brand">🐄 Sapi<span>Deteksi</span></div>
</div>
""", unsafe_allow_html=True)

# =========================================================
# TABS NAVIGASI UTAMA
# =========================================================
tab_beranda, tab_deteksi, tab_info = st.tabs([
    "🏠   Beranda",
    "🔍   Deteksi Sapi",
    "📖   Informasi Jenis Sapi",
])

# =========================================================
# FUNGSI DETEKSI
# =========================================================
def prediksi(image, conf_threshold):
    results = model.predict(source=image, conf=conf_threshold)
    annotated_img = results[0].plot()
    annotated_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)

    boxes = results[0].boxes
    jumlah_objek = len(boxes)

    col_img, col_info = st.columns([1.1, 1], gap="large")
    with col_img:
        st.markdown('<div class="section-label">Hasil Deteksi</div>', unsafe_allow_html=True)
        st.image(annotated_rgb, use_container_width=True)
        if jumlah_objek == 0:
            st.warning("⚠️ Tidak ada sapi terdeteksi.")
        else:
            st.success(f"✅ Terdeteksi **{jumlah_objek}** objek sapi.")

    with col_info:
        if jumlah_objek > 0:
            st.markdown('<div class="section-label">Informasi Breed Terdeteksi</div>', unsafe_allow_html=True)

            kelas_terdeteksi = {}
            for box in boxes:
                cls_id = int(box.cls[0])
                cls_name = model.names[cls_id]
                conf = float(box.conf[0])
                if cls_name not in kelas_terdeteksi or conf > kelas_terdeteksi[cls_name]:
                    kelas_terdeteksi[cls_name] = conf

            for cls_name, conf in kelas_terdeteksi.items():
                info = INFO_SAPI.get(cls_name, {})
                badge_cls = "badge-high" if conf >= 0.7 else "badge-mid" if conf >= 0.5 else "badge-low"
                st.markdown(f"""
                <div class="breed-result-card">
                    <div class="breed-result-name">
                        {info.get('emoji','🐄')} {cls_name}
                        <span class="{badge_cls}">{conf:.0%}</span>
                    </div>
                    <div class="info-row"><div class="info-label">Tipe</div><div class="info-value">{info.get('tipe','—')}</div></div>
                    <div class="info-row"><div class="info-label">Asal</div><div class="info-value">{info.get('asal','—')}</div></div>
                    <div class="info-row"><div class="info-label">Ciri Fisik</div><div class="info-value">{info.get('ciri_fisik','—')}</div></div>
                    <div class="info-row"><div class="info-label">Keunggulan</div><div class="info-value">{info.get('keunggulan','—')}</div></div>
                    <div class="info-row"><div class="info-label">Pemanfaatan</div><div class="info-value">{info.get('pemanfaatan','—')}</div></div>
                </div>
                """, unsafe_allow_html=True)

# =========================================================
# TAB 1: BERANDA
# =========================================================
with tab_beranda:
    st.markdown("""
    <div class="hero-section">
        <div class="hero-eyebrow">Teknologi Deteksi Berbasis YOLOv8</div>
        <h1>Identifikasi Jenis Sapi<br>Secara Otomatis</h1>
        <p>Unggah foto atau gunakan kamera real-time untuk mendeteksi dan mengklasifikasikan
        jenis sapi secara instan — cepat, akurat, dan mudah digunakan.</p>
    </div>

    <div class="section-label">Jenis Sapi yang Dapat Dikenali</div>
    <div class="breed-chips">
        <div class="breed-chip">🐂 Brangus</div>
        <div class="breed-chip">🐄 Friesian Holstein</div>
        <div class="breed-chip">🐂 Limousin</div>
        <div class="breed-chip">🐄 Simmental</div>
    </div>

    <div class="section-label">Cara Menggunakan</div>
    <div class="steps-grid" style="margin-top:10px;">
        <div class="step-card">
            <div class="step-num">01</div>
            <div class="step-title">Buka Tab Deteksi</div>
            <div class="step-desc">Klik tab "Deteksi Sapi" di bagian atas halaman ini.</div>
        </div>
        <div class="step-card">
            <div class="step-num">02</div>
            <div class="step-title">Upload atau Kamera Real-Time</div>
            <div class="step-desc">Pilih sumber deteksi — unggah foto dari perangkat atau gunakan kamera secara real-time.</div>
        </div>
        <div class="step-card">
            <div class="step-num">03</div>
            <div class="step-title">Lihat Hasil & Info</div>
            <div class="step-desc">Model menampilkan breed beserta deskripsi karakteristiknya.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# =========================================================
# TAB 2: DETEKSI
# =========================================================
with tab_deteksi:
    col_set1, col_set2 = st.columns([1, 2])
    with col_set1:
        metode = st.selectbox(
            "Metode Input",
            ("📁  Upload Foto", "📷  Kamera Real-Time"),
            label_visibility="visible",
        )
    with col_set2:
        conf_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.10, max_value=1.00,
            value=0.50, step=0.05,
        )

    st.markdown("<hr style='margin:12px 0 20px;border:none;border-top:1px solid #EDE8DF;'>", unsafe_allow_html=True)

    if "Upload" in metode:
        file = st.file_uploader(
            "Pilih foto sapi (JPG / PNG)",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed",
        )
        if file:
            img = Image.open(file).convert("RGB")
            st.markdown('<div class="section-label" style="margin-bottom:6px;">Foto Asli</div>', unsafe_allow_html=True)
            st.image(img, width=340)
            st.markdown("<div style='margin-bottom:12px;'></div>", unsafe_allow_html=True)
            if st.button("🚀  Mulai Deteksi", type="primary", use_container_width=False):
                with st.spinner("Mendeteksi..."):
                    prediksi(img, conf_threshold)
        else:
            st.markdown("""
            <div class="upload-hint">
                📂  Seret foto ke sini atau klik <b>Browse files</b><br>
                <span style='font-size:12px;'>Format: JPG, JPEG, PNG</span>
            </div>
            """, unsafe_allow_html=True)

    else:  # Kamera Real-Time
        st.markdown(
            '<div class="section-label">Kamera Real-Time</div>',
            unsafe_allow_html=True
        )
        st.caption(
            "Arahkan kamera ke sapi. Sistem akan mendeteksi jenis sapi "
            "secara langsung tanpa perlu mengambil foto."
        )

        # PENTING: video_frame_callback dijalankan oleh streamlit-webrtc di
        # THREAD TERPISAH (bukan thread utama Streamlit). st.session_state
        # TIDAK BOLEH diakses dari thread lain -> akan raise exception yang
        # ditelan diam-diam, sehingga frame asli dikembalikan tanpa box.
        # Solusinya: simpan state di resource global (cache_resource).
        #
        # ARSITEKTUR: callback video TIDAK BOLEH menunggu YOLO selesai,
        # kalau tidak video jadi patah-patah (blocking). Jadi YOLO dijalankan
        # di THREAD BACKGROUND TERPISAH yang terus memproses frame terbaru
        # secepat model mampu, sementara callback video hanya menggambar
        # hasil deteksi TERAKHIR yang tersedia ke SETIAP frame (tidak
        # pernah skip) -> video tetap mulus & box tidak berkedip.
        @st.cache_resource
        def get_camera_state():
            state = {
                "lock": threading.Lock(),
                "latest_frame": None,      # frame mentah terbaru dari kamera
                "detections": [],           # hasil deteksi terakhir (list of dict)
                "conf": 0.5,                 # threshold terkini (diupdate dari slider)
                "worker_started": False,
                "error": None,
            }

            def _worker():
                while True:
                    frame = None
                    with state["lock"]:
                        if state["latest_frame"] is not None:
                            frame = state["latest_frame"]
                            state["latest_frame"] = None  # tandai sudah diambil
                            conf = state["conf"]
                    if frame is not None:
                        try:
                            results = model.predict(
                                source=frame,
                                conf=conf,
                                imgsz=480,   # inferensi di resolusi lebih kecil -> cepat,
                                             # tapi koordinat box otomatis dikembalikan
                                             # ke skala frame ASLI oleh Ultralytics
                                verbose=False,
                            )
                            result = results[0]
                            dets = []
                            for box in result.boxes:
                                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int).tolist()
                                cls_name = model.names[int(box.cls[0])]
                                conf_val = float(box.conf[0])
                                dets.append({
                                    "xyxy": (x1, y1, x2, y2),
                                    "cls": cls_name,
                                    "conf": conf_val,
                                })
                            with state["lock"]:
                                state["detections"] = dets
                                state["error"] = None
                        except Exception as e:
                            with state["lock"]:
                                state["error"] = str(e)
                    else:
                        time.sleep(0.01)  # tidak ada frame baru, hindari busy-loop

            t = threading.Thread(target=_worker, daemon=True)
            t.start()
            state["worker_started"] = True
            return state

        camera_state = get_camera_state()
        camera_state["conf"] = conf_threshold  # update threshold tiap rerun (main thread, aman)

        RTC_CONFIGURATION = RTCConfiguration({
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        })

        def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
            img = frame.to_ndarray(format="bgr24")

            # Kirim frame terbaru ke worker (non-blocking, tidak menunggu hasil).
            with camera_state["lock"]:
                camera_state["latest_frame"] = img.copy()
                dets = list(camera_state["detections"])
                err = camera_state["error"]

            annotated = img  # gambar langsung di atas frame asli (resolusi penuh, tajam)

            if err:
                cv2.putText(
                    annotated, f"Error: {err}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA
                )
            elif dets:
                for d in dets:
                    x1, y1, x2, y2 = d["xyxy"]
                    label = f"{d['cls']} {d['conf']:.0%}"
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (45, 106, 79), 2)
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    ty = max(y1, th + 8)
                    cv2.rectangle(annotated, (x1, ty - th - 8), (x1 + tw + 8, ty), (45, 106, 79), -1)
                    cv2.putText(
                        annotated, label, (x1 + 4, ty - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA
                    )
                cv2.rectangle(annotated, (10, 10), (330, 55), (45, 106, 79), -1)
                cv2.putText(
                    annotated, f"Sapi terdeteksi: {len(dets)}", (22, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA
                )
            else:
                cv2.rectangle(annotated, (10, 10), (380, 58), (80, 80, 80), -1)
                cv2.putText(
                    annotated, "Tidak ada sapi terdeteksi", (22, 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA
                )

            return av.VideoFrame.from_ndarray(annotated, format="bgr24")

        webrtc_ctx = webrtc_streamer(
            key="sapi-realtime-camera",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIGURATION,
            video_frame_callback=video_frame_callback,
            media_stream_constraints={
                # Batasi resolusi & frame rate capture di sisi browser.
                # Ini mengurangi beban encode/decode WebRTC secara signifikan
                # (penyebab utama video patah-patah), terpisah dari beban YOLO.
                "video": {
                    "width": {"ideal": 640, "max": 960},
                    "height": {"ideal": 480, "max": 720},
                    "frameRate": {"ideal": 15, "max": 20},
                },
                "audio": False
            },
            async_processing=True,
        )

        st.info(
            "💡 Klik **START** untuk menyalakan kamera. "
            "Jika tidak ada sapi, video akan menampilkan "
            "\"Tidak ada sapi terdeteksi\". Jika sapi terdeteksi, "
            "bounding box dan confidence akan muncul langsung."
        )

        # =====================================================
        # KARTU INFO BREED HASIL DETEKSI (mirror mode Upload Foto)
        # =====================================================
        st.markdown('<div class="section-label" style="margin-top:20px;">Informasi Breed Terdeteksi</div>', unsafe_allow_html=True)
        info_placeholder = st.empty()

        def render_detection_info():
            with camera_state["lock"]:
                dets_now = list(camera_state["detections"])
            with info_placeholder.container():
                if not dets_now:
                    st.caption("Belum ada sapi yang terdeteksi kamera.")
                    return
                # Ambil confidence tertinggi per kelas yang sedang terdeteksi.
                kelas_terdeteksi = {}
                for d in dets_now:
                    if d["cls"] not in kelas_terdeteksi or d["conf"] > kelas_terdeteksi[d["cls"]]:
                        kelas_terdeteksi[d["cls"]] = d["conf"]
                for cls_name, conf in kelas_terdeteksi.items():
                    info = INFO_SAPI.get(cls_name, {})
                    badge_cls = "badge-high" if conf >= 0.7 else "badge-mid" if conf >= 0.5 else "badge-low"
                    st.markdown(f"""
                    <div class="breed-result-card">
                        <div class="breed-result-name">
                            {info.get('emoji','🐄')} {cls_name}
                            <span class="{badge_cls}">{conf:.0%}</span>
                        </div>
                        <div class="info-row"><div class="info-label">Tipe</div><div class="info-value">{info.get('tipe','—')}</div></div>
                        <div class="info-row"><div class="info-label">Asal</div><div class="info-value">{info.get('asal','—')}</div></div>
                        <div class="info-row"><div class="info-label">Ciri Fisik</div><div class="info-value">{info.get('ciri_fisik','—')}</div></div>
                        <div class="info-row"><div class="info-label">Keunggulan</div><div class="info-value">{info.get('keunggulan','—')}</div></div>
                        <div class="info-row"><div class="info-label">Pemanfaatan</div><div class="info-value">{info.get('pemanfaatan','—')}</div></div>
                    </div>
                    """, unsafe_allow_html=True)

        render_detection_info()

        if webrtc_ctx.state.playing:
                from streamlit_autorefresh import st_autorefresh
                st_autorefresh(interval=1200, key="camera_info_autorefresh")
            
# =========================================================
# TAB 3: INFORMASI JENIS SAPI
# =========================================================
with tab_info:
    st.markdown("""
    <p style="color:#6B6B6B;font-size:14px;margin-bottom:24px;">
    Berikut adalah beberapa jenis sapi pada DE5MUTIYARAFARM .
    </p>
    """, unsafe_allow_html=True)

    breed_list = list(INFO_SAPI.items())
    for i in range(0, len(breed_list), 2):
        cols = st.columns(2, gap="large")
        for j, col in enumerate(cols):
            if i + j >= len(breed_list):
                break
            nama, info = breed_list[i + j]
            foto = load_breed_image(info["foto"])

            with col:
                # Header
                st.markdown(f"""
                <div class="breed-card-header" style="border-radius:12px 12px 0 0;">
                    <div class="bname">{info['emoji']} {nama}</div>
                    <div class="breed-type-badge">{info['tipe']}</div>
                </div>
                """, unsafe_allow_html=True)

                # Foto dari repo
                if foto:
                    st.image(foto, use_container_width=True)
                else:
                    st.markdown(f"""
                    <div class="img-placeholder">
                        <span style="font-size:38px;">{info['emoji']}</span>
                        <span>Tambahkan foto ke <code>{info['foto']}</code></span>
                    </div>
                    """, unsafe_allow_html=True)

                # Informasi
                st.markdown(f"""
                <div style="background:white;border:1px solid #EDE8DF;border-top:none;
                    border-radius:0 0 12px 12px;padding:18px;margin-bottom:4px;">
                    <div class="info-row"><div class="info-label">Asal</div><div class="info-value">{info['asal']}</div></div>
                    <div class="info-row"><div class="info-label">Ciri Fisik</div><div class="info-value">{info['ciri_fisik']}</div></div>
                    <div class="info-row"><div class="info-label">Keunggulan</div><div class="info-value">{info['keunggulan']}</div></div>
                    <div class="info-row"><div class="info-label">Pemanfaatan</div><div class="info-value">{info['pemanfaatan']}</div></div>
                </div>
                """, unsafe_allow_html=True)
