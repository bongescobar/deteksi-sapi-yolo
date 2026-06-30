import streamlit as st
import cv2
from PIL import Image
import numpy as np
from ultralytics import YOLO

# =========================================================
# KONFIGURASI HALAMAN & MODEL
# =========================================================
st.set_page_config(page_title="Pendeteksi Sapi", page_icon="🐄", layout="centered")

@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# =========================================================
# DATA INFORMASI JENIS SAPI
# =========================================================
INFO_SAPI = {
    "Brangus": {
        "asal": "Amerika Serikat (hasil persilangan Angus dan Brahman)",
        "ciri_fisik": "Tubuh berwarna hitam pekat, telinga sedang, gelambir (lipatan kulit leher) tidak terlalu besar, tubuh kompak dan berotot.",
        "keunggulan": "Tahan terhadap iklim panas, daya tahan tubuh kuat, kualitas daging baik, mudah beradaptasi di lingkungan tropis.",
        "pemanfaatan": "Sapi potong (penghasil daging).",
    },
    "Friesian Holstein": {
        "asal": "Belanda (wilayah Friesland dan Holstein)",
        "ciri_fisik": "Corak belang hitam-putih khas, tubuh besar, ambing susu berkembang baik, kaki relatif panjang.",
        "keunggulan": "Produksi susu sangat tinggi dibanding breed lain, banyak dipakai sebagai sapi perah utama di peternakan.",
        "pemanfaatan": "Sapi perah (penghasil susu).",
    },
    "Limousin": {
        "asal": "Prancis (wilayah Limousin)",
        "ciri_fisik": "Warna tubuh cokelat keemasan hingga cokelat tua, otot tubuh sangat besar terutama di bagian paha dan punggung, tanduk melengkung ke samping.",
        "keunggulan": "Pertumbuhan otot cepat, rendemen daging tinggi dengan lemak relatif rendah, banyak digunakan untuk persilangan sapi potong.",
        "pemanfaatan": "Sapi potong (penghasil daging premium).",
    },
    "Simmental": {
        "asal": "Swiss (lembah Sungai Simme)",
        "ciri_fisik": "Warna tubuh cokelat kemerahan dengan bagian wajah, perut, dan kaki berwarna putih, tubuh besar dan tinggi.",
        "keunggulan": "Pertumbuhan badan cepat, dapat dimanfaatkan ganda (daging dan susu), adaptasi baik di berbagai iklim.",
        "pemanfaatan": "Sapi dwiguna (daging dan susu).",
    },
}

# =========================================================
# NAVIGASI SIDEBAR
# =========================================================
st.sidebar.title("🐄 Menu Navigasi")
halaman = st.sidebar.radio(
    "Pilih Halaman:",
    ("Beranda", "Deteksi Sapi", "Informasi Jenis Sapi"),
)

# =========================================================
# FUNGSI DETEKSI
# =========================================================
def prediksi(image, conf_threshold):
    results = model.predict(source=image, conf=conf_threshold)
    annotated_img = results[0].plot()  # hasil dalam format BGR (OpenCV)
    annotated_img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)  # convert ke RGB untuk Streamlit

    st.image(annotated_img_rgb, caption="Hasil Deteksi", use_container_width=True)

    boxes = results[0].boxes
    jumlah_objek = len(boxes)

    if jumlah_objek == 0:
        st.warning("Tidak ada sapi yang terdeteksi pada gambar ini.")
        return

    st.success(f"Terdeteksi {jumlah_objek} objek sapi!")

    # Ambil daftar kelas unik yang terdeteksi beserta confidence tertinggi per kelas
    kelas_terdeteksi = {}
    for box in boxes:
        cls_id = int(box.cls[0])
        cls_name = model.names[cls_id]
        conf = float(box.conf[0])
        if cls_name not in kelas_terdeteksi or conf > kelas_terdeteksi[cls_name]:
            kelas_terdeteksi[cls_name] = conf

    st.markdown("---")
    st.subheader("📋 Informasi Jenis Sapi Terdeteksi")

    for cls_name, conf in kelas_terdeteksi.items():
        info = INFO_SAPI.get(cls_name)
        with st.expander(f"🐂 {cls_name} (confidence: {conf:.2f})", expanded=True):
            if info:
                st.markdown(f"**Asal:** {info['asal']}")
                st.markdown(f"**Ciri Fisik:** {info['ciri_fisik']}")
                st.markdown(f"**Keunggulan:** {info['keunggulan']}")
                st.markdown(f"**Pemanfaatan:** {info['pemanfaatan']}")
            else:
                st.write("Informasi detail untuk jenis ini belum tersedia.")


# =========================================================
# HALAMAN: BERANDA
# =========================================================
if halaman == "Beranda":
    st.title("🐄 Pendeteksi Jenis Sapi Web App")
    st.markdown(
        """
        Selamat datang di **Pendeteksi Jenis Sapi**!

        Aplikasi ini menggunakan model **YOLOv8** untuk mendeteksi dan mengklasifikasikan
        jenis sapi dari gambar atau foto kamera secara otomatis.

        ### Jenis sapi yang dapat dikenali:
        - 🐂 **Brangus**
        - 🐄 **Friesian Holstein**
        - 🐂 **Limousin**
        - 🐄 **Simmental**

        ### Cara menggunakan aplikasi:
        1. Buka halaman **Deteksi Sapi** melalui menu di sidebar.
        2. Pilih metode input: upload foto atau ambil foto lewat kamera.
        3. Klik tombol **Mulai Deteksi** untuk melihat hasilnya.
        4. Lihat juga halaman **Informasi Jenis Sapi** untuk mempelajari karakteristik
           masing-masing jenis sapi secara lebih lengkap.
        """
    )

# =========================================================
# HALAMAN: DETEKSI SAPI
# =========================================================
elif halaman == "Deteksi Sapi":
    st.title("🔍 Deteksi Jenis Sapi")

    metode_input = st.sidebar.selectbox("Metode Input:", ("Upload Foto", "Kamera (Snapshot)"))
    conf_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.5)

    if metode_input == "Upload Foto":
        file = st.file_uploader("Pilih foto sapi...", type=["jpg", "jpeg", "png"])
        if file is not None:
            img = Image.open(file).convert("RGB")
            st.image(img, caption="Foto Asli", use_container_width=True)
            if st.button("Mulai Deteksi"):
                prediksi(img, conf_threshold)

    elif metode_input == "Kamera (Snapshot)":
        kamera_file = st.camera_input("Ambil foto objek")
        if kamera_file is not None:
            img = Image.open(kamera_file).convert("RGB")
            prediksi(img, conf_threshold)

# =========================================================
# HALAMAN: INFORMASI JENIS SAPI
# =========================================================
elif halaman == "Informasi Jenis Sapi":
    st.title("📖 Informasi Jenis Sapi")
    st.markdown("Berikut adalah penjelasan karakteristik dari masing-masing jenis sapi yang dapat dideteksi oleh aplikasi ini.")

    for nama, info in INFO_SAPI.items():
        st.markdown("---")
        st.subheader(f"🐂 {nama}")
        st.markdown(f"**Asal:** {info['asal']}")
        st.markdown(f"**Ciri Fisik:** {info['ciri_fisik']}")
        st.markdown(f"**Keunggulan:** {info['keunggulan']}")
        st.markdown(f"**Pemanfaatan:** {info['pemanfaatan']}")