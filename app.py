import streamlit as st
import cv2
from PIL import Image
import numpy as np
from ultralytics import YOLO

# Memuat model yang sudah ditraining
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

st.set_page_config(page_title="Pendeteksi Sapi", layout="centered")
st.title("🐄 Pendeteksi Jenis Sapi Web App")

option = st.sidebar.selectbox("Metode Input:", ("Upload Foto", "Kamera (Snapshot)"))
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.4)

def prediksi(image):
    img_array = np.array(image)
    results = model.predict(source=Image, conf=0.65)
    annotated_img = results[0].plot()
    st.image(annotated_img, caption="Hasil Deteksi", use_container_width=True)
    
    # Menghitung jumlah sapi yang lewat/terdeteksi
    jumlah_objek = len(results[0].boxes)
    st.success(f"Terdeteksi {jumlah_objek} Sapi!")

if option == "Upload Foto":
    file = st.file_uploader("Pilih foto sapi...", type=["jpg", "jpeg", "png"])
    if file is not None:
        img = Image.open(file)
        st.image(img, caption="Foto Asli", use_container_width=True)
        if st.button("Mulai Deteksi"):
            prediksi(img)

elif option == "Kamera (Snapshot)":
    kamera_file = st.camera_input("Ambil foto objek")
    if kamera_file is not None:
        img = Image.open(kamera_file)
        prediksi(img)