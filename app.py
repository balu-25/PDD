import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import os

# Load YOLOv8 model only once
@st.cache_resource
def load_model():
    model = YOLO("best.pt")  # ensure best.pt is in project root
    model.to("cpu")
    return model

model = load_model()
st.set_page_config(
    page_title="Plant Disease Detection",  # <-- This sets the browser tab title
    page_icon="🌿",                        # <-- Optional: a small icon in the tab
    layout="centered"                      # <-- Optional: page layout
)
st.markdown("<style>body, .stApp {background-color: #d4edda;}</style>", unsafe_allow_html=True)
st.title("🌿  Plant Disease Detection  🌿")

uploaded_file = st.file_uploader("Upload a plant leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Run prediction
    results = model.predict(image, conf=0.1, iou=0.7, imgsz=320, device="cpu")

    # Show annotated image (convert BGR → RGB for Streamlit)
    annotated = results[0].plot()
    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    st.image(annotated_rgb, caption="Predictions", use_container_width=True)

    # Find top detection (highest confidence)
    top_detection = None
    top_conf = 0.0
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            conf = float(box.conf[0])
            if conf > top_conf:
                top_detection = (label, conf)
                top_conf = conf

    # Display top detection result
    if top_detection:
        label, conf = top_detection
        st.success(f"✅ Highest confidence disease: **{label}** (confidence: {conf:.2f})")
    else:
        st.error("❌ No disease detected in the image.")
