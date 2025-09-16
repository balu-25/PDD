import streamlit as st
from ultralytics import YOLO
from PIL import Image

# Load YOLOv8 model only once
@st.cache_resource
def load_model():
    model = YOLO("best.pt")  # ensure best.pt is in project root
    model.to("cpu")
    return model

model = load_model()

st.title("🌱 Plant Disease Detection (YOLOv8)")

uploaded_file = st.file_uploader("Upload a plant leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Run prediction
    results = model.predict(image, imgsz=320, conf=0.25, device="cpu")

    # Show annotated image
    annotated = results[0].plot()  # numpy array (BGR)
    st.image(annotated, caption="Predictions", use_column_width=True)

    # Show detected classes
    detections = []
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            conf = float(box.conf[0])
            detections.append(f"{label} ({conf:.2f})")

    if detections:
        st.subheader("Detections:")
        for d in detections:
            st.write("- " + d)
