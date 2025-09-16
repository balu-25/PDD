from flask import Flask, request, jsonify, render_template
from ultralytics import YOLO
import os
from PIL import Image
import cv2

app = Flask(__name__)

# Load YOLOv8n model
model = YOLO("best.pt")  # keep your trained model here

# Folders
UPLOAD_FOLDER = "uploads"
RESULT_FOLDER = "static/results"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "image" not in request.files:
            return jsonify({"error": "No image uploaded"}), 400

        image = request.files["image"]
        image_path = os.path.join(UPLOAD_FOLDER, image.filename)
        image.save(image_path)

        # Run YOLOv8 prediction (optimized for Render Free Tier)
        results = model.predict(
            source=image_path,
            conf=0.25,
            iou=0.45,
            imgsz=320,   # reduce input size
            device="cpu",# run on CPU
            batch=1
        )

        # Save annotated result
        annotated_image = results[0].plot()
        annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
        output_file_path = os.path.join(RESULT_FOLDER, "predicted_" + image.filename)
        Image.fromarray(annotated_image_rgb).save(output_file_path)

        # Get best detection
        top_detection = None
        top_conf = 0.0
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                label = model.names[cls_id]
                conf = float(box.conf[0])
                if conf > top_conf:
                    top_detection = {"label": label, "confidence": round(conf, 2)}
                    top_conf = conf

        return jsonify({
            "prediction": top_detection,
            "image_url": "/" + output_file_path.replace("\\", "/")
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
