from flask import Flask, request, jsonify
from model.face_recognizer import FaceRecognizer
from PIL import Image
import numpy as np
import io
import cv2
import os

app = Flask(__name__)
recognizer = FaceRecognizer()

def convert_to_cv2_image(file_bytes):
    """Convert uploaded image bytes to OpenCV format"""
    image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

@app.route("/recognize", methods=["POST"])
def recognize():
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    try:
        # Convert image to cv2 format (as numpy array)
        file_bytes = request.files["image"].read()
        image_array = convert_to_cv2_image(file_bytes)

        # Save temp image (if your recognizer still needs path)
        temp_path = "temp_input.jpg"
        cv2.imwrite(temp_path, image_array)

        # Recognize using saved image
        name, confidence = recognizer.recognize_face(temp_path)

        # Clean up
        if os.path.exists(temp_path):
            os.remove(temp_path)

        return jsonify({
            "name": name,
            "confidence": confidence
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/", methods=["GET"])
def root():
    return "DeepFace API is running!"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
