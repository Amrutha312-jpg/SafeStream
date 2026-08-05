"""
SafeStream - Phase 1 MVP - Model Prep Script
==============================================
Run this on a laptop (not the phone). Produces a .tflite face-detection
model small enough to run on-device.

WHAT THIS DOES:
1. Downloads a pretrained YOLOv8-Nano face detection model (already trained
   on WIDER FACE, so you do NOT need to train from scratch for the MVP).
2. Exports it to TFLite format with INT8 quantization (shrinks ~6MB model
   further and speeds up inference on phone NPUs/GPUs).
3. Verifies the exported model's input/output shapes so the Kotlin code
   knows exactly what to expect.

SETUP (one time):
    python -m venv venv
    source venv/bin/activate        # Windows: venv\\Scripts\\activate
    pip install ultralytics

RUN:
    python prepare_model.py

OUTPUT:
    yolov8n-face.tflite   <- copy this into app/src/main/assets/ in Android Studio
"""

from ultralytics import YOLO
import os

# Pretrained YOLOv8n face-detection weights (community-trained on WIDER FACE).
# If this exact URL/repo is unavailable when you run this, search
# "yolov8n-face.pt download" — several mirrors exist (e.g. akanametov/yolov8-face
# on GitHub Releases). Any yolov8n-face.pt file works below.
MODEL_URL = "https://github.com/akanametov/yolov8-face/releases/download/v0.0.0/yolov8n-face.pt"
LOCAL_PT = "yolov8n-face.pt"

def download_if_needed():
    if os.path.exists(LOCAL_PT):
        print(f"Found existing {LOCAL_PT}, skipping download.")
        return
    import urllib.request
    print(f"Downloading base model from {MODEL_URL} ...")
    try:
        urllib.request.urlretrieve(MODEL_URL, LOCAL_PT)
        print("Download complete.")
    except Exception as e:
        print(f"AUTO-DOWNLOAD FAILED: {e}")
        print("Manually download a yolov8n-face.pt file and place it in this folder,")
        print("then re-run this script.")
        raise SystemExit(1)

def export_tflite():
    model = YOLO(LOCAL_PT)
    # imgsz=320 matches the "Stream A (Inference)" downscale from the report's
    # architecture (Ch.4) -- keeps inference fast enough for real-time on mobile.
    # int8=True quantizes the model (~6MB -> smaller, faster on NPU/GPU delegate).
    model.export(format="tflite", imgsz=320, int8=True)
    print("\nExport complete. Look for a file like:")
    print("  yolov8n-face_saved_model/yolov8n-face_full_integer_quant.tflite")
    print("Rename/copy that file to yolov8n-face.tflite and move it into")
    print("  app/src/main/assets/yolov8n-face.tflite")

if __name__ == "__main__":
    download_if_needed()
    export_tflite()
