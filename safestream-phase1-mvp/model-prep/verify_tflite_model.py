"""
Run this AFTER prepare_model.py, on the .tflite file, before writing/trusting
the Kotlin parsing code. YOLOv8 export shape varies slightly by ultralytics
version, so don't assume -- check.

    pip install tflite-runtime   # or: pip install tensorflow
    python verify_tflite_model.py yolov8n-face.tflite
"""
import sys

def main(path):
    try:
        import tflite_runtime.interpreter as tflite
        Interpreter = tflite.Interpreter
    except ImportError:
        import tensorflow as tf
        Interpreter = tf.lite.Interpreter

    interpreter = Interpreter(model_path=path)
    interpreter.allocate_tensors()

    print("=== INPUT DETAILS ===")
    for d in interpreter.get_input_details():
        print(f"name={d['name']}  shape={d['shape']}  dtype={d['dtype']}")

    print("\n=== OUTPUT DETAILS ===")
    for d in interpreter.get_output_details():
        print(f"name={d['name']}  shape={d['shape']}  dtype={d['dtype']}")

    print("\nExpected for YOLOv8n at imgsz=320, 1 class (face):")
    print("  input:  [1, 320, 320, 3]")
    print("  output: [1, 5, 2100]  (x, y, w, h, confidence) x 2100 anchor points")
    print("If your output shape differs, update YoloDetector.kt's parsing")
    print("logic to match what's printed above -- this is the #1 cause of")
    print("'detections look like garbage' bugs.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python verify_tflite_model.py <path-to-model.tflite>")
        sys.exit(1)
    main(sys.argv[1])
