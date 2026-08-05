# SafeStream 🛡️📹

**Real-Time Edge AI Middleware for Privacy-Preserving Live Video Anonymization**

SafeStream is an Android application designed for high-performance, low-latency, privacy-preserving live video streaming. By processing video feeds locally on edge hardware (Mobile NPUs/GPUs), SafeStream automatically detects sensitive regions (such as human faces) in real time and applies dynamic blurring before streaming output, ensuring end-to-end privacy without relying on cloud computation.

---

## 🔑 Key Features

* **Real-Time Face Anonymization**: Ultra-fast face detection using custom optimized YOLOv8-Nano models.
* **On-Device Edge AI Processing**: Full execution on local hardware via **LiteRT (TensorFlow Lite)** with **GPU Acceleration** using `GpuDelegate` and FP16 precision.
* **Custom Dynamic Blur Overlay**: Responsive box transformation and hardware-rendered blurring synchronized directly with camera frames.
* **Flexible Tensor Formatting**: Native support for Float32 and Quantized (INT8/UINT8) YOLO models across NCHW/NHWC layouts.
* **Thermal & FPS Optimization**: Frame-rate decoupling, non-maximum suppression (NMS), linear interpolation (lerp), and memory reuse algorithms for sustained 30+ FPS live streams without device overheating.

---

## 🛠️ Tech Stack & Architecture

| Component | Technology |
| :--- | :--- |
| **Language** | Kotlin, C++, Python |
| **UI Framework** | Android Jetpack, XML Layouts |
| **Camera Pipeline** | CameraX (`ImageAnalysis`, `Preview`) |
| **AI / Machine Learning** | YOLOv8-Nano (TFLite / LiteRT), OpenCV Native C++ SDK |
| **Hardware Acceleration** | OpenCL / TFLite GPU Delegate (FP16 precision) |
| **Model Conversion** | Python, Ultralytics YOLOv8 API |
| **Version Control** | Git / GitHub |

---

## 📁 Repository Structure

```text
SafeStream/
├── app/                  # Android application source code
│   ├── src/
│   │   ├── main/
│   │   │   ├── java/com/safestream/app/
│   │   │   │   ├── MainActivity.kt         # CameraX pipeline & frame analyzer
│   │   │   │   ├── YoloDetector.kt         # TFLite inference engine & tensor parsing
│   │   │   │   └── BlurOverlayView.kt      # Real-time blur rendering view
│   │   │   ├── assets/                     # YOLOv8 TFLite model assets (.tflite)
│   │   │   └── res/                        # Layouts, graphics, and string resources
│   │   └── build.gradle.kts                # App-level dependencies and build configuration
├── scripts/              # Model training, export & conversion pipeline
│   ├── export.py         # YOLOv8 to TFLite export script
│   ├── yolov8n.pt        # PyTorch model weights
│   └── yolov8n-face.tflite # Exported mobile TFLite model
├── sdk/                  # OpenCV Native SDK dependencies
├── build.gradle.kts      # Top-level project build configuration
├── settings.gradle.kts   # Gradle settings & module references
├── REQUIREMENTS.md       # Setup, hardware, and environment dependencies
└── README.md             # Project overview documentation
```

---

## 🚀 Getting Started

### 1. Prerequisites
* **Android Studio**: Ladybug / Jellyfish (2024.1+) or newer.
* **Android SDK**: API 24 (Android 7.0) minimum, API 34/35 target.
* **Physical Device**: Android device with ARM64 architecture and OpenCL/GPU support (Emulator will lack GPU delegate acceleration).
* **JDK**: OpenJDK 17 or Java 21.
* **Python Environment**: Python 3.10+ (for model export).

### 2. Installation & Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/project14team/SafeStream.git
   cd SafeStream
   ```
2. Open the project in **Android Studio**.
3. Allow Gradle to synchronize dependencies and build native libraries (`sdk/`).
4. Connect a physical Android device via USB with **Developer Options** and **USB Debugging** enabled.
5. Click **Run** (`Shift + F10`) to build and deploy the application.

---

## ⚙️ How It Works (Pipeline)

1. **Model Export**: Python script (`scripts/export.py`) converts PyTorch YOLOv8 weights into an optimized TFLite model targeting 320x320 input.
2. **Camera Frame Capture**: CameraX grabs frames from the camera sensor in YUV_420_888 / RGBA format.
3. **Pre-Processing**: The image is cropped and resized to model dimensions (320x320) and loaded into native `ByteBuffer`.
4. **Edge AI Inference**: `YoloDetector` runs the frame through TFLite interpreter accelerated by GPU delegate to extract face detection candidate bounding boxes.
5. **Post-Processing & Scaling**: Confidence filtering (>0.10) and Non-Maximum Suppression (NMS, IoU > 0.45) remove duplicate boxes, which are then re-mapped back to normalized display coordinates.
6. **Blur Overlay Application**: Custom `BlurOverlayView` receives normalized bounding boxes and renders GPU-accelerated blur regions over the video preview.

---

## 📜 License
This project is developed for academic and non-commercial research purposes under the Mangalore Institute of Technology & Engineering (MITE).