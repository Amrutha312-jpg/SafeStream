# SafeStream 🛡️📹

**Real-Time Edge AI Middleware for Privacy-Preserving Live Video Anonymization**

SafeStream is an Android application designed for high-performance, low-latency, privacy-preserving live video streaming. By processing video feeds locally on edge hardware, SafeStream automatically detects human faces in real time. It utilizes selective anonymization to seamlessly whitelist the primary streamer while dynamically blurring all unauthorized bystanders before streaming output, ensuring end-to-end privacy without relying on cloud computation.

---

## 🔑 Key Features

* **Selective Streamer Whitelisting**: Registers the primary streamer using a continuous 5-second facial capture burst, storing multiple angles for robust recognition.
* **Real-Time Face Anonymization**: Ultra-fast face detection using custom optimized YOLOv8-Nano models.
* **Facial Embedding & Matching**: Utilizes the MobileFaceNet architecture to extract facial embeddings and computes Cosine Similarity (threshold: 0.48) to instantly distinguish the streamer from strangers.
* **Temporal Debouncing & Smoothing**: Employs a frame-buffer memory system to prevent visual flickering and maintain stable unblurring even during brief lighting drops or extreme angles.
* **On-Device Edge AI Processing**: Full execution on local hardware via **LiteRT (TensorFlow Lite)** with **GPU Acceleration** using `GpuDelegate` and FP16 precision.
* **Custom Dynamic Blur Overlay**: Responsive box transformation and hardware-rendered blurring synchronized directly with camera frames.
* **Persistent Local Storage**: Embeddings are safely serialized and stored on-device using Gson and SharedPreferences, ensuring the streamer remains recognized across app restarts.

---

## 🛠️ Tech Stack & Architecture

| Component | Technology |
| :--- | :--- |
| **Language** | Kotlin, C++, Python |
| **UI Framework** | Android Jetpack, XML Layouts |
| **Camera Pipeline** | CameraX (`ImageAnalysis`, `Preview`) |
| **AI / Machine Learning** | YOLOv8-Nano (Face Detection), MobileFaceNet (Recognition) via TFLite |
| **Hardware Acceleration** | OpenCL / TFLite GPU Delegate (FP16 precision) |
| **Local Storage** | Android SharedPreferences, Google Gson |
| **Version Control** | Git / GitHub |

---

## 📁 Repository Structure

```text
SafeStream/
├── app/                  
│   ├── src/
│   │   ├── main/
│   │   │   ├── java/com/safestream/app/
│   │   │   │   ├── MainActivity.kt         # CameraX pipeline & frame analyzer
│   │   │   │   ├── YoloDetector.kt         # YOLOv8 TFLite inference engine
│   │   │   │   ├── FaceEmbedder.kt         # MobileFaceNet embedding generator
│   │   │   │   ├── IdentityMatcher.kt      # Cosine similarity classification logic
│   │   │   │   ├── EnrollmentHelper.kt     # 5-second enrollment burst manager
│   │   │   │   ├── EnrollmentStore.kt      # Gson serialization for local storage
│   │   │   │   ├── FaceCropUtils.kt        # Bitmap cropping and normalization
│   │   │   │   └── BlurOverlayView.kt      # Real-time blur rendering view
│   │   │   ├── assets/                     
│   │   │   │   ├── yolov8n-face.tflite     # Face detection model
│   │   │   │   └── mobilefacenet.tflite    # Face recognition model
│   │   │   └── res/                        # Layouts, graphics, and string resources
│   │   └── build.gradle.kts                # App-level dependencies
├── scripts/              
│   ├── export.py         # PyTorch to TFLite export script
│   └── yolov8n.pt        # PyTorch model weights
├── sdk/                  # OpenCV Native SDK dependencies
├── build.gradle.kts      # Top-level project build configuration
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
   git clone [https://github.com/project14team/SafeStream.git](https://github.com/project14team/SafeStream.git)
   cd SafeStream
2. Open the project in Android Studio.
3. Allow Gradle to synchronize dependencies and build native libraries (`sdk/`).
4. Connect a physical Android device via USB with **Developer Options** and **USB Debugging** enabled.
5. Click **Run** (`Shift + F10`) to build and deploy the application.

---

## ⚙️ How It Works (Pipeline)

### Phase 1: Streamer Enrollment
1. **Trigger Burst**: User taps "Register Face" to initiate a 5-second capture window.
2. **Detection & Cropping**: YOLOv8 detects the face; `FaceCropUtils` extracts and normalizes the face to 112x112 pixels.
3. **Embedding Generation**: MobileFaceNet converts the face crop into a multi-dimensional mathematical vector (`FloatArray`).
4. **Persistent Storage**: Up to 30 diverse facial embeddings are serialized via Gson and saved to `SharedPreferences` for future sessions.

### Phase 2: Live Anonymization
1. **Camera Frame Capture**: CameraX grabs real-time frames in YUV_420_888 / RGBA format.
2. **Edge AI Inference**: `YoloDetector` extracts bounding boxes for all faces currently in the frame.
3. **Identity Classification**: Each detected face is cropped and embedded. `IdentityMatcher` compares these live embeddings against the stored streamer profile using Cosine Similarity.
4. **Filtering & Debouncing**: Faces scoring below the threshold (`0.48`) are classified as bystanders. A temporal debouncing buffer prevents flickering.
5. **Blur Overlay Application**: Custom `BlurOverlayView` receives the normalized bounding boxes of **only the bystanders** and renders GPU-accelerated blur regions over the video preview.

---

## 📜 License
This project is developed for academic and non-commercial research purposes under the Mangalore Institute of Technology & Engineering (MITE).