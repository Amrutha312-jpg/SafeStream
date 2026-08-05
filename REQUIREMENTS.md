# SafeStream - System Requirements & Environment Specifications

This document outlines the hardware, software, dependencies, and environment configuration required to build and run the **SafeStream** project.

---

## 📱 Hardware Requirements

### Target Mobile Device
* **Processor (CPU)**: Octa-core ARM64-v8a architecture (Qualcomm Snapdragon, MediaTek Dimensity, Samsung Exynos, or Google Tensor).
* **Graphics (GPU)**: Adreno / Mali / Xclipse GPU with **OpenCL 2.0+** support (required for TFLite GPU Delegate).
* **RAM**: Minimum 4 GB (6 GB or higher recommended for continuous 30 FPS processing).
* **Camera**: Front or rear camera with $1080p$ @ $30	ext{ FPS}$ capture capability.

---
git add README.md REQUIREMENTS.md
## 💻 Development Workstation Requirements

* **Operating System**: Windows 10/11 (64-bit), macOS 12+, or Ubuntu 22.04 LTS.
* **RAM**: Minimum 16 GB (32 GB recommended for building OpenCV native C++ bindings).
* **Disk Space**: At least 15 GB free disk space (for Android SDK, NDK, and Gradle cache).

---

## 🔧 Software & Toolchain Specifications

| Tool / Component | Version / Specification |
| :--- | :--- |
| **Android Studio** | Ladybug / Jellyfish (2024.1.1+) |
| **JDK (Java Development Kit)** | OpenJDK 17 or Java 21 |
| **Android SDK Build Tools** | 34.0.0 or 35.0.0 |
| **Min SDK Level** | API 24 (Android 7.0 Nougat) |
| **Target SDK Level** | API 34 / 35 (Android 14 / 15) |
| **Android NDK** | Version 25.2.9519653 or newer (for OpenCV JNI) |
| **CMake** | 3.22.1+ |
| **Gradle** | 8.4+ / AGP 8.2+ |

---

## 📦 Core Libraries & Dependencies

### 1. CameraX Framework
```kotlin
implementation("androidx.camera:camera-core:1.3.1")
implementation("androidx.camera:camera-camera2:1.3.1")
implementation("androidx.camera:camera-lifecycle:1.3.1")
implementation("androidx.camera:camera-view:1.3.1")
```

### 2. LiteRT (TensorFlow Lite) & Acceleration
```kotlin
implementation("org.tensorflow:tensorflow-lite:2.14.0")
implementation("org.tensorflow:tensorflow-lite-gpu:2.14.0")
implementation("org.tensorflow:tensorflow-lite-gpu-api:2.14.0")
implementation("org.tensorflow:tensorflow-lite-support:0.4.4")
```

### 3. OpenCV Native C++ SDK
* **Version**: OpenCV 4.x for Android (`sdk/` folder included in repository).
* **Modules**: `core`, `imgproc`, `dnn`, `video`.

---

## 🤖 Neural Network Model Specs

* **Architecture**: YOLOv8-Nano Face Detection
* **Input Resolution**: $320 \times 320 \times 3$ (RGB)
* **Precision Format**: Float32 / Quantized INT8
* **Output Tensor Format**: `[1, 20, 2100]` / `[1, 2100, 20]` (Bounding boxes + Confidence scores)