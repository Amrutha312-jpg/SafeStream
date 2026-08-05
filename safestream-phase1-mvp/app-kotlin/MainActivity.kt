package com.safestream.app

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Matrix
import android.graphics.RectF
import android.os.Bundle
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import org.opencv.android.OpenCVLoader
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

/**
 * Phase 1 MVP: Camera preview -> YOLOv8-Nano face detection -> OpenCV blur
 * -> on-screen display. No streaming/encoding yet (that's Milestone 3).
 *
 * Layout expected (activity_main.xml): a PreviewView (id: previewView) with
 * a BlurOverlayView (id: blurOverlay) stacked exactly on top of it, e.g. in
 * a FrameLayout.
 */
class MainActivity : AppCompatActivity() {

    private lateinit var previewView: androidx.camera.view.PreviewView
    private lateinit var blurOverlay: BlurOverlayView
    private lateinit var detector: YoloDetector
    private lateinit var cameraExecutor: ExecutorService

    private val requestPermissionLauncher =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
            if (granted) startCamera() else {
                // TODO: show a message -- camera permission is required for this app
            }
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        if (!OpenCVLoader.initDebug()) {
            // Older OpenCV SDKs need initDebug(); newer 4.9+ SDKs auto-init
            // via System.loadLibrary("opencv_java4") instead -- check which
            // your imported SDK module uses if this branch triggers.
        }

        previewView = findViewById(R.id.previewView)
        blurOverlay = findViewById(R.id.blurOverlay)
        detector = YoloDetector(this)
        cameraExecutor = Executors.newSingleThreadExecutor()

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
            == PackageManager.PERMISSION_GRANTED
        ) {
            startCamera()
        } else {
            requestPermissionLauncher.launch(Manifest.permission.CAMERA)
        }
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)

        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()

            val preview = Preview.Builder().build().also {
                it.setSurfaceProvider(previewView.surfaceProvider)
            }

            // Stream A (Inference): downscaled analysis frames, per report's
            // dual-stream architecture (Ch.4). Full-res "Stream B" broadcast
            // path is added in the streaming milestone.
            val imageAnalyzer = ImageAnalysis.Builder()
                .setTargetResolution(android.util.Size(320, 320))
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()
                .also {
                    it.setAnalyzer(cameraExecutor, FrameAnalyzer())
                }

            val cameraSelector = CameraSelector.DEFAULT_FRONT_CAMERA

            try {
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(
                    this, cameraSelector, preview, imageAnalyzer
                )
            } catch (e: Exception) {
                e.printStackTrace()
            }
        }, ContextCompat.getMainExecutor(this))
    }

    /**
     * Frame-skipping (per report Ch.4 "Optimization"): run YOLO every other
     * frame and reuse the previous frame's boxes in between, to save battery.
     * frameCount starts at 0 so the very first frame always runs inference.
     */
    private var frameCount = 0
    private var lastBoxes: List<RectF> = emptyList()

    inner class FrameAnalyzer : ImageAnalysis.Analyzer {
        override fun analyze(imageProxy: ImageProxy) {
            try {
                val bitmap = imageProxy.toBitmap() ?: return
                val resized = Bitmap.createScaledBitmap(
                    bitmap, YoloDetector.INPUT_SIZE, YoloDetector.INPUT_SIZE, true
                )

                val boxes: List<RectF>
                if (frameCount % 2 == 0) {
                    boxes = detector.detect(resized).map { it.box }
                    lastBoxes = boxes
                } else {
                    boxes = lastBoxes
                }
                frameCount++

                runOnUiThread {
                    blurOverlay.updateFrame(bitmap, boxes)
                }
            } finally {
                imageProxy.close()
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        detector.close()
        cameraExecutor.shutdown()
    }
}

/**
 * ImageProxy (YUV_420_888) -> Bitmap conversion. CameraX's ImageProxy does
 * not give you a Bitmap directly; this is the standard conversion path.
 * NOTE: this is a CPU-cost step -- if profiling shows it's the bottleneck,
 * switch to CameraX's built-in ImageAnalysis.Builder().setOutputImageFormat
 * (RGBA_8888) available in camera-core 1.3+, which skips this conversion.
 */
fun ImageProxy.toBitmap(): Bitmap? {
    val yBuffer = planes[0].buffer
    val uBuffer = planes[1].buffer
    val vBuffer = planes[2].buffer

    val ySize = yBuffer.remaining()
    val uSize = uBuffer.remaining()
    val vSize = vBuffer.remaining()

    val nv21 = ByteArray(ySize + uSize + vSize)
    yBuffer.get(nv21, 0, ySize)
    vBuffer.get(nv21, ySize, vSize)
    uBuffer.get(nv21, ySize + vSize, uSize)

    val yuvImage = android.graphics.YuvImage(
        nv21, android.graphics.ImageFormat.NV21, width, height, null
    )
    val out = java.io.ByteArrayOutputStream()
    yuvImage.compressToJpeg(android.graphics.Rect(0, 0, width, height), 90, out)
    val bytes = out.toByteArray()
    val bmp = android.graphics.BitmapFactory.decodeByteArray(bytes, 0, bytes.size)

    // Front camera frames are typically rotated 90/270 -- rotate to upright.
    val rotation = imageInfo.rotationDegrees
    if (rotation == 0) return bmp
    val matrix = Matrix().apply { postRotate(rotation.toFloat()) }
    return Bitmap.createBitmap(bmp, 0, 0, bmp.width, bmp.height, matrix, true)
}
