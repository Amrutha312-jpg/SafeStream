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
import androidx.camera.core.resolutionselector.AspectRatioStrategy
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import org.opencv.android.OpenCVLoader
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

/**
 * Phase 1 MVP: Camera preview -> YOLOv8-Nano face detection -> OpenCV blur
 * -> on-screen display with proportional aspect ratio preservation.
 */
class MainActivity : AppCompatActivity() {

    private lateinit var previewView: androidx.camera.view.PreviewView
    private lateinit var blurOverlay: BlurOverlayView
    private lateinit var detector: YoloDetector
    private lateinit var cameraExecutor: ExecutorService
    private var lensFacing = CameraSelector.LENS_FACING_BACK

    private val requestPermissionLauncher =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
            if (granted) startCamera() else {
                // Handle permission denied
            }
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        OpenCVLoader.initDebug()

        previewView = findViewById(R.id.previewView)
        blurOverlay = findViewById(R.id.blurOverlay)
        detector = YoloDetector(this, "yolov8n-face.tflite")
        cameraExecutor = Executors.newSingleThreadExecutor()

        findViewById<android.widget.ImageButton>(R.id.switchCameraButton).setOnClickListener {
            toggleCamera()
        }

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
            == PackageManager.PERMISSION_GRANTED
        ) {
            startCamera()
        } else {
            requestPermissionLauncher.launch(Manifest.permission.CAMERA)
        }
    }

    private fun toggleCamera() {
        lensFacing = if (lensFacing == CameraSelector.LENS_FACING_BACK) {
            CameraSelector.LENS_FACING_FRONT
        } else {
            CameraSelector.LENS_FACING_BACK
        }
        startCamera()
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)

        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()

            // Define a resolution selector to prefer 16:9 across all use cases.
            // This prevents stretching at the camera hardware level.
            val resolutionSelector = ResolutionSelector.Builder()
                .setAspectRatioStrategy(AspectRatioStrategy.RATIO_16_9_FALLBACK_AUTO_STRATEGY)
                .build()

            val preview = Preview.Builder()
                .setResolutionSelector(resolutionSelector)
                .build().also {
                    it.setSurfaceProvider(previewView.surfaceProvider)
                }

            val imageAnalyzer = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setResolutionSelector(resolutionSelector)
                .build()
                .also {
                    it.setAnalyzer(cameraExecutor, FrameAnalyzer())
                }

            val cameraSelector = CameraSelector.Builder()
                .requireLensFacing(lensFacing)
                .build()

            try {
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(
                    this, cameraSelector, preview, imageAnalyzer
                )
            } catch (e: Exception) {
                android.util.Log.e("MainActivity", "Binding failed", e)
            }
        }, ContextCompat.getMainExecutor(this))
    }

    private var frameCount = 0
    private var lastBoxes: List<RectF> = emptyList()

    /**
     * Helper to crop a bitmap to a 1:1 square from the center.
     * This removes the rectangular "wings" and avoids stretching during resize.
     */
    private fun cropToSquare(bitmap: Bitmap): Bitmap {
        val size = minOf(bitmap.width, bitmap.height)
        val x = (bitmap.width - size) / 2
        val y = (bitmap.height - size) / 2
        return Bitmap.createBitmap(bitmap, x, y, size, size)
    }

    inner class FrameAnalyzer : ImageAnalysis.Analyzer {
        override fun analyze(imageProxy: ImageProxy) {
            try {
                android.util.Log.d("TEST", "Analyzer is running!")

                val isFrontCamera = lensFacing == CameraSelector.LENS_FACING_FRONT
                
                // 1. Convert ImageProxy to an upright (and mirrored if front) Bitmap
                val fullBitmap = imageProxy.toUprightBitmap(isFrontCamera) ?: run {
                    android.util.Log.e("TEST", "Failed to convert ImageProxy to Bitmap")
                    imageProxy.close()
                    return
                }

                val width = fullBitmap.width.toFloat()
                val height = fullBitmap.height.toFloat()
                
                val size = minOf(width, height)
                val leftOffset = (width - size) / 2f
                val topOffset = (height - size) / 2f

                val boxes: List<RectF>
                if (frameCount % 2 == 0) {
                    val squareBitmap = cropToSquare(fullBitmap)
                    val detections = detector.detect(squareBitmap)

                    for (det in detections) {
                        android.util.Log.d("TEST", "Raw model box: ${det.box}")
                    }
                    android.util.Log.d("TEST", "Detections count: ${detections.size}")
                    for (det in detections) {
                        android.util.Log.d("TEST", "Conf: ${det.confidence}, Box: ${det.box}")
                    }
                    boxes = detections.map { det ->
                        RectF(
                            (det.box.left * size + leftOffset) / width,
                            (det.box.top * size + topOffset) / height,
                            (det.box.right * size + leftOffset) / width,
                            (det.box.bottom * size + topOffset) / height
                        )
                    }
                    lastBoxes = boxes
                    if (squareBitmap !== fullBitmap) squareBitmap.recycle()
                } else {
                    boxes = lastBoxes
                }
                frameCount++

                // 5. Apply blur on the background thread
                val processedBitmap = blurOverlay.applyBlurToBoxes(fullBitmap, boxes)
                
                fullBitmap.recycle()
                    runOnUiThread {
                    // 7. Push the finished frame to the display
                    blurOverlay.setFrame(processedBitmap)
                }
            } catch (e: Exception) {
                android.util.Log.e("TEST", "Error in analyze: ${e.message}", e)
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
 * Robust ImageProxy -> Bitmap conversion.
 * Renamed to avoid recursion. We MUST apply rotationDegrees manually because
 * CameraX's toBitmap() returns the raw buffer orientation.
 */
fun ImageProxy.toUprightBitmap(mirror: Boolean = false): Bitmap? {
    val bitmap = try {
        this.toBitmap() 
    } catch (e: Exception) {
        android.util.Log.e("MainActivity", "toBitmap failed", e)
        null
    } ?: return null

    val rotation = imageInfo.rotationDegrees.toFloat()
    val matrix = Matrix()
    
    // 1. Rotate to upright orientation
    if (rotation != 0f) matrix.postRotate(rotation)
    
    // 2. Mirror horizontally for front camera (applied after rotation)
    if (mirror) matrix.postScale(-1f, 1f)

    return if (rotation != 0f || mirror) {
        val result = Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
        bitmap.recycle()
        result
    } else {
        bitmap
    }
}
