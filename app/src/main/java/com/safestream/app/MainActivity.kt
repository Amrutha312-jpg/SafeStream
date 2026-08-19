package com.safestream.app

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Matrix
import android.graphics.RectF
import android.os.Bundle
import android.util.Log
import android.widget.Button
import android.widget.ImageButton
import android.widget.Toast
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

class MainActivity : AppCompatActivity() {

    private lateinit var previewView: androidx.camera.view.PreviewView
    private lateinit var blurOverlay: BlurOverlayView
    private lateinit var detector: YoloDetector
    private lateinit var cameraExecutor: ExecutorService
    private var lensFacing = CameraSelector.LENS_FACING_BACK

    private lateinit var faceEmbedder: FaceEmbedder
    private lateinit var enrollmentStore: EnrollmentStore
    private lateinit var identityMatcher: IdentityMatcher
    private lateinit var enrollmentHelper: EnrollmentHelper

    private val requestPermissionLauncher =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
            if (granted) startCamera() else {
                Toast.makeText(this, "Camera permission required", Toast.LENGTH_SHORT).show()
            }
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        OpenCVLoader.initDebug()

        previewView = findViewById(R.id.previewView)
        blurOverlay = findViewById(R.id.blurOverlay)
        detector = YoloDetector(this, "yolov8n-face.tflite")
        faceEmbedder = FaceEmbedder(this, "mobilefacenet.tflite")
        enrollmentStore = EnrollmentStore(this)
        identityMatcher = IdentityMatcher(faceEmbedder, enrollmentStore)
        enrollmentHelper = EnrollmentHelper(faceEmbedder, enrollmentStore)

        cameraExecutor = Executors.newSingleThreadExecutor()

        findViewById<ImageButton>(R.id.switchCameraButton).setOnClickListener {
            toggleCamera()
        }

        // 5-Second Continuous Whitelisting Enrollment Listener
        findViewById<Button>(R.id.btnRegisterFace)?.setOnClickListener {
            enrollmentHelper.startEnrollment(
                onProgress = { secondsLeft ->
                    runOnUiThread {
                        Toast.makeText(
                            this,
                            "Slowly rotate head... $secondsLeft sec left",
                            Toast.LENGTH_SHORT
                        ).show()
                    }
                },
                onComplete = {
                    // Refresh the identity matcher with the newly populated store
                    identityMatcher = IdentityMatcher(faceEmbedder, enrollmentStore)

                    runOnUiThread {
                        Toast.makeText(
                            this,
                            "Whitelisting Complete! Streamer Registered.",
                            Toast.LENGTH_LONG
                        ).show()
                    }
                }
            )
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
                Log.e("MainActivity", "Binding failed", e)
            }
        }, ContextCompat.getMainExecutor(this))
    }

    private var frameCount = 0
    private var lastBoxes: List<RectF> = emptyList()

    private fun cropToSquare(bitmap: Bitmap): Bitmap {
        val size = minOf(bitmap.width, bitmap.height)
        val x = (bitmap.width - size) / 2
        val y = (bitmap.height - size) / 2
        return Bitmap.createBitmap(bitmap, x, y, size, size)
    }

    inner class FrameAnalyzer : ImageAnalysis.Analyzer {
        override fun analyze(imageProxy: ImageProxy) {
            try {
                val isFrontCamera = lensFacing == CameraSelector.LENS_FACING_FRONT

                val fullBitmap = imageProxy.toUprightBitmap(isFrontCamera) ?: run {
                    Log.e("TEST", "Failed to convert ImageProxy to Bitmap")
                    return
                }

                val width = fullBitmap.width.toFloat()
                val height = fullBitmap.height.toFloat()

                val size = minOf(width, height)
                val leftOffset = (width - size) / 2f
                val topOffset = (height - size) / 2f

                val squareBitmap = cropToSquare(fullBitmap)
                val detections = detector.detect(squareBitmap)

                // 1. ENROLLMENT CAPTURE PHASE (5-Second Burst)
                if (enrollmentHelper.isEnrolling) {
                    for (det in detections) {
                        val faceCrop = FaceCropUtils.cropNormalized(squareBitmap, det.box)
                        if (faceCrop != null) {
                            enrollmentHelper.processFrameForEnrollment(faceCrop)
                            faceCrop.recycle()
                        }
                    }

                    if (squareBitmap !== fullBitmap) squareBitmap.recycle()

                    // Keep camera preview alive during 5s capture without blurring
                    val rawProcessedBitmap = blurOverlay.applyBlurToBoxes(fullBitmap, emptyList())
                    fullBitmap.recycle()

                    runOnUiThread {
                        blurOverlay.setFrame(rawProcessedBitmap)
                    }
                    return
                }

                // 2. NORMAL STREAMING / ANONYMIZATION PHASE
                val boxes: List<RectF>
                if (frameCount % 2 == 0) {
                    // Classify faces and identify who is NOT the streamer
                    val identified = identityMatcher.classify(squareBitmap, detections)

                    Log.d("SafeStreamDebug", "Faces detected: ${identified.size}")
                    identified.forEach { person ->
                        Log.d("SafeStreamDebug", "Streamer Match: ${person.isStreamer} (Similarity: ${person.similarity})")
                    }

                    val detectionsToBlur = identified.filter { !it.isStreamer }.map { it.detection }

                    boxes = detectionsToBlur.map { det ->
                        RectF(
                            (det.box.left * size + leftOffset) / width,
                            (det.box.top * size + topOffset) / height,
                            (det.box.right * size + leftOffset) / width,
                            (det.box.bottom * size + topOffset) / height
                        )
                    }
                    lastBoxes = boxes
                } else {
                    boxes = lastBoxes
                }
                frameCount++

                if (squareBitmap !== fullBitmap) squareBitmap.recycle()

                val processedBitmap = blurOverlay.applyBlurToBoxes(fullBitmap, boxes)
                fullBitmap.recycle()

                runOnUiThread {
                    blurOverlay.setFrame(processedBitmap)
                }
            } catch (e: Exception) {
                Log.e("TEST", "Error in FrameAnalyzer: ${e.message}", e)
            } finally {
                imageProxy.close()
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        detector.close()
        faceEmbedder.close()
        cameraExecutor.shutdown()
    }
}

fun ImageProxy.toUprightBitmap(mirror: Boolean = false): Bitmap? {
    val bitmap = try {
        this.toBitmap()
    } catch (e: Exception) {
        Log.e("MainActivity", "toBitmap failed", e)
        null
    } ?: return null

    val rotation = imageInfo.rotationDegrees.toFloat()
    val matrix = Matrix()

    if (rotation != 0f) matrix.postRotate(rotation)
    if (mirror) matrix.postScale(-1f, 1f)

    return if (rotation != 0f || mirror) {
        val result = Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
        bitmap.recycle()
        result
    } else {
        bitmap
    }
}