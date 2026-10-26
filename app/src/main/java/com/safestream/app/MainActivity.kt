package com.safestream.app

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Matrix
import android.graphics.RectF
import android.os.Bundle
import android.util.Log
import android.util.Size
import android.widget.Button
import android.widget.EditText
import android.widget.ImageButton
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.*
import androidx.camera.core.resolutionselector.AspectRatioStrategy
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import com.pedro.common.ConnectChecker
import org.opencv.android.OpenCVLoader
import java.io.File
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

    // ---- Stage A: local recording ----
    private lateinit var streamManager: StreamManager
    @Volatile private var isRecording = false
    @Volatile private var pendingRecordingStart = false

    // ---- Stage B: live RTMPS ----
    @Volatile private var pendingStreamStart = false
    private var pendingStreamUrl: String = ""

    private val connectChecker = object : ConnectChecker {
        override fun onConnectionStarted(url: String) {
            Log.d("SafeStream_Stream", "Connection started: $url")
        }
        override fun onConnectionSuccess() {
            Log.d("SafeStream_Stream", "Connection success")
            runOnUiThread {
                findViewById<Button>(R.id.btnGoLive)?.text = "Stop Live"
                Toast.makeText(this@MainActivity, "Live!", Toast.LENGTH_SHORT).show()
            }
        }
        override fun onConnectionFailed(reason: String) {
            Log.e("SafeStream_Stream", "Connection failed: $reason")
            runOnUiThread {
                findViewById<Button>(R.id.btnGoLive)?.text = "Go Live"
                Toast.makeText(this@MainActivity, "Stream failed: $reason", Toast.LENGTH_LONG).show()
            }
        }
        override fun onDisconnect() {
            Log.d("SafeStream_Stream", "Disconnected")
            runOnUiThread {
                findViewById<Button>(R.id.btnGoLive)?.text = "Go Live"
            }
        }
        override fun onAuthError() {
            Log.e("SafeStream_Stream", "Auth error")
            runOnUiThread {
                findViewById<Button>(R.id.btnGoLive)?.text = "Go Live"
                Toast.makeText(this@MainActivity, "Auth error - check your stream key", Toast.LENGTH_LONG).show()
            }
        }
        override fun onAuthSuccess() {
            Log.d("SafeStream_Stream", "Auth success")
        }
    }

    private val requestPermissionsLauncher =
        registerForActivityResult(ActivityResultContracts.RequestMultiplePermissions()) { results ->
            val cameraGranted = results[Manifest.permission.CAMERA] == true
            val audioGranted = results[Manifest.permission.RECORD_AUDIO] == true
            if (cameraGranted) {
                startCamera()
            } else {
                Toast.makeText(this, "Camera permission required", Toast.LENGTH_SHORT).show()
            }
            if (!audioGranted) {
                Toast.makeText(this, "Microphone permission needed for recording", Toast.LENGTH_SHORT).show()
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
        streamManager = StreamManager(this, connectChecker)

        cameraExecutor = Executors.newSingleThreadExecutor()

        findViewById<ImageButton>(R.id.switchCameraButton).setOnClickListener {
            toggleCamera()
        }

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

        findViewById<Button>(R.id.btnRecord)?.setOnClickListener { view ->
            if (!isRecording) {
                pendingRecordingStart = true
                (view as? Button)?.text = "Starting..."
            } else {
                streamManager.stopLocalRecording()
                isRecording = false
                (view as? Button)?.text = "Start Recording"
                Toast.makeText(this, "Recording stopped", Toast.LENGTH_SHORT).show()
            }
        }

        findViewById<Button>(R.id.btnGoLive)?.setOnClickListener { view ->
            if (!streamManager.isStreaming) {
                val url = findViewById<EditText>(R.id.etStreamUrl).text.toString().trim()
                if (url.isEmpty()) {
                    Toast.makeText(this, "Paste your RTMPS URL first", Toast.LENGTH_SHORT).show()
                    return@setOnClickListener
                }
                pendingStreamUrl = url
                pendingStreamStart = true
                (view as? Button)?.text = "Connecting..."
            } else {
                streamManager.stopStream()
                (view as? Button)?.text = "Go Live"
                Toast.makeText(this, "Live stream stopped", Toast.LENGTH_SHORT).show()
            }
        }
        findViewById<Button>(R.id.btnLogout)?.setOnClickListener {
            getSharedPreferences("SafeStreamSession", MODE_PRIVATE)
                .edit().remove("current_user").apply()
            startActivity(
                android.content.Intent(this, LoginActivity::class.java)
                    .addFlags(android.content.Intent.FLAG_ACTIVITY_CLEAR_TOP)
            )
            finish()
        }

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
            == PackageManager.PERMISSION_GRANTED
        ) {
            startCamera()
        } else {
            requestPermissionsLauncher.launch(
                arrayOf(Manifest.permission.CAMERA, Manifest.permission.RECORD_AUDIO)
            )
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

            val previewResolutionSelector = ResolutionSelector.Builder()
                .setAspectRatioStrategy(AspectRatioStrategy.RATIO_16_9_FALLBACK_AUTO_STRATEGY)
                .build()

            val preview = Preview.Builder()
                .setResolutionSelector(previewResolutionSelector)
                .build().also {
                    it.setSurfaceProvider(previewView.surfaceProvider)
                }

            val analysisResolutionSelector = ResolutionSelector.Builder()
                .setAspectRatioStrategy(AspectRatioStrategy.RATIO_16_9_FALLBACK_AUTO_STRATEGY)
                .setResolutionStrategy(
                    ResolutionStrategy(Size(640, 360), ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER)
                )
                .build()

            val imageAnalyzer = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setResolutionSelector(analysisResolutionSelector)
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
    private var lastDetections: List<YoloDetector.Detection> = emptyList()
    private var dimsLogCounter = 0

    // NEW -- timing diagnostics to find the real cause of the quality/lag
    // complaint, instead of guessing again. Logs every ~15th frame so
    // Logcat stays readable but samples come frequently enough during a
    // short test. Filter Logcat by "SafeStream_Timing".
    private var timingLogCounter = 0

    private fun cropToSquare(bitmap: Bitmap): Bitmap {
        val size = minOf(bitmap.width, bitmap.height)
        val x = (bitmap.width - size) / 2
        val y = (bitmap.height - size) / 2
        return Bitmap.createBitmap(bitmap, x, y, size, size)
    }

    inner class FrameAnalyzer : ImageAnalysis.Analyzer {
        override fun analyze(imageProxy: ImageProxy) {
            val tFrameStart = System.nanoTime()
            try {
                val isFrontCamera = lensFacing == CameraSelector.LENS_FACING_FRONT

                val tConvertStart = System.nanoTime()
                val fullBitmap = imageProxy.toUprightBitmap(isFrontCamera) ?: run {
                    Log.e("TEST", "Failed to convert ImageProxy to Bitmap")
                    return
                }
                val tConvertEnd = System.nanoTime()

                dimsLogCounter++
                if (dimsLogCounter % 30 == 0) {
                    Log.d(
                        "SafeStream_Dims",
                        "fullBitmap: ${fullBitmap.width}x${fullBitmap.height}, " +
                                "imageProxy: ${imageProxy.width}x${imageProxy.height}, " +
                                "rotation: ${imageProxy.imageInfo.rotationDegrees}"
                    )
                }

                if (pendingRecordingStart) {
                    pendingRecordingStart = false
                    val ready = if (streamManager.isPrepared) true
                    else streamManager.prepare(fullBitmap.width, fullBitmap.height)
                    if (ready) {
                        val outputFile = File(
                            getExternalFilesDir(null),
                            "safestream_${System.currentTimeMillis()}.mp4"
                        )
                        streamManager.startLocalRecording(outputFile.absolutePath) { status ->
                            Log.d("SafeStream_Record", "Status: $status")
                        }
                        isRecording = true
                        runOnUiThread {
                            findViewById<Button>(R.id.btnRecord)?.text = "Stop Recording"
                            Toast.makeText(
                                this@MainActivity,
                                "Recording: ${outputFile.name}",
                                Toast.LENGTH_SHORT
                            ).show()
                        }
                    } else {
                        runOnUiThread {
                            findViewById<Button>(R.id.btnRecord)?.text = "Start Recording"
                            Toast.makeText(this@MainActivity, "Failed: ${streamManager.lastError}", Toast.LENGTH_LONG).show()
                        }
                    }
                }

                if (pendingStreamStart) {
                    pendingStreamStart = false
                    val ready = if (streamManager.isPrepared) true
                    else streamManager.prepare(fullBitmap.width, fullBitmap.height)
                    if (ready) {
                        streamManager.startRtmpsStream(pendingStreamUrl)
                    } else {
                        runOnUiThread {
                            findViewById<Button>(R.id.btnGoLive)?.text = "Go Live"
                            Toast.makeText(this@MainActivity, "Failed: ${streamManager.lastError}", Toast.LENGTH_LONG).show()
                        }
                    }
                }

                val width = fullBitmap.width.toFloat()
                val height = fullBitmap.height.toFloat()

                val size = minOf(width, height)
                val leftOffset = (width - size) / 2f
                val topOffset = (height - size) / 2f

                val tCropStart = System.nanoTime()
                val squareBitmap = cropToSquare(fullBitmap)
                val tCropEnd = System.nanoTime()

                if (enrollmentHelper.isEnrolling) {
                    val enrollDetections = detector.detect(squareBitmap)
                    for (det in enrollDetections) {
                        val faceCrop = FaceCropUtils.cropNormalized(squareBitmap, det.box)
                        if (faceCrop != null) {
                            enrollmentHelper.processFrameForEnrollment(faceCrop)
                            faceCrop.recycle()
                        }
                    }

                    if (squareBitmap !== fullBitmap) squareBitmap.recycle()

                    val rawProcessedBitmap = blurOverlay.applyBlurToBoxes(fullBitmap, emptyList())
                    if (rawProcessedBitmap !== fullBitmap) fullBitmap.recycle()

                    runOnUiThread {
                        blurOverlay.setFrame(rawProcessedBitmap)
                    }
                    return
                }

                // Timing markers -- populated only on frames that run
                // detection (frameCount % 2 == 0); left at 0 on skipped
                // frames, and the log line below reports "skip" for those.
                var tDetectStart = 0L
                var tDetectEnd = 0L
                var tMatchStart = 0L
                var tMatchEnd = 0L
                var detectedCount = 0

                val boxes: List<RectF>
                if (frameCount % 3 == 0) {
                    tDetectStart = System.nanoTime()
                    val detections = detector.detect(squareBitmap)
                    tDetectEnd = System.nanoTime()
                    lastDetections = detections
                    detectedCount = detections.size

                    tMatchStart = System.nanoTime()
                    val identified = identityMatcher.classify(squareBitmap, detections)
                    tMatchEnd = System.nanoTime()

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

                val tBlurStart = System.nanoTime()
                val processedBitmap = blurOverlay.applyBlurToBoxes(fullBitmap, boxes)
                val tBlurEnd = System.nanoTime()

                if (processedBitmap !== fullBitmap) fullBitmap.recycle()

                val tPushStart = System.nanoTime()
                // Fixed vs. the version you're running: previously this only
                // checked isRecording, so pushFrame() was never called
                // during a live-only (no local recording) session -- no
                // frames would have reached the RTMPS stream at all.
                if (isRecording || streamManager.isStreaming) {
                    streamManager.pushFrame(processedBitmap)
                }
                val tPushEnd = System.nanoTime()

                runOnUiThread {
                    blurOverlay.setFrame(processedBitmap)
                }

                val tFrameEnd = System.nanoTime()

                timingLogCounter++
                if (timingLogCounter % 7 == 0) {
                    fun ms(start: Long, end: Long) = "%.1f".format((end - start) / 1_000_000.0)
                    val detectStr = if (tDetectEnd > 0) ms(tDetectStart, tDetectEnd) else "skip"
                    val matchStr = if (tMatchEnd > 0) ms(tMatchStart, tMatchEnd) else "skip"
                    Log.d(
                        "SafeStream_Timing",
                        "TOTAL=${ms(tFrameStart, tFrameEnd)}ms | " +
                                "convert=${ms(tConvertStart, tConvertEnd)} " +
                                "crop=${ms(tCropStart, tCropEnd)} " +
                                "detect=$detectStr (freshFaces=$detectedCount blurBoxes=${boxes.size}) " +
                                "match=$matchStr " +
                                "blur=${ms(tBlurStart, tBlurEnd)} " +
                                "push=${ms(tPushStart, tPushEnd)}"
                    )
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
        try { if (isRecording) streamManager.stopLocalRecording() } catch (e: Exception) { Log.w("MainActivity", "stop rec", e) }
        try { if (streamManager.isStreaming) streamManager.stopStream() } catch (e: Exception) { Log.w("MainActivity", "stop stream", e) }
        try { streamManager.release() } catch (e: Exception) { Log.w("MainActivity", "release", e) }
        try { detector.close() } catch (e: Exception) { Log.w("MainActivity", "detector", e) }
        try { faceEmbedder.close() } catch (e: Exception) { Log.w("MainActivity", "embedder", e) }
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