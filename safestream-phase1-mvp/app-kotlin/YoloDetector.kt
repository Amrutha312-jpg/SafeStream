package com.safestream.app

import android.content.Context
import android.graphics.Bitmap
import android.graphics.RectF
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.gpu.GpuDelegate
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import kotlin.math.max
import kotlin.math.min

/**
 * Wraps the YOLOv8-Nano .tflite face detection model.
 *
 * IMPORTANT: Run model-prep/verify_tflite_model.py on your exported model
 * FIRST and check the printed output shape matches OUTPUT_SIZE below.
 * If it doesn't, update INPUT_SIZE / OUTPUT_ROWS / OUTPUT_COLS to match --
 * this is the #1 source of "detections are nonsense" bugs.
 */
class YoloDetector(context: Context, modelPath: String = "yolov8n-face.tflite") {

    companion object {
        const val INPUT_SIZE = 320          // must match imgsz used in prepare_model.py
        const val OUTPUT_COLS = 2100        // anchor points; verify against your export
        const val CONF_THRESHOLD = 0.45f
        const val IOU_THRESHOLD = 0.45f
    }

    data class Detection(val box: RectF, val confidence: Float)

    private var interpreter: Interpreter
    private var gpuDelegate: GpuDelegate? = null

    init {
        val model = loadModelFile(context, modelPath)
        val options = Interpreter.Options()

        // Prefer GPU delegate; NNAPI is being phased out on newer Android
        // versions, so GPU delegate + CPU fallback is the safer default.
        val compatList = CompatibilityList()
        if (compatList.isDelegateSupportedOnThisDevice) {
            gpuDelegate = GpuDelegate(compatList.bestOptionsForThisDevice)
            options.addDelegate(gpuDelegate)
        } else {
            options.setNumThreads(4) // CPU fallback
        }

        interpreter = Interpreter(model, options)
    }

    private fun loadModelFile(context: Context, modelPath: String): MappedByteBuffer {
        val fd = context.assets.openFd(modelPath)
        val inputStream = FileInputStream(fd.fileDescriptor)
        val channel = inputStream.channel
        return channel.map(FileChannel.MapMode.READ_ONLY, fd.startOffset, fd.declaredLength)
    }

    /** bitmap should already be resized to INPUT_SIZE x INPUT_SIZE before calling this. */
    fun detect(bitmap: Bitmap): List<Detection> {
        require(bitmap.width == INPUT_SIZE && bitmap.height == INPUT_SIZE) {
            "Bitmap must be pre-resized to ${INPUT_SIZE}x$INPUT_SIZE"
        }

        val inputBuffer = bitmapToByteBuffer(bitmap)

        // Output shape [1, 5, OUTPUT_COLS] -> (x, y, w, h, conf) per anchor.
        // If verify_tflite_model.py showed a different shape, change this.
        val output = Array(1) { Array(5) { FloatArray(OUTPUT_COLS) } }

        interpreter.run(inputBuffer, output)

        return parseOutput(output[0])
    }

    private fun bitmapToByteBuffer(bitmap: Bitmap): ByteBuffer {
        val buffer = ByteBuffer.allocateDirect(4 * INPUT_SIZE * INPUT_SIZE * 3)
        buffer.order(ByteOrder.nativeOrder())
        val pixels = IntArray(INPUT_SIZE * INPUT_SIZE)
        bitmap.getPixels(pixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE)

        for (pixel in pixels) {
            buffer.putFloat(((pixel shr 16) and 0xFF) / 255.0f) // R
            buffer.putFloat(((pixel shr 8) and 0xFF) / 255.0f)  // G
            buffer.putFloat((pixel and 0xFF) / 255.0f)          // B
        }
        buffer.rewind()
        return buffer
    }

    private fun parseOutput(output: Array<FloatArray>): List<Detection> {
        val candidates = mutableListOf<Detection>()

        for (i in 0 until OUTPUT_COLS) {
            val conf = output[4][i]
            if (conf < CONF_THRESHOLD) continue

            val cx = output[0][i]
            val cy = output[1][i]
            val w = output[2][i]
            val h = output[3][i]

            // Coordinates come out normalized to INPUT_SIZE -- convert to
            // 0..1 range so downstream code can scale to any display size.
            val left = (cx - w / 2f) / INPUT_SIZE
            val top = (cy - h / 2f) / INPUT_SIZE
            val right = (cx + w / 2f) / INPUT_SIZE
            val bottom = (cy + h / 2f) / INPUT_SIZE

            candidates.add(
                Detection(
                    RectF(
                        max(0f, left), max(0f, top),
                        min(1f, right), min(1f, bottom)
                    ),
                    conf
                )
            )
        }

        return nonMaxSuppression(candidates)
    }

    private fun nonMaxSuppression(detections: List<Detection>): List<Detection> {
        val sorted = detections.sortedByDescending { it.confidence }.toMutableList()
        val result = mutableListOf<Detection>()

        while (sorted.isNotEmpty()) {
            val best = sorted.removeAt(0)
            result.add(best)
            sorted.removeAll { iou(best.box, it.box) > IOU_THRESHOLD }
        }
        return result
    }

    private fun iou(a: RectF, b: RectF): Float {
        val interLeft = max(a.left, b.left)
        val interTop = max(a.top, b.top)
        val interRight = min(a.right, b.right)
        val interBottom = min(a.bottom, b.bottom)

        val interArea = max(0f, interRight - interLeft) * max(0f, interBottom - interTop)
        val aArea = (a.right - a.left) * (a.bottom - a.top)
        val bArea = (b.right - b.left) * (b.bottom - b.top)

        return interArea / (aArea + bArea - interArea + 1e-6f)
    }

    fun close() {
        interpreter.close()
        gpuDelegate?.close()
    }
}
