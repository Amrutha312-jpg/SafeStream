package com.safestream.app

import android.content.Context
import android.graphics.*
import android.util.AttributeSet
import android.view.View
import org.opencv.android.Utils
import org.opencv.core.Mat
import org.opencv.core.Rect as CvRect
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc

/**
 * Sits on top of the CameraX PreviewView (same size, same position).
 * Each new frame: takes the raw camera bitmap + the face boxes YoloDetector
 * found, applies real OpenCV Gaussian blur to just those regions, and draws
 * the result. This satisfies the report's "Gaussian Blur Redaction using
 * OpenCV" step (Ch.4) without needing a full native video pipeline for the
 * MVP -- streaming/encoding comes in a later milestone.
 *
 * Make sure OpenCV's native library is loaded once at app startup, e.g. in
 * MainActivity's onCreate:  System.loadLibrary("opencv_java4")
 * (or use OpenCVLoader.initDebug() if using the newer SDK loader).
 */
class BlurOverlayView @JvmOverloads constructor(
    context: Context, attrs: AttributeSet? = null
) : View(context, attrs) {

    private var currentBitmap: Bitmap? = null
    private var boxes: List<RectF> = emptyList()

    /** Call this once per analyzed frame from MainActivity. */
    fun updateFrame(sourceBitmap: Bitmap, detectionBoxes: List<RectF>) {
        boxes = detectionBoxes
        currentBitmap = applyBlurToBoxes(sourceBitmap, detectionBoxes)
        invalidate()
    }

    private fun applyBlurToBoxes(source: Bitmap, normalizedBoxes: List<RectF>): Bitmap {
        if (normalizedBoxes.isEmpty()) return source

        val mat = Mat()
        Utils.bitmapToMat(source, mat)

        for (box in normalizedBoxes) {
            // Convert normalized (0..1) coords to pixel coords on this bitmap
            val left = (box.left * source.width).toInt().coerceIn(0, source.width - 1)
            val top = (box.top * source.height).toInt().coerceIn(0, source.height - 1)
            val right = (box.right * source.width).toInt().coerceIn(left + 1, source.width)
            val bottom = (box.bottom * source.height).toInt().coerceIn(top + 1, source.height)

            val roiRect = CvRect(left, top, right - left, bottom - top)
            val roi = Mat(mat, roiRect)

            // Kernel size must be odd and scales with box size so small/far
            // faces still get fully obscured, not just softened.
            val k = (minOf(roiRect.width, roiRect.height) / 3).let { if (it % 2 == 0) it + 1 else it }
                .coerceAtLeast(15)
            Imgproc.GaussianBlur(roi, roi, Size(k.toDouble(), k.toDouble()), 0.0)
            roi.copyTo(Mat(mat, roiRect))
        }

        val outBitmap = Bitmap.createBitmap(source.width, source.height, Bitmap.Config.ARGB_8888)
        Utils.matToBitmap(mat, outBitmap)
        mat.release()
        return outBitmap
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        currentBitmap?.let { bmp ->
            val destRect = android.graphics.Rect(0, 0, width, height)
            canvas.drawBitmap(bmp, null, destRect, null)
        }
    }
}
