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

class BlurOverlayView @JvmOverloads constructor(
    context: Context, attrs: AttributeSet? = null
) : View(context, attrs) {

    private var currentBitmap: Bitmap? = null
    private var boxes: List<RectF> = emptyList()

    /**
     * Updates the view with a pre-processed (blurred) bitmap.
     * This should be called from the UI thread.
     */
    fun setFrame(blurredBitmap: Bitmap) {
        android.util.Log.d("BlurOverlayView", "setFrame: bmp=${blurredBitmap.width}x${blurredBitmap.height}")
        val oldBmp = currentBitmap
        currentBitmap = blurredBitmap
        
        // Recycle the old display bitmap to prevent memory leaks
        if (oldBmp != null && oldBmp !== blurredBitmap) {
            oldBmp.recycle()
        }
        invalidate()
    }

    /**
     * Heavy OpenCV processing. Should be called from a background thread.
     */
    fun applyBlurToBoxes(source: Bitmap, normalizedBoxes: List<RectF>): Bitmap {
        android.util.Log.d("BlurOverlayView", "applyBlurToBoxes: blurring ${normalizedBoxes.size} boxes")
        if (normalizedBoxes.isEmpty()) {
            // Return a copy if we need to modify it later, or just the original
            return source.copy(source.config!!, true)
        }

        val mat = Mat()
        Utils.bitmapToMat(source, mat)

        for (box in normalizedBoxes) {
            val left = (box.left * source.width).toInt().coerceIn(0, source.width - 1)
            val top = (box.top * source.height).toInt().coerceIn(0, source.height - 1)
            val right = (box.right * source.width).toInt().coerceIn(left + 1, source.width)
            val bottom = (box.bottom * source.height).toInt().coerceIn(top + 1, source.height)

            if (right <= left || bottom <= top) continue

            val roiRect = CvRect(left, top, right - left, bottom - top)
            val roi = Mat(mat, roiRect)

            // Dynamic blur strength based on face size
            val k = (minOf(roiRect.width, roiRect.height) / 3)
                .let { if (it % 2 == 0) it + 1 else it }
                .coerceAtLeast(15)
            
            Imgproc.GaussianBlur(roi, roi, Size(k.toDouble(), k.toDouble()), 0.0)
            roi.release() // Release ROI header
        }

        val outBitmap = Bitmap.createBitmap(source.width, source.height, Bitmap.Config.ARGB_8888)
        Utils.matToBitmap(mat, outBitmap)
        mat.release()
        return outBitmap
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        currentBitmap?.let { bmp ->
            val viewWidth = width.toFloat()
            val viewHeight = height.toFloat()
            val bmpWidth = bmp.width.toFloat()
            val bmpHeight = bmp.height.toFloat()

            if (viewWidth == 0f || viewHeight == 0f) return

            // Calculate center-crop scale and translation to match PreviewView.fill_center
            val scale = Math.max(viewWidth / bmpWidth, viewHeight / bmpHeight)
            val dx = (viewWidth - bmpWidth * scale) / 2f
            val dy = (viewHeight - bmpHeight * scale) / 2f

            val matrix = Matrix()
            matrix.postScale(scale, scale)
            matrix.postTranslate(dx, dy)

            canvas.drawBitmap(bmp, matrix, null)
            
            // Debug: Draw a small red circle to prove onDraw is running
            val debugPaint = Paint().apply { color = Color.RED; style = Paint.Style.FILL }
            canvas.drawCircle(30f, 30f, 15f, debugPaint)
        }
    }
}
