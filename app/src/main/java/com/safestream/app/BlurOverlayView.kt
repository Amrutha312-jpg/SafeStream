package com.safestream.app

import android.content.Context
import android.graphics.*
import android.util.AttributeSet
import android.view.View
import org.opencv.android.Utils
import org.opencv.core.Mat
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc

class BlurOverlayView @JvmOverloads constructor(
    context: Context, attrs: AttributeSet? = null
) : View(context, attrs) {

    private var currentBitmap: Bitmap? = null

    // NEW: how much extra room to blur beyond the raw detection box, as a
    // fraction of that box's own width/height. 0.30 = 30% padding on every
    // side. Two reasons this helps:
    //  1. Requested directly -- a bigger blur box reads as more comfortably
    //     "fully covering" the face instead of hugging it tightly.
    //  2. YOLO boxes shrink/shift when a face is partially out of frame or
    //     at an angle, so a bit of extra margin keeps ears/jaw/hairline
    //     covered even when the raw box undershoots slightly.
    // Tune this single number if it ever looks too big or too small.
    private val blurBoxPadding = 0.30f

    // Back to the original Gaussian-blur look (requested), just stronger.
    // The old cap (15-41px) meant a large, close-up face got the same max
    // blur strength as a small one -- not enough to be unrecognizable.
    // Raising the cap and applying the blur twice (a cheap way to roughly
    // double effective strength without doubling kernel size) fixes that
    // without changing the visual style or reintroducing the lag spikes.
    private val blurKernelMax = 61

    fun setFrame(blurredBitmap: Bitmap) {
        val oldBmp = currentBitmap
        currentBitmap = blurredBitmap
        if (oldBmp != null && oldBmp !== blurredBitmap) {
            oldBmp.recycle()
        }
        invalidate()
    }

    /**
     * Heavy OpenCV processing. Should be called from a background thread.
     *
     * Only the cropped (and now padded) face region gets converted to Mat.
     * When there are no boxes, this returns `source` directly with NO copy
     * at all -- the caller must NOT recycle `source` when no boxes were
     * blurred, since the returned bitmap IS source in that case.
     */
    fun applyBlurToBoxes(source: Bitmap, normalizedBoxes: List<RectF>): Bitmap {
        if (normalizedBoxes.isEmpty()) {
            return source
        }

        val output = source.copy(Bitmap.Config.ARGB_8888, true)
        val canvas = Canvas(output)

        for (box in normalizedBoxes) {
            // Expand the box by blurBoxPadding on each side BEFORE converting
            // to pixel coordinates, so the padding scales with face size
            // (a face close to the camera gets more absolute padding than
            // a small, distant one).
            val boxW = box.right - box.left
            val boxH = box.bottom - box.top
            val padX = boxW * blurBoxPadding
            val padY = boxH * blurBoxPadding

            val paddedLeftN = box.left - padX
            val paddedTopN = box.top - padY
            val paddedRightN = box.right + padX
            val paddedBottomN = box.bottom + padY

            val left = (paddedLeftN * source.width).toInt().coerceIn(0, source.width - 1)
            val top = (paddedTopN * source.height).toInt().coerceIn(0, source.height - 1)
            val right = (paddedRightN * source.width).toInt().coerceIn(left + 1, source.width)
            val bottom = (paddedBottomN * source.height).toInt().coerceIn(top + 1, source.height)
            if (right <= left || bottom <= top) continue

            val w = right - left
            val h = bottom - top

            val patch = Bitmap.createBitmap(output, left, top, w, h)
            val mat = Mat()
            Utils.bitmapToMat(patch, mat)

            val k = (minOf(w, h) / 3)
                .let { if (it % 2 == 0) it + 1 else it }
                .coerceIn(15, blurKernelMax)

            // Two passes at half-ish cost each read as noticeably stronger
            // than one pass at the same kernel size, without the O(k^2)
            // cost of just doubling k in a single call.
            Imgproc.GaussianBlur(mat, mat, Size(k.toDouble(), k.toDouble()), 0.0)
            Imgproc.GaussianBlur(mat, mat, Size(k.toDouble(), k.toDouble()), 0.0)
            Utils.matToBitmap(mat, patch)
            mat.release()

            canvas.drawBitmap(patch, left.toFloat(), top.toFloat(), null)
            patch.recycle()
        }

        return output
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        currentBitmap?.let { bmp ->
            val viewWidth = width.toFloat()
            val viewHeight = height.toFloat()
            val bmpWidth = bmp.width.toFloat()
            val bmpHeight = bmp.height.toFloat()

            if (viewWidth == 0f || viewHeight == 0f) return

            val scale = Math.max(viewWidth / bmpWidth, viewHeight / bmpHeight)
            val dx = (viewWidth - bmpWidth * scale) / 2f
            val dy = (viewHeight - bmpHeight * scale) / 2f

            val matrix = Matrix()
            matrix.postScale(scale, scale)
            matrix.postTranslate(dx, dy)

            canvas.drawBitmap(bmp, matrix, null)
        }
    }
}