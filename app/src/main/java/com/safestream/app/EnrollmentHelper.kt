package com.safestream.app

import android.graphics.Bitmap
import android.os.CountDownTimer
import android.os.Handler
import android.os.Looper
import android.util.Log

class EnrollmentHelper(
    private val faceEmbedder: FaceEmbedder,
    private val enrollmentStore: EnrollmentStore
) {
    var isEnrolling = false
        private set

    private var framesCollected = 0

    fun startEnrollment(onProgress: (Int) -> Unit, onComplete: () -> Unit) {
        if (isEnrolling) return
        isEnrolling = true
        framesCollected = 0
        enrollmentStore.clear() // Clear old data for a fresh 5-second capture

        Log.d("TEST", "Starting 5-second continuous enrollment...")

        // Run timer on Main Thread so UI updates (like Toasts) work safely
        Handler(Looper.getMainLooper()).post {
            object : CountDownTimer(5000, 1000) {
                override fun onTick(millisUntilFinished: Long) {
                    val secondsLeft = (millisUntilFinished / 1000).toInt()
                    onProgress(secondsLeft)
                }

                override fun onFinish() {
                    isEnrolling = false
                    onComplete()
                    Log.d("TEST", "Enrollment finished! Collected $framesCollected frames.")
                }
            }.start()
        }
    }

    // This will be called rapidly by your FrameAnalyzer for 5 seconds
    fun processFrameForEnrollment(faceCrop: Bitmap) {
        if (!isEnrolling) return
        if (framesCollected >= 30) return // Cap at 30 to prevent memory bloat

        try {
            val embedding = faceEmbedder.embed(faceCrop)
            enrollmentStore.save(embedding)
            framesCollected++
            Log.d("TEST", "Captured and saved streamer embedding #$framesCollected")
        } catch (e: Exception) {
            Log.e("TEST", "Failed to process enrollment frame", e)
        }
    }
}