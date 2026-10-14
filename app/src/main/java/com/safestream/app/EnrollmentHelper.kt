package com.safestream.app

import android.graphics.Bitmap
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.launch
import kotlinx.coroutines.delay

/**
 * Matches the actual MainActivity call sites:
 *   enrollmentHelper.startEnrollment(onProgress = {...}, onComplete = {...})
 *   enrollmentHelper.isEnrolling
 *   enrollmentHelper.processFrameForEnrollment(faceCrop: Bitmap)
 *
 * Runs a fixed-duration (5s) "continuous" enrollment burst: every face crop
 * passed in via processFrameForEnrollment() during that window gets
 * embedded and stored as a sample -- giving multiple embeddings per
 * streamer (different angles as they rotate their head, per the
 * "Slowly rotate head..." onProgress prompt) rather than one single shot.
 */
class EnrollmentHelper(
    private val faceEmbedder: FaceEmbedder,
    private val enrollmentStore: EnrollmentStore,
    private val durationSeconds: Int = 5
) {
    @Volatile
    var isEnrolling: Boolean = false
        private set

    private var job: Job? = null

    fun startEnrollment(onProgress: (secondsLeft: Int) -> Unit, onComplete: () -> Unit) {
        if (isEnrolling) return // already running -- ignore a second tap

        enrollmentStore.clear() // fresh session: don't mix in a previous person's samples
        isEnrolling = true

        job = CoroutineScope(Dispatchers.Default).launch {
            for (secondsLeft in durationSeconds downTo 1) {
                onProgress(secondsLeft)
                delay(1000)
            }
            isEnrolling = false
            onComplete()
        }
    }

    /**
     * Called once per detected face, per frame, while isEnrolling is true.
     * faceCrop arrives already cropped (via FaceCropUtils in MainActivity).
     *
     * WORTH KNOWING: MainActivity currently calls this for EVERY face
     * detected in the frame, not just the largest/closest one. If anyone
     * besides the streamer is visible during the 5-second window, their
     * face gets embedded into the SAME sample pool as the streamer's --
     * which can hurt matching accuracy afterward (their embedding would
     * then partly represent a bystander too). For now the safest thing is
     * making sure only the streamer is in frame while enrolling. If you
     * want, I can add a largest-face-only filter here, but that needs the
     * full detection list (with sizes), not just the cropped bitmap, so it
     * would mean a small change to the call site in MainActivity too.
     */
    fun processFrameForEnrollment(faceCrop: Bitmap) {
        if (!isEnrolling) return
        try {
            val embedding = faceEmbedder.embed(faceCrop)
            enrollmentStore.addSample(embedding)
        } catch (e: Exception) {
            // don't let one bad frame kill the whole enrollment burst
        }
    }

    fun samplesCaptured(): Int = enrollmentStore.sampleCount()

    /** Optional: call from onDestroy() if you want the countdown to stop
     * cleanly if the Activity is destroyed mid-enrollment. Not required
     * for this to compile or run -- just avoids a harmless orphaned
     * coroutine in that edge case. */
    fun cancelEnrollment() {
        job?.cancel()
        isEnrolling = false
    }
}