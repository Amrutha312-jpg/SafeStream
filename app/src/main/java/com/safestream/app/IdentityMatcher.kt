package com.safestream.app

import android.graphics.Bitmap
import android.util.Log

data class IdentifiedDetection(
    val detection: YoloDetector.Detection,
    val isStreamer: Boolean,
    val similarity: Float
)

class IdentityMatcher(
    private val faceEmbedder: FaceEmbedder,
    private val enrollmentStore: EnrollmentStore,
    private val similarityThreshold: Float = 0.48f // 👈 Stricter threshold to block other people
) {
    // Memory buffer to prevent frame-by-frame flicker
    private var consecutiveStreamerFrames = 0
    private val debounceThreshold = 2 // Shorter hold to prevent strangers from staying unblurred

    fun classify(
        squareBitmap: Bitmap,
        detections: List<YoloDetector.Detection>
    ): List<IdentifiedDetection> {
        val enrolledEmbeddings = enrollmentStore.loadAll()
        Log.d("SafeStreamDebug", "Loaded enrolled embeddings count: ${enrolledEmbeddings.size}")

        if (enrolledEmbeddings.isEmpty()) {
            return detections.map { IdentifiedDetection(it, isStreamer = false, similarity = 0f) }
        }

        return detections.map { det ->
            val faceCrop = FaceCropUtils.cropNormalized(squareBitmap, det.box)
            if (faceCrop == null) {
                IdentifiedDetection(det, isStreamer = false, similarity = 0f)
            } else {
                val liveEmbedding = faceEmbedder.embed(faceCrop)

                // Compare against ALL stored face poses and find the best match
                val maxSimilarity = enrolledEmbeddings.maxOfOrNull { storedEmbedding ->
                    FaceEmbedder.cosineSimilarity(liveEmbedding, storedEmbedding)
                } ?: 0f

                val rawMatch = maxSimilarity >= similarityThreshold

                // Apply Temporal Debouncing strictly for the streamer
                val smoothedMatch = if (rawMatch) {
                    consecutiveStreamerFrames = debounceThreshold
                    true
                } else if (consecutiveStreamerFrames > 0) {
                    consecutiveStreamerFrames--
                    true
                } else {
                    false
                }

                Log.d("SafeStreamDebug", "Best Similarity = $maxSimilarity | RawMatch = $rawMatch | Smoothed = $smoothedMatch")
                IdentifiedDetection(det, smoothedMatch, maxSimilarity)
            }
        }
    }
}