package com.safestream.app

import android.graphics.Bitmap
import android.graphics.RectF
import android.util.Log
import kotlin.math.max
import kotlin.math.min

data class IdentifiedDetection(
    val detection: YoloDetector.Detection,
    val isStreamer: Boolean,
    val similarity: Float
)

class IdentityMatcher(
    private val faceEmbedder: FaceEmbedder,
    private val enrollmentStore: EnrollmentStore,
    // Kept at 0.58 for a fresh/first-time match. Partial-face frames are
    // handled separately below instead of by loosening this.
    private var similarityThreshold: Float = 0.58f
) {
    private var consecutiveStreamerFrames = 0

    // Slightly lower than the previous 6 -- that value, combined with the OR
    // rescue below, was holding "streamer" status alive for too long/too
    // loosely. 5 still comfortably covers a short run of partial-face frames.
    private val debounceThreshold = 5

    private var lastStreamerBox: RectF? = null

    // FIXED BUG: a bystander standing close to the streamer could satisfy
    // EITHER the spatial check OR the score check alone (they were combined
    // with OR), so a friend who simply stepped near your last known box
    // position could inherit "streamer" status even with a mediocre identity
    // score, and briefly get left unblurred while you got blurred instead.
    // Both signals now have to agree together (AND) that the currently-held
    // rescue is still looking at the SAME real person: close to where the
    // streamer was AND still scoring meaningfully like the streamer. Raised
    // both thresholds back up from the too-permissive values that caused
    // this.
    private val keepAliveThreshold = 0.42f
    private val iouRescueThreshold = 0.45f

    // NEW: raw per-frame similarity is noisy (0.33-0.94 for the SAME person
    // across the same short clip -- normal for a quantized on-device
    // embedding model, worse at distance/angle where the face crop has
    // fewer real pixels in it). Comparing a single frame's raw score
    // straight against the thresholds meant one unlucky frame could flip
    // the enrolled streamer to "bystander" and back a moment later.
    // Smoothing the top candidate's score across frames absorbs that
    // single-frame noise without meaningfully slowing down a real
    // person-swap, since an actual bystander's score stays consistently
    // low rather than just dipping once.
    private var emaScore = 0f

    // Weight given to the newest frame's score. Lower = smoother / more
    // resistant to single-frame noise, but slower to react to an actual
    // person swap.
    private val emaAlpha = 0.4f

    fun classify(
        squareBitmap: Bitmap,
        detections: List<YoloDetector.Detection>
    ): List<IdentifiedDetection> {
        val enrolledEmbeddings = enrollmentStore.loadAll()

        if (enrolledEmbeddings.isEmpty() || detections.isEmpty()) {
            return detections.map { IdentifiedDetection(it, isStreamer = false, similarity = 0f) }
        }

        // STEP 1: Score EVERY face in the frame first, before making any decisions
        val scoredFaces = detections.map { det ->
            val faceCrop = FaceCropUtils.cropNormalized(squareBitmap, det.box)
            val maxSimilarity = if (faceCrop != null) {
                val liveEmbedding = faceEmbedder.embed(faceCrop)
                enrolledEmbeddings.maxOfOrNull { stored ->
                    FaceEmbedder.cosineSimilarity(liveEmbedding, stored)
                } ?: 0f
            } else {
                0f
            }
            Pair(det, maxSimilarity)
        }

        // STEP 2: Find the single highest scoring face in this specific frame
        val bestMatchInFrame = scoredFaces.maxByOrNull { it.second }
        var streamerFoundThisFrame = false

        // STEP 3: Smooth the best candidate's score across frames, then
        // classify based on who is the absolute best match.
        val bestScore = bestMatchInFrame?.second ?: 0f
        emaScore = if (emaScore == 0f) bestScore else emaAlpha * bestScore + (1 - emaAlpha) * emaScore

        val results = scoredFaces.map { (det, score) ->
            val isAbsoluteBest = (det == bestMatchInFrame?.first)
            // Only the current best candidate's decision uses the smoothed
            // score -- everyone else is evaluated on their own raw score,
            // but they can never be classified as the streamer anyway since
            // isAbsoluteBest is false for them.
            val effectiveScore = if (isAbsoluteBest) emaScore else score
            val rawMatch = isAbsoluteBest && effectiveScore >= similarityThreshold

            var smoothedMatch = false

            if (rawMatch) {
                // Genuine, confident match: full reset of the hold window.
                consecutiveStreamerFrames = debounceThreshold
                lastStreamerBox = det.box
                smoothedMatch = true
                streamerFoundThisFrame = true
            } else if (consecutiveStreamerFrames > 0 && lastStreamerBox != null) {
                // We recently confirmed this is the streamer. To keep that
                // status for one more frame, this face now needs to be BOTH
                // spatially close to where the streamer just was AND still
                // scoring like the streamer -- not either one alone. This is
                // what stops a nearby bystander from hijacking the rescue.
                val iou = calculateIoU(det.box, lastStreamerBox!!)
                val spatiallyClose = iou > iouRescueThreshold
                val stillLooksLikeThem = isAbsoluteBest && effectiveScore >= keepAliveThreshold

                if (spatiallyClose && stillLooksLikeThem) {
                    lastStreamerBox = det.box
                    smoothedMatch = true
                    streamerFoundThisFrame = true
                    Log.d(
                        "SafeStream_Match",
                        "Rescue used (iou=${"%.2f".format(iou)}, score=${"%.2f".format(score)})"
                    )
                }
            }

            IdentifiedDetection(det, smoothedMatch, score)
        }

        // STEP 4: Decrease debounce counter only if the streamer wasn't found at all in this frame
        if (!streamerFoundThisFrame && consecutiveStreamerFrames > 0) {
            consecutiveStreamerFrames--
        }

        return results
    }

    private fun calculateIoU(box1: RectF, box2: RectF): Float {
        val xA = max(box1.left, box2.left)
        val yA = max(box1.top, box2.top)
        val xB = min(box1.right, box2.right)
        val yB = min(box1.bottom, box2.bottom)

        val intersectionArea = max(0f, xB - xA) * max(0f, yB - yA)
        val box1Area = (box1.right - box1.left) * (box1.bottom - box1.top)
        val box2Area = (box2.right - box2.left) * (box2.bottom - box2.top)

        val iou = intersectionArea / (box1Area + box2Area - intersectionArea)
        return if (iou.isNaN()) 0f else iou
    }
}