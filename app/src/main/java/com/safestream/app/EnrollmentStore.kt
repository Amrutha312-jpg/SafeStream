package com.safestream.app

import android.content.Context

/**
 * Persists the streamer's enrolled face embeddings locally (SharedPreferences).
 * UPDATED: now stores MULTIPLE samples per streamer (matching the "Capture
 * Multiple Samples" step from the enrollment design, and IdentityMatcher's
 * loadAll() usage) instead of a single overwritten embedding.
 *
 * Nothing here ever leaves the device — consistent with the Zero-Cloud
 * Leakage principle from the rest of the project.
 */
class EnrollmentStore(context: Context) {

    private val prefs = context.getSharedPreferences("safestream_enrollment", Context.MODE_PRIVATE)

    /** Adds one more sample without erasing previously stored ones. */
    fun addSample(embedding: FloatArray) {
        val existing = loadAllRaw().toMutableList()
        existing.add(serialize(embedding))
        prefs.edit().putString(KEY_EMBEDDINGS, existing.joinToString(SAMPLE_SEPARATOR)).apply()
    }

    /** Replaces ALL stored samples with a single one. Kept for compatibility
     * with any code still calling the old single-sample save(). */
    fun save(embedding: FloatArray) {
        prefs.edit().putString(KEY_EMBEDDINGS, serialize(embedding)).apply()
    }

    /** Returns every stored embedding sample — this is what IdentityMatcher calls. */
    fun loadAll(): List<FloatArray> = loadAllRaw().mapNotNull { deserialize(it) }

    /** Backward-compatible single-sample accessor: returns the first stored sample. */
    fun load(): FloatArray? = loadAll().firstOrNull()

    fun isEnrolled(): Boolean = loadAllRaw().isNotEmpty()

    fun sampleCount(): Int = loadAllRaw().size

    /** Wipes all stored samples — call at the START of a fresh enrollment session
     * so old samples from a previous person/session don't linger and get
     * matched against. */
    fun clear() {
        prefs.edit().remove(KEY_EMBEDDINGS).apply()
    }

    private fun loadAllRaw(): List<String> {
        val stored = prefs.getString(KEY_EMBEDDINGS, null) ?: return emptyList()
        return stored.split(SAMPLE_SEPARATOR).filter { it.isNotBlank() }
    }

    private fun serialize(embedding: FloatArray): String = embedding.joinToString(",")

    private fun deserialize(raw: String): FloatArray? {
        return try {
            raw.split(",").map { it.toFloat() }.toFloatArray()
        } catch (e: Exception) {
            null
        }
    }

    companion object {
        private const val KEY_EMBEDDINGS = "streamer_embeddings"
        private const val SAMPLE_SEPARATOR = ";"
    }
}
