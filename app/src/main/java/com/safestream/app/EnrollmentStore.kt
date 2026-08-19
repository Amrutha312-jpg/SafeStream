package com.safestream.app

import android.content.Context
import com.google.gson.Gson
import com.google.gson.reflect.TypeToken

class EnrollmentStore(context: Context) {
    private val prefs = context.getSharedPreferences("safestream_prefs", Context.MODE_PRIVATE)
    private val gson = Gson()

    fun save(embedding: FloatArray) {
        val currentList = loadAll().toMutableList()
        currentList.add(embedding)

        // Allow up to 30 embeddings for a full 5-second capture
        while (currentList.size > 30) {
            currentList.removeAt(0)
        }

        // Convert FloatArrays to List<List<Float>> for proper Gson serialization
        val listToSave = currentList.map { it.toList() }
        val json = gson.toJson(listToSave)
        prefs.edit().putString("enrolled_embeddings", json).commit()
    }

    fun load(): FloatArray? {
        return loadAll().lastOrNull()
    }

    fun loadAll(): List<FloatArray> {
        val json = prefs.getString("enrolled_embeddings", null) ?: return emptyList()
        return try {
            val type = object : TypeToken<List<List<Float>>>() {}.type
            val savedLists: List<List<Float>> = gson.fromJson(json, type) ?: emptyList()
            // Convert back to FloatArray
            savedLists.map { floatList -> floatList.toFloatArray() }
        } catch (e: Exception) {
            emptyList()
        }
    }

    fun clear() {
        prefs.edit().remove("enrolled_embeddings").commit()
    }
}