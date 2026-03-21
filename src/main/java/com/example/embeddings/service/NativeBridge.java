package com.example.embeddings.service;

/**
 * Bridge to native Rust embedding functions
 */
public class NativeBridge
{
    static
    {
        System.loadLibrary("gembed_jni");
    }

    /**
     * Validate an backend by name
     *
     * @param backend The backend name (e.g., "embed_anything", "fastembed")
     * @return Backend ID (>= 0) if valid, negative value if invalid
     */
    public static native int validateBackend(String backend);

    /**
     * Validate an embedding model for a given backend and input type
     * <p>
     * Note: The native function requires an input_type parameter.
     * This Java wrapper validates for INPUT_TYPE_TEXT (0) by default.
     *
     * @param backendId The backend ID returned by validateBackend
     * @param model      The model name/identifier
     * @return Model ID (>= 0) if valid, negative value if invalid
     */
    public static native int validateModel(int backendId, String model);

    /**
     * Generate embeddings from text inputs
     *
     * @param backendId  The backend ID
     * @param modelId     The model ID
     * @param inputsPtr   Pointer to StringSlice array
     * @param nInputs     Number of input texts
     * @param outBatchPtr Pointer to EmbeddingBatch struct (must be pre-allocated)
     * @return 0 on success, non-zero error code on failure
     */
    public static native int generateEmbeddingsFromTexts(
            int backendId,
            int modelId,
            long inputsPtr,
            int nInputs,
            long outBatchPtr
    );

    /**
     * Free an embedding batch allocated by the Rust library
     * <p>
     * This must be called to clean up the float array allocated by Rust.
     * The EmbeddingBatch struct itself is allocated by Java and freed separately.
     *
     * @param batchPtr Pointer to EmbeddingBatch struct
     */
    public static native void freeEmbeddingBatch(long batchPtr);
}
