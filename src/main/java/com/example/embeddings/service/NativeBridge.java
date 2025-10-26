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
     * Validate an embedding method
     *
     * @param method The method name ("fastembed" or "remote")
     * @return Method ID if valid, negative value if invalid
     */
    public static native int validateEmbeddingMethod(String method);

    /**
     * Validate an embedding model for a given method
     *
     * @param methodId The method ID returned by validateEmbeddingMethod
     * @param model    The model name/identifier
     * @return Model ID if valid, negative value if invalid
     */
    public static native int validateEmbeddingModel(int methodId, String model);

    /**
     * Generate embeddings from texts
     *
     * @param methodId    The method ID
     * @param modelId     The model ID
     * @param inputsPtr   Pointer to StringSlice array
     * @param nInputs     Number of input texts
     * @param outBatchPtr Pointer to EmbeddingBatch struct
     * @return 0 on success, non-zero error code on failure
     */
    public static native int generateEmbeddingsFromTexts(
            int methodId,
            int modelId,
            long inputsPtr,
            int nInputs,
            long outBatchPtr
    );

    /**
     * Free an embedding batch
     *
     * @param batchPtr Pointer to EmbeddingBatch struct
     */
    public static native void freeEmbeddingBatch(long batchPtr);
}
