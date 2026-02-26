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
     * Validate an embedder by name
     *
     * @param embedder The embedder name (e.g., "embed_anything", "fastembed")
     * @return Embedder ID (>= 0) if valid, negative value if invalid
     */
    public static native int validateEmbedder(String embedder);

    /**
     * Validate an embedding model for a given embedder and input type
     * <p>
     * Note: The native function requires an input_type parameter.
     * This Java wrapper validates for INPUT_TYPE_TEXT (0) by default.
     *
     * @param embedderId The embedder ID returned by validateEmbedder
     * @param model      The model name/identifier
     * @return Model ID (>= 0) if valid, negative value if invalid
     */
    public static native int validateEmbeddingModel(int embedderId, String model);

    /**
     * Generate embeddings from text inputs
     *
     * @param embedderId  The embedder ID
     * @param modelId     The model ID
     * @param inputsPtr   Pointer to StringSlice array
     * @param nInputs     Number of input texts
     * @param outBatchPtr Pointer to EmbeddingBatch struct (must be pre-allocated)
     * @return 0 on success, non-zero error code on failure
     */
    public static native int generateEmbeddingsFromTexts(
            int embedderId,
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
