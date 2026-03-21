package com.example.embeddings.service;

import com.example.embeddings.model.CEmbeddingBatch;
import com.example.embeddings.model.EmbeddingBatch;
import com.example.embeddings.model.NativeMemory;

import java.util.List;

/**
 * Java interface to the Rust embedding generation library.
 */
public class EmbeddingGenerator
{
    private final String backend;
    private final String model;
    private final int backendId;
    private final int modelId;

    /**
     * Create an embedding generator with specified backend and model
     *
     * @param backend The backend name (e.g., "embed_anything", "fastembed")
     * @param model    The model name/identifier
     * @throws IllegalArgumentException if backend or model is invalid
     */
    public EmbeddingGenerator(String backend, String model)
    {
        this.backend = backend;
        this.model = model;

        this.backendId = NativeBridge.validateBackend(backend);
        if (this.backendId < 0)
        {
            throw new IllegalArgumentException("Invalid backend: " + backend);
        }

        this.modelId = NativeBridge.validateModel(this.backendId, model);
        if (this.modelId < 0)
        {
            throw new IllegalArgumentException("Model not allowed: " + model);
        }
    }

    /**
     * Generate embeddings for a list of texts
     *
     * @param texts List of text strings to embed
     * @return EmbeddingBatch containing the generated embeddings
     * @throws RuntimeException if embedding generation fails
     */
    public EmbeddingBatch generateEmbeddings(List<String> texts)
    {
        if (texts == null || texts.isEmpty())
        {
            throw new IllegalArgumentException("Input texts cannot be null or empty");
        }

        final int nInputs = texts.size();

        // Allocate native memory for StringSlice array and text data
        try (NativeMemory memory = NativeMemory.fromTexts(texts))
        {
            // Allocate memory for output batch
            final long outBatchPtr = NativeMemory.allocateEmbeddingBatch();

            try
            {
                // Call native function
                final int result = NativeBridge.generateEmbeddingsFromTexts(
                        backendId,
                        modelId,
                        memory.getStringSlicesPtr(),
                        nInputs,
                        outBatchPtr
                );

                if (result != 0)
                {
                    throw new RuntimeException("Embedding generation failed with code: " + result);
                }

                // Read the populated batch structure
                final CEmbeddingBatch cBatch = CEmbeddingBatch.read(outBatchPtr);

                return new EmbeddingBatch(
                        outBatchPtr,
                        cBatch.dataPtr(),
                        (int) cBatch.nVectors(),
                        (int) cBatch.dim()
                );
            }
            catch (Exception e)
            {
                NativeMemory.free(outBatchPtr);
                throw e;
            }

        }
    }

    /**
     * Generate a single embedding for one text
     */
    public float[] generateEmbedding(String text)
    {
        try (EmbeddingBatch batch = generateEmbeddings(List.of(text)))
        {
            return batch.getEmbedding(0);
        }
    }

    public String getBackend()
    {
        return backend;
    }

    public String getModel()
    {
        return model;
    }
}
