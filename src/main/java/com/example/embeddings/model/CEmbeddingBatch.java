package com.example.embeddings.model;

/**
 * Represents the C EmbeddingBatch struct
 * Layout: { float* data, size_t n_vectors, size_t dim }
 *
 * @param dataPtr  float*
 * @param nVectors size_t
 * @param dim      size_t
 */
public record CEmbeddingBatch(long dataPtr, long nVectors, long dim)
{
    private static final long STRUCT_SIZE = 24; // 3 * 8 bytes on 64-bit systems

    /**
     * Read a CEmbeddingBatch from native memory at the given address
     */
    public static CEmbeddingBatch read(long ptr)
    {
        final long dataPtr = NativeMemory.readLong(ptr);
        final long nVectors = NativeMemory.readLong(ptr + 8);
        final long dim = NativeMemory.readLong(ptr + 16);
        return new CEmbeddingBatch(dataPtr, nVectors, dim);
    }

    /**
     * Get the size of EmbeddingBatch struct in bytes
     */
    public static long getStructSize()
    {
        return STRUCT_SIZE;
    }

    /**
     * Read all embeddings as a 2D array [vector_index][dimension]
     */
    public float[][] readAllEmbeddings()
    {
        final int nVec = (int) nVectors;
        final int dimension = (int) dim;
        final float[][] result = new float[nVec][dimension];
        final long totalFloats = nVectors * dim;
        final float[] flatData = NativeMemory.readFloatArray(dataPtr, totalFloats);

        for (int i = 0; i < nVec; i++)
        {
            System.arraycopy(flatData, i * dimension, result[i], 0, dimension);
        }

        return result;
    }

    /**
     * Read a single embedding vector at the given index
     */
    public float[] readSingleEmbedding(int index)
    {
        final int dimension = (int) dim;
        final long offset = (long) index * dimension * 4; // 4 bytes per float
        return NativeMemory.readFloatArray(dataPtr + offset, dimension);
    }
}
