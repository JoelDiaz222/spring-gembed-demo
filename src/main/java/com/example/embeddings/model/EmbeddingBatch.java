package com.example.embeddings.model;

import com.example.embeddings.service.NativeBridge;

/**
 * Java wrapper for generated embeddings with automatic resource management
 */
public class EmbeddingBatch implements AutoCloseable
{
    private final long batchPtr;
    private final long dataPtr;
    private final int nVectors;
    private final int dim;
    private boolean closed = false;

    public EmbeddingBatch(long batchPtr, long dataPtr, int nVectors, int dim)
    {
        this.batchPtr = batchPtr;
        this.dataPtr = dataPtr;
        this.nVectors = nVectors;
        this.dim = dim;
    }

    public int getNumVectors()
    {
        return nVectors;
    }

    public int getDimension()
    {
        return dim;
    }

    /**
     * Get all embeddings as a 2D array [vector_index][dimension]
     */
    public float[][] getAllEmbeddings()
    {
        checkNotClosed();
        final CEmbeddingBatch cBatch = new CEmbeddingBatch(dataPtr, nVectors, dim);
        return cBatch.readAllEmbeddings();
    }

    /**
     * Get a single embedding vector
     */
    public float[] getEmbedding(int index)
    {
        checkNotClosed();
        if (index < 0 || index >= nVectors)
        {
            throw new IndexOutOfBoundsException("Index: " + index + ", Size: " + nVectors);
        }
        final CEmbeddingBatch cBatch = new CEmbeddingBatch(dataPtr, nVectors, dim);
        return cBatch.readSingleEmbedding(index);
    }

    private void checkNotClosed()
    {
        if (closed)
        {
            throw new IllegalStateException("EmbeddingBatch has been closed");
        }
    }

    @Override
    public void close()
    {
        if (!closed)
        {
            NativeBridge.freeEmbeddingBatch(batchPtr);
            closed = true;
        }
    }
}
