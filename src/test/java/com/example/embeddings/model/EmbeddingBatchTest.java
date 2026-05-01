package com.example.embeddings.model;

import com.example.embeddings.service.NativeBridge;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import java.lang.reflect.Field;

import static org.junit.jupiter.api.Assertions.*;

class EmbeddingBatchTest
{
    private static final long SAFE_NULL_BATCH_POINTER = 0L;
    private static final long FAKE_DATA_POINTER = 100L;

    @BeforeAll
    static void loadNativeLibrary()
    {
        NativeBridge.validateBackend("embed_anything");
    }

    private static NativeAllocation nativeFloats(float... values)
    {
        final byte[] bytes = new byte[values.length * Float.BYTES];
        final java.nio.ByteBuffer buffer = java.nio.ByteBuffer.wrap(bytes)
                .order(java.nio.ByteOrder.nativeOrder());
        for (float value : values)
        {
            buffer.putFloat(value);
        }

        final long ptr = NativeMemory.allocate(bytes.length);
        NativeMemory.copyToNative(ptr, bytes);
        return new NativeAllocation(ptr);
    }

    @Test
    void gettersReturnBatchMetadata()
    {
        // Arrange
        try (final EmbeddingBatch batch = new EmbeddingBatch(SAFE_NULL_BATCH_POINTER, FAKE_DATA_POINTER, 2, 3))
        {

            // Act & Assert
            assertEquals(2, batch.getNumVectors());
            assertEquals(3, batch.getDimension());
        }
    }

    @Test
    void getAllEmbeddingsReadsData()
    {
        try (final NativeAllocation data = nativeFloats(1.0f, 2.0f, 3.0f, 4.0f))
        {
            // Arrange
            try (final EmbeddingBatch batch = new EmbeddingBatch(SAFE_NULL_BATCH_POINTER, data.ptr(), 2, 2))
            {

                // Act
                final float[][] embeddings = batch.getAllEmbeddings();

                // Assert
                assertArrayEquals(new float[]{1.0f, 2.0f}, embeddings[0]);
                assertArrayEquals(new float[]{3.0f, 4.0f}, embeddings[1]);
            }
        }
    }

    @Test
    void getEmbeddingReadsSingleVector()
    {
        try (final NativeAllocation data = nativeFloats(1.0f, 2.0f, 3.0f, 4.0f))
        {
            // Arrange
            try (final EmbeddingBatch batch = new EmbeddingBatch(SAFE_NULL_BATCH_POINTER, data.ptr(), 2, 2))
            {

                // Act & Assert
                assertArrayEquals(new float[]{3.0f, 4.0f}, batch.getEmbedding(1));
            }
        }
    }

    @Test
    void getEmbeddingRejectsInvalidIndex()
    {
        // Arrange
        try (final EmbeddingBatch batch = new EmbeddingBatch(SAFE_NULL_BATCH_POINTER, FAKE_DATA_POINTER, 2, 2))
        {

            // Act & Assert
            assertThrows(IndexOutOfBoundsException.class, () -> batch.getEmbedding(-1));
            assertThrows(IndexOutOfBoundsException.class, () -> batch.getEmbedding(2));
        }
    }

    @Test
    void closedBatchRejectsReads() throws Exception
    {
        // Arrange
        try (final EmbeddingBatch batch = new EmbeddingBatch(SAFE_NULL_BATCH_POINTER, FAKE_DATA_POINTER, 2, 2))
        {
            final Field closed = EmbeddingBatch.class.getDeclaredField("closed");
            closed.setAccessible(true);
            closed.set(batch, true);

            // Act & Assert
            assertThrows(IllegalStateException.class, batch::getAllEmbeddings);
            assertThrows(IllegalStateException.class, () -> batch.getEmbedding(0));
        }
    }

    private record NativeAllocation(long ptr) implements AutoCloseable
    {
        @Override
        public void close()
        {
            NativeMemory.free(ptr);
        }
    }
}
