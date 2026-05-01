package com.example.embeddings.model;

import com.example.embeddings.service.NativeBridge;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

class CEmbeddingBatchTest
{
    private static final long FAKE_NATIVE_POINTER = 100L;

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
    void readBuildsBatchFromNativeStruct()
    {
        try (final NativeAllocation struct = NativeAllocation.bytes(CEmbeddingBatch.getStructSize()))
        {
            // Arrange
            NativeMemory.writeLong(struct.ptr(), FAKE_NATIVE_POINTER);
            NativeMemory.writeLong(struct.ptr() + 8, 2L);
            NativeMemory.writeLong(struct.ptr() + 16, 3L);

            // Act
            final CEmbeddingBatch batch = CEmbeddingBatch.read(struct.ptr());

            // Assert
            assertEquals(FAKE_NATIVE_POINTER, batch.dataPtr());
            assertEquals(2L, batch.nVectors());
            assertEquals(3L, batch.dim());
        }
    }

    @Test
    void readAllEmbeddingsConvertsFlatNativeArrayToMatrix()
    {
        try (final NativeAllocation data = nativeFloats(1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f))
        {
            // Arrange
            final CEmbeddingBatch batch = new CEmbeddingBatch(data.ptr(), 2L, 3L);

            // Act
            final float[][] embeddings = batch.readAllEmbeddings();

            // Assert
            assertArrayEquals(new float[]{1.0f, 2.0f, 3.0f}, embeddings[0]);
            assertArrayEquals(new float[]{4.0f, 5.0f, 6.0f}, embeddings[1]);
        }
    }

    @Test
    void readSingleEmbeddingReadsAtVectorOffset()
    {
        try (final NativeAllocation data = nativeFloats(1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f))
        {
            // Arrange
            final CEmbeddingBatch batch = new CEmbeddingBatch(data.ptr(), 2L, 3L);

            // Act & Assert
            assertArrayEquals(new float[]{4.0f, 5.0f, 6.0f}, batch.readSingleEmbedding(1));
        }
    }

    @Test
    void getStructSizeReturnsNativeLayoutSize()
    {
        // Arrange, Act & Assert
        assertEquals(24L, CEmbeddingBatch.getStructSize());
    }

    private record NativeAllocation(long ptr) implements AutoCloseable
    {
        static NativeAllocation bytes(long size)
        {
            return new NativeAllocation(NativeMemory.allocate(size));
        }

        @Override
        public void close()
        {
            NativeMemory.free(ptr);
        }
    }
}
