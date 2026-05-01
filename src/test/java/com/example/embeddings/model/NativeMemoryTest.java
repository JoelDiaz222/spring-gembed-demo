package com.example.embeddings.model;

import com.example.embeddings.service.NativeBridge;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class NativeMemoryTest
{
    @BeforeAll
    static void loadNativeLibrary()
    {
        NativeBridge.validateBackend("embed_anything");
    }

    @Test
    void fromTextsAllocatesSliceArrayAndFreesOnClose()
    {
        // Arrange & Act
        try (final NativeMemory memory = NativeMemory.fromTexts(List.of("one", "two")))
        {
            // Assert
            assertNotEquals(0L, memory.getStringSlicesPtr());
        }
    }

    @Test
    void fromTextsCleansUpWhenTextIsNull()
    {
        // Arrange, Act & Assert
        assertThrows(
                NullPointerException.class, () ->
                {
                    try (final NativeMemory ignored = NativeMemory.fromTexts(java.util.Arrays.asList("one", null)))
                    {
                    }
                }
        );
    }

    @Test
    void allocateEmbeddingBatchUsesStructSize()
    {
        // Arrange & Act
        try (final NativeAllocation batch = NativeAllocation.embeddingBatch())
        {
            // Assert
            assertNotEquals(0L, batch.ptr());
        }
    }

    @Test
    void freeIgnoresZeroPointer()
    {
        // Arrange, Act & Assert
        NativeMemory.free(0L);
    }

    private record NativeAllocation(long ptr) implements AutoCloseable
    {
        static NativeAllocation embeddingBatch()
        {
            return new NativeAllocation(NativeMemory.allocateEmbeddingBatch());
        }

        @Override
        public void close()
        {
            NativeMemory.free(ptr);
        }
    }
}
