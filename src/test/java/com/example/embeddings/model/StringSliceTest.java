package com.example.embeddings.model;

import com.example.embeddings.service.NativeBridge;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.mockito.MockedStatic;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.mockito.Mockito.mockStatic;

class StringSliceTest
{
    private static final long FAKE_NATIVE_POINTER = 100L;

    @BeforeAll
    static void loadNativeLibrary()
    {
        NativeBridge.validateBackend("embed_anything");
    }

    @Test
    void fromStringCopiesUtf8BytesToNativeMemory()
    {
        // Arrange & Act
        try (final NativeStringSlice slice = NativeStringSlice.fromString("hello"))
        {
            // Assert
            assertEquals(5L, slice.value().len());
        }
    }

    @Test
    void getStructSizeReturnsNativeLayoutSize()
    {
        // Arrange, Act & Assert
        assertEquals(16L, StringSlice.getStructSize());
    }

    @Test
    void writeToNativeWritesPointerAndLength()
    {
        // Arrange
        final StringSlice slice = new StringSlice(FAKE_NATIVE_POINTER, 5L);

        try (final NativeAllocation dest = NativeAllocation.bytes(StringSlice.getStructSize()))
        {
            // Arrange & Act
            slice.writeToNative(dest.ptr());

            // Assert
            assertEquals(FAKE_NATIVE_POINTER, NativeMemory.readLong(dest.ptr()));
            assertEquals(5L, NativeMemory.readLong(dest.ptr() + 8));
        }
    }

    @Test
    void freeReleasesNonZeroPointer()
    {
        try (MockedStatic<NativeMemory> nativeMemory = mockStatic(NativeMemory.class))
        {
            // Arrange
            final StringSlice slice = new StringSlice(FAKE_NATIVE_POINTER, 1L);

            // Act
            slice.free();

            // Assert
            nativeMemory.verify(() -> NativeMemory.free(FAKE_NATIVE_POINTER));
        }
    }

    @Test
    void freeIgnoresZeroPointer()
    {
        try (MockedStatic<NativeMemory> nativeMemory = mockStatic(NativeMemory.class))
        {
            // Arrange
            final StringSlice slice = new StringSlice(0L, 0L);

            // Act
            slice.free();

            // Assert
            nativeMemory.verifyNoInteractions();
        }
    }

    private record NativeStringSlice(StringSlice value) implements AutoCloseable
    {
        static NativeStringSlice fromString(String text)
        {
            return new NativeStringSlice(StringSlice.fromString(text));
        }

        @Override
        public void close()
        {
            value.free();
        }
    }

    private static final class NativeAllocation implements AutoCloseable
    {
        private long ptr;

        private NativeAllocation(long ptr)
        {
            this.ptr = ptr;
        }

        static NativeAllocation bytes(long size)
        {
            return new NativeAllocation(NativeMemory.allocate(size));
        }

        long ptr()
        {
            return ptr;
        }

        long release()
        {
            final long released = ptr;
            ptr = 0L;
            return released;
        }

        @Override
        public void close()
        {
            NativeMemory.free(ptr);
        }
    }
}
