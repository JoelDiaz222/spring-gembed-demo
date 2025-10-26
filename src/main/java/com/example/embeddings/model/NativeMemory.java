package com.example.embeddings.model;

import java.util.ArrayList;
import java.util.List;

/**
 * Manages native memory allocation and deallocation for embedding operations
 */
public class NativeMemory implements AutoCloseable
{
    private final long stringSlicesPtr;
    private final List<StringSlice> stringSlices;

    private NativeMemory(long stringSlicesPtr, List<StringSlice> stringSlices)
    {
        this.stringSlicesPtr = stringSlicesPtr;
        this.stringSlices = stringSlices;
    }

    /**
     * Allocate native memory for a list of texts
     */
    public static NativeMemory fromTexts(List<String> texts)
    {
        final List<StringSlice> slices = new ArrayList<>(texts.size());

        // Allocate array of StringSlice structs
        final long slicesArrayPtr = allocate(texts.size() * StringSlice.getStructSize());
        long currentSlicePtr = slicesArrayPtr;

        try
        {
            for (String text : texts)
            {
                final StringSlice slice = StringSlice.fromString(text);
                slices.add(slice);
                slice.writeToNative(currentSlicePtr);
                currentSlicePtr += StringSlice.getStructSize();
            }

            return new NativeMemory(slicesArrayPtr, slices);
        }
        catch (Exception e)
        {
            // Clean up on failure
            free(slicesArrayPtr);
            for (StringSlice slice : slices)
            {
                slice.free();
            }
            throw e;
        }
    }

    /**
     * Allocate memory for an EmbeddingBatch struct
     */
    public static long allocateEmbeddingBatch()
    {
        return allocate(CEmbeddingBatch.getStructSize());
    }

    // Native method wrappers
    public static long allocate(long size)
    {
        return allocateMemory(size);
    }

    public static void free(long ptr)
    {
        if (ptr != 0)
        {
            freeNativeMemory(ptr);
        }
    }

    // Native JNI methods
    private static native long allocateMemory(long size);

    private static native void freeNativeMemory(long ptr);

    static native void copyToNative(long destPtr, byte[] src);

    static native void writeLong(long ptr, long value);

    static native long readLong(long ptr);

    static native float[] readFloatArray(long ptr, long count);

    public long getStringSlicesPtr()
    {
        return stringSlicesPtr;
    }

    @Override
    public void close()
    {
        if (stringSlicesPtr != 0)
        {
            free(stringSlicesPtr);
        }
        for (StringSlice slice : stringSlices)
        {
            slice.free();
        }
    }
}
