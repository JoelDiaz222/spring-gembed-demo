package com.example.embeddings.model;

import java.nio.charset.StandardCharsets;

/**
 * Represents the C StringSlice struct
 * Layout: { const char* ptr, size_t len }
 *
 * @param ptr const char*
 * @param len size_t
 */
public record StringSlice(long ptr, long len)
{
    private static final long STRUCT_SIZE = 16; // 2 * 8 bytes on 64-bit systems

    /**
     * Allocate native memory and create a StringSlice from a Java String
     */
    public static StringSlice fromString(String text)
    {
        final byte[] utf8Bytes = text.getBytes(StandardCharsets.UTF_8);
        final long textPtr = NativeMemory.allocate(utf8Bytes.length);
        NativeMemory.copyToNative(textPtr, utf8Bytes);
        return new StringSlice(textPtr, utf8Bytes.length);
    }

    /**
     * Get the size of StringSlice struct in bytes
     */
    public static long getStructSize()
    {
        return STRUCT_SIZE;
    }

    /**
     * Write this StringSlice to native memory at the given address
     */
    public void writeToNative(long destPtr)
    {
        NativeMemory.writeLong(destPtr, ptr);           // ptr field
        NativeMemory.writeLong(destPtr + 8, len);       // len field
    }

    /**
     * Free the native memory allocated for the text data
     */
    public void free()
    {
        if (ptr != 0)
        {
            NativeMemory.free(ptr);
        }
    }
}
