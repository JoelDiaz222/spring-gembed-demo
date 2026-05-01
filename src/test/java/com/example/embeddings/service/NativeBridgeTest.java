package com.example.embeddings.service;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertNotNull;

class NativeBridgeTest
{
    @Test
    void canInstantiateBridgeClass()
    {
        // Arrange, Act & Assert
        assertNotNull(new NativeBridge());
    }
}
