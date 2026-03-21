package com.example.embeddings.service;

import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;

import java.util.Arrays;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;

@SpringBootTest
class EmbeddingServiceTest
{

    private static final String BACKEND = "embed_anything";
    private static final String MODEL = "sentence-transformers/all-MiniLM-L6-v2";

    @Autowired
    private EmbeddingService embeddingService;

    @Test
    void testEmbedWithValidInputs()
    {
        // Arrange
        final List<String> texts = Arrays.asList("Spring Boot testing", "Integration test example");

        // Act
        final float[][] result = embeddingService.embed(BACKEND, MODEL, texts);

        // Assert
        assertNotNull(result);
        assertEquals(2, result.length);
        assertEquals(384, result[0].length);
        assertEquals(384, result[1].length);
    }
}
