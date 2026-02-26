package com.example.embeddings.service;

import com.example.embeddings.model.EmbeddingBatch;
import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class EmbeddingGeneratorTest
{

    private static final String EMBEDDER = "embed_anything";
    private static final String MODEL = "sentence-transformers/all-MiniLM-L6-v2";

    @Test
    void testEmbeddingGeneratorValidEmbedderAndModel()
    {
        // Arrange & Act
        final EmbeddingGenerator generator = new EmbeddingGenerator(EMBEDDER, MODEL);

        // Assert
        assertEquals(EMBEDDER, generator.getEmbedder());
        assertEquals(MODEL, generator.getModel());
    }

    @Test
    void testEmbeddingGeneratorInvalidEmbedder()
    {
        // Arrange, Act & Assert
        assertThrows(
                IllegalArgumentException.class, () ->
                        new EmbeddingGenerator("invalid_method", MODEL)
        );
    }

    @Test
    void testEmbeddingGeneratorInvalidModel()
    {
        // Arrange, Act & Assert
        assertThrows(
                IllegalArgumentException.class, () ->
                        new EmbeddingGenerator(EMBEDDER, "invalid_model_abc")
        );
    }

    @Test
    void testGenerateEmbeddingsSingle()
    {
        // Arrange
        final EmbeddingGenerator generator = new EmbeddingGenerator(EMBEDDER, MODEL);
        final String text = "Hello world";

        // Act
        final float[] embedding = generator.generateEmbedding(text);

        // Assert
        assertNotNull(embedding);
        assertEquals(384, embedding.length); // sentence-transformers/all-MiniLM-L6-v2 has 384 dimensions
    }

    @Test
    void testGenerateEmbeddingsBatch()
    {
        // Arrange
        final EmbeddingGenerator generator = new EmbeddingGenerator(EMBEDDER, MODEL);
        final List<String> texts = Arrays.asList("Hello world", "Another test string", "And a third one");

        // Act
        try (final EmbeddingBatch batch = generator.generateEmbeddings(texts))
        {
            // Assert
            assertEquals(3, batch.getNumVectors());
            assertEquals(384, batch.getDimension());

            final float[][] embeddings = batch.getAllEmbeddings();
            assertEquals(3, embeddings.length);
            for (final float[] emb : embeddings)
            {
                assertEquals(384, emb.length);
            }
        }
    }

    @Test
    void testGenerateEmbeddingsEmptyList()
    {
        // Arrange
        final EmbeddingGenerator generator = new EmbeddingGenerator(EMBEDDER, MODEL);

        // Act & Assert
        assertThrows(
                IllegalArgumentException.class, () ->
                        generator.generateEmbeddings(List.of())
        );

        assertThrows(
                IllegalArgumentException.class, () ->
                        generator.generateEmbeddings(null)
        );
    }
}
