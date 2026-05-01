package com.example.embeddings.controller;

import com.example.embeddings.service.EmbeddingService;
import org.junit.jupiter.api.Test;

import java.lang.reflect.Constructor;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.mockito.Mockito.*;

class EmbeddingControllerTest
{
    private static EmbeddingController newController(EmbeddingService embeddingService) throws Exception
    {
        final Constructor<EmbeddingController> constructor =
                EmbeddingController.class.getDeclaredConstructor(EmbeddingService.class);
        constructor.setAccessible(true);
        return constructor.newInstance(embeddingService);
    }

    @Test
    void generateEmbeddingsDelegatesToService() throws Exception
    {
        // Arrange
        final EmbeddingService embeddingService = mock(EmbeddingService.class);
        final EmbeddingController controller = newController(embeddingService);
        final List<String> texts = List.of("hello", "world");
        final float[][] embeddings = new float[][]{{1.0f, 2.0f}, {3.0f, 4.0f}};
        final EmbeddingController.EmbeddingRequest request =
                new EmbeddingController.EmbeddingRequest("embed_anything", "model", texts);

        when(embeddingService.embed("embed_anything", "model", texts)).thenReturn(embeddings);

        // Act
        final Map<String, Object> response = controller.generateEmbeddings(request);

        // Assert
        assertEquals("embed_anything", response.get("backend"));
        assertEquals("model", response.get("model"));
        assertArrayEquals(embeddings, (float[][]) response.get("embeddings"));
        verify(embeddingService).embed("embed_anything", "model", texts);
    }
}
