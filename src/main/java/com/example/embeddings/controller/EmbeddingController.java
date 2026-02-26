package com.example.embeddings.controller;

import com.example.embeddings.service.EmbeddingService;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.util.List;
import java.util.Map;

@RestController
@RequestMapping("/embed")
public class EmbeddingController
{
    private final EmbeddingService embeddingService;

    private EmbeddingController(EmbeddingService embeddingService)
    {
        this.embeddingService = embeddingService;
    }

    /**
     * Generate embeddings for a batch of texts
     * <p>
     * Example POST request:
     * POST /embed
     * {
     * "embedder": "embed_anything",
     * "model": "sentence-transformers/all-MiniLM-L6-v2",
     * "texts": ["Hello world"]
     * }
     */
    @PostMapping
    public Map<String, Object> generateEmbeddings(@RequestBody EmbeddingRequest request)
    {
        float[][] embeddings = embeddingService.embed(request.embedder(), request.model(), request.texts());

        return Map.of(
                "embedder", request.embedder(),
                "model", request.model(),
                "embeddings", embeddings
        );
    }

    // Using a record for the request body
    public record EmbeddingRequest(String embedder, String model, List<String> texts) {}
}
