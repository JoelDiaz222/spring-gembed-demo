package com.example.embeddings.controller;

import com.example.embeddings.service.EmbeddingService;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.util.List;
import java.util.Map;

@RestController
@RequestMapping("/")
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
     * "method": "fastembed",
     * "model": "example-model",
     * "texts": ["Hello world", "Another text"]
     * }
     */
    @PostMapping
    public Map<String, Object> generateEmbeddings(@RequestBody EmbeddingRequest request)
    {
        float[][] embeddings = embeddingService.embed(request.method(), request.model(), request.texts());

        return Map.of(
                "method", request.method(),
                "model", request.model(),
                "embeddings", embeddings
        );
    }

    // Using a record for the request body
    public record EmbeddingRequest(String method, String model, List<String> texts) {}
}
