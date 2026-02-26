package com.example.embeddings.service;

import com.example.embeddings.model.EmbeddingBatch;
import org.springframework.stereotype.Service;

import java.util.List;

@Service
public class EmbeddingService
{
    public float[][] embed(String embedder, String model, List<String> texts)
    {
        final EmbeddingGenerator generator = new EmbeddingGenerator(embedder, model);
        try (final EmbeddingBatch batch = generator.generateEmbeddings(texts))
        {
            return batch.getAllEmbeddings();
        }
    }
}
