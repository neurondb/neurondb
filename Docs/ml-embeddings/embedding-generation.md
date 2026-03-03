# Embedding Generation

Generate embeddings from text, images, and multimodal data with intelligent caching.

## Text Embeddings

Generate embeddings from text:

```sql
SELECT embed_text('Hello world') AS embedding;
```

Specify model:

```sql
SELECT embed_text('Hello world', 'all-MiniLM-L6-v2') AS embedding;
```

## Batch Generation

Generate embeddings in batches:

```sql
SELECT embed_text_batch(ARRAY['First text', 'Second text']) AS embeddings;
```

## Caching

Embeddings are automatically cached to improve performance:

```sql
SELECT * FROM neurondb.embedding_cache_stats;
```

## Learn More

For detailed documentation on embedding models, providers, caching strategies, and multimodal embeddings, visit:

**[Embedding Generation Documentation](https://neurondb.ai/docs/ml/embeddings/)**

## Related Topics

- [Model Management](model-management.md) - Manage embedding models
- [Vector Search](../vector-search/indexing.md) - Index and search embeddings
