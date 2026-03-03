# Multi-Vector

Use multiple embeddings per document for enhanced retrieval.

## Store Multiple Embeddings

```sql
-- Create table with multiple embeddings
CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
    content TEXT,
    title_embedding vector(384),
    content_embedding vector(384),
    summary_embedding vector(384)
);
```

## Multi-Vector Search

Search across multiple embeddings:

```sql
-- Search with multiple vectors
SELECT id, content,
       multi_vector_search(
           embed_text('query'),
           ARRAY[
               title_embedding,
               content_embedding,
               summary_embedding
           ],
           ARRAY[0.2, 0.6, 0.2]  -- weights per embedding
       ) AS combined_score
FROM documents
ORDER BY combined_score DESC
LIMIT 10;
```

## Learn More

For detailed documentation on multi-vector strategies, embedding selection, weight optimization, and performance tuning, visit:

**[Multi-Vector Documentation](https://neurondb.ai/docs/hybrid/multi-vector/)**

## Related Topics

- [Hybrid Search](overview.md) - Combine multiple search types
- [Vector Search](../vector-search/indexing.md) - Vector similarity

