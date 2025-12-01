# Cross-Encoder Reranking

Neural reranking models for improved relevance.

## Rerank Results

Rerank search results with cross-encoder:

```sql
-- Rerank search results
SELECT idx, score FROM rerank_cross_encoder(
    'What is machine learning?',           -- query
    ARRAY[                                  -- candidate texts
        'Neural networks tutorial',
        'Deep learning basics',
        'AI history'
    ],
    'ms-marco-MiniLM-L-6-v2',              -- model name
    3                                       -- top K
);
```

## Batch Reranking

Rerank multiple queries:

```sql
-- Batch reranking
SELECT query_id, idx, score
FROM rerank_cross_encoder_batch(
    ARRAY['query 1', 'query 2'],
    ARRAY[
        ARRAY['doc 1', 'doc 2'],
        ARRAY['doc 3', 'doc 4']
    ],
    'model_name',
    5
);
```

## Learn More

For detailed documentation on cross-encoder models, model selection, fine-tuning, and performance optimization, visit:

**[Cross-Encoder Documentation](https://neurondb.ai/docs/reranking/cross-encoder/)**

## Related Topics

- [LLM Reranking](llm-reranking.md) - LLM-powered reranking
- [Ensemble](ensemble.md) - Combine reranking strategies

