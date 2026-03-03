# Ensemble Reranking

Combine multiple reranking strategies for best results.

## Ensemble Reranking

Combine multiple rerankers:

```sql
-- Ensemble reranking
SELECT idx, score FROM rerank_ensemble(
    'query text',
    ARRAY['doc 1', 'doc 2'],
    ARRAY[                    -- reranker configs
        '{"type": "cross_encoder", "weight": 0.6}',
        '{"type": "llm", "weight": 0.4}'
    ]::jsonb[],
    5
);
```

## Learn More

For detailed documentation on ensemble strategies, weight optimization, and combining rerankers, visit:

**[Ensemble Reranking Documentation](https://neurondb.ai/docs/reranking/ensemble/)**

## Related Topics

- [Cross-Encoder](cross-encoder.md) - Neural reranking
- [LLM Reranking](llm-reranking.md) - LLM-powered reranking

