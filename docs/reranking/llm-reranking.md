# LLM Reranking

GPT/Claude-powered scoring for reranking.

## LLM Reranking

Rerank using LLM:

```sql
-- LLM reranking
SELECT idx, score FROM rerank_llm(
    'What is artificial intelligence?',     -- query
    ARRAY[                                  -- candidates
        'AI definition text',
        'ML explanation text'
    ],
    'gpt-4',                                -- model
    3                                       -- top K
);
```

## Learn More

For detailed documentation on LLM reranking, model configuration, cost optimization, and prompt engineering, visit:

**[LLM Reranking Documentation](https://neurondb.ai/docs/reranking/llm/)**

## Related Topics

- [Cross-Encoder](cross-encoder.md) - Neural reranking
- [Ensemble](ensemble.md) - Combine rerankers

