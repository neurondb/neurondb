# Complete RAG Support

End-to-end Retrieval Augmented Generation pipeline.

## Basic RAG

Complete RAG workflow:

```sql
-- RAG query
SELECT rag_query(
    'What is machine learning?',     -- query
    'documents',                     -- source table
    'content',                       -- text column
    'embedding',                     -- embedding column
    5,                               -- top K retrieved
    'gpt-4'                          -- LLM model
) AS answer;
```

## RAG with Context

Add additional context:

```sql
-- RAG with custom context
SELECT rag_query_with_context(
    'query text',
    'documents',
    'content',
    'embedding',
    5,
    'gpt-4',
    '{"system_prompt": "You are a helpful assistant"}'::jsonb
) AS answer;
```

## Learn More

For detailed documentation on RAG pipelines, prompt engineering, context management, and evaluation, visit:

**[RAG Documentation](https://neurondb.ai/docs/rag/)**

## Related Topics

- [LLM Integration](llm-integration.md) - LLM providers
- [Document Processing](document-processing.md) - Text processing

