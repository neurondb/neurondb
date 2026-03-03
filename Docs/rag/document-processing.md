# Document Processing

Text processing and NLP capabilities.

## Text Processing

Process and clean text:

```sql
-- Clean and normalize text
SELECT process_text(
    'Raw text with   multiple   spaces',
    '{"lowercase": true, "remove_extra_spaces": true}'::jsonb
) AS processed_text;
```

## Chunking

Split documents into chunks:

```sql
-- Chunk text
SELECT chunk_text(
    'long document text...',
    500,  -- chunk size
    50    -- overlap
) AS chunks;
```

## Tokenization

```sql
-- Tokenize text
SELECT tokenize_text('Hello world', 'whitespace') AS tokens;
```

## Learn More

For detailed documentation on document processing, chunking strategies, tokenization, and NLP features, visit:

**[Document Processing Documentation](https://neurondb.ai/docs/ml/text-ml/)**

## Related Topics

- [RAG Overview](overview.md) - RAG pipeline
- [Embedding Generation](../ml-embeddings/embedding-generation.md) - Generate embeddings

