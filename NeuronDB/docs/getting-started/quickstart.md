# Quick Start Guide

Get up and running with NeuronDB quickly.

## Prerequisites

- PostgreSQL 16, 17, or 18 installed
- NeuronDB extension installed (see [Installation Guide](installation.md))

## Step 1: Create Extension

```sql
CREATE EXTENSION neurondb;
```

## Step 2: Create Vector Table

```sql
CREATE TABLE articles (
    id SERIAL PRIMARY KEY,
    title TEXT,
    content TEXT,
    embedding vector(384)
);
```

## Step 3: Generate Embeddings

```sql
INSERT INTO articles (title, content, embedding)
VALUES (
    'Machine Learning',
    'Machine learning is a subset of AI',
    embed_text('Machine learning is a subset of AI')
);
```

## Step 4: Create Index

```sql
SELECT hnsw_create_index('articles', 'embedding', 'articles_idx', 16, 200);
```

## Step 5: Search

```sql
SELECT id, title,
       embedding <-> embed_text('artificial intelligence') AS distance
FROM articles
ORDER BY distance
LIMIT 5;
```

## Next Steps

- **[Vector Types](../vector-search/vector-types.md)** - Learn about different vector formats
- **[Embeddings](../ml-embeddings/embedding-generation.md)** - Embedding generation
- **[Hybrid Search](../hybrid-search/overview.md)** - Combine semantic and keyword search
- **[Configuration](../configuration.md)** - Configuration options

## Learn More

For detailed documentation, examples, and comprehensive guides, visit:

**[Detailed Documentation](https://neurondb.ai/docs)**
