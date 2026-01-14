# 🎯 Top 20 NeuronDB Functions You Need

<div align="center">

**The most commonly used SQL functions in NeuronDB**

[![Functions](https://img.shields.io/badge/functions-20-blue)](.)
[![Complete API](https://img.shields.io/badge/complete-520+-functions-blue)](../../NeuronDB/docs/sql-api.md)

</div>

---

> [!TIP]
> **Need more?** See the [Complete SQL API Reference](../../NeuronDB/docs/sql-api.md) for all 520+ functions.

---

## 📚 Table of Contents

- [Vector Search & Similarity](#-vector-search--similarity)
- [Indexing](#-indexing)
- [Search & Retrieval](#-search--retrieval)
- [Hybrid Search](#-hybrid-search)
- [RAG Pipeline](#-rag-pipeline)
- [ML Algorithms](#-ml-algorithms)
- [Utility Functions](#-utility-functions)

---

## 🔍 Vector Search & Similarity

### 1. `embed_text(text, model)` → `vector`

**Generate text embeddings for semantic search.**

```sql
-- Basic usage
SELECT embed_text('Hello world', 'all-MiniLM-L6-v2');
-- Returns: vector(384)

-- Store in table
UPDATE documents 
SET embedding = embed_text(content, 'all-MiniLM-L6-v2')
WHERE embedding IS NULL;
```

**Parameters:**
- `text` - Text to embed
- `model` - Embedding model name (e.g., `'all-MiniLM-L6-v2'`)

**Returns:** `vector` - Embedding vector (dimension depends on model)

**Use cases:**
- Generate embeddings for search
- Convert text to vectors
- Prepare data for similarity search

> [!NOTE]
> **Model dimensions:** Different models produce different vector dimensions. `all-MiniLM-L6-v2` produces 384-dimensional vectors.

---

### 2. `embed_text_batch(text[], model)` → `vector[]`

**Batch embedding generation (more efficient).**

```sql
-- Generate embeddings for multiple texts
SELECT embed_text_batch(
  ARRAY['text1', 'text2', 'text3'],
  'all-MiniLM-L6-v2'
);

-- Use with INSERT
INSERT INTO documents (content, embedding)
SELECT 
  content,
  unnest(embed_text_batch(ARRAY[content], 'all-MiniLM-L6-v2'))
FROM source_table;
```

**Parameters:**
- `text[]` - Array of texts to embed
- `model` - Embedding model name

**Returns:** `vector[]` - Array of embedding vectors

**Use cases:**
- Bulk embedding generation
- Processing large datasets
- ETL pipelines

> [!TIP]
> **Performance:** Batch operations are 5-10x faster than individual calls. Use `embed_text_batch` when processing multiple texts.

---

### 3. Distance Operators: `<->`, `<=>`, `<#>`

**Vector similarity operators for different distance metrics.**

```sql
-- L2 (Euclidean) distance
SELECT embedding <-> '[0.1, 0.2, 0.3]'::vector AS distance;

-- Cosine distance (most common)
SELECT embedding <=> '[0.1, 0.2, 0.3]'::vector AS distance;

-- Inner product (negative = more similar)
SELECT embedding <#> '[0.1, 0.2, 0.3]'::vector AS distance;
```

**Operators:**
- `<->` - L2/Euclidean distance (0 = identical, ∞ = different)
- `<=>` - Cosine distance (0 = identical, 2 = opposite)
- `<#>` - Inner product (higher = more similar, negative = more similar)

**Use cases:**
- Similarity search
- Ranking results
- Clustering

> [!NOTE]
> **Which to use?** Cosine distance (`<=>`) is most common for text embeddings. L2 (`<->`) is better for normalized vectors. Inner product (`<#>`) is fastest but requires normalized vectors.

---

## 🗂️ Indexing

### 4. `CREATE INDEX ... USING hnsw`

**Create HNSW index for fast approximate nearest neighbor search.**

```sql
-- Basic HNSW index
CREATE INDEX documents_embedding_idx 
ON documents 
USING hnsw (embedding vector_cosine_ops);

-- With custom parameters
CREATE INDEX documents_embedding_idx 
ON documents 
USING hnsw (embedding vector_cosine_ops)
WITH (
  m = 16,              -- Number of connections per node
  ef_construction = 200 -- Search width during construction
);
```

**Parameters:**
- `m` - Number of connections per node (default: 16, range: 4-64)
- `ef_construction` - Search width during construction (default: 200, range: 4-1000)

**Use cases:**
- Fast similarity search
- Large-scale vector databases
- Production deployments

> [!TIP]
> **Performance:** HNSW indexes make search 100-1000x faster on large datasets. Always create an index for production use.

---

### 5. `CREATE INDEX ... USING ivf`

**Create IVF index for faster builds on large datasets.**

```sql
-- Basic IVF index
CREATE INDEX documents_embedding_idx 
ON documents 
USING ivf (embedding vector_cosine_ops);

-- With custom parameters
CREATE INDEX documents_embedding_idx 
ON documents 
USING ivf (embedding vector_cosine_ops)
WITH (
  lists = 1024  -- Number of clusters
);
```

**Parameters:**
- `lists` - Number of clusters (default: 100, range: 1-10000)

**Use cases:**
- Very large datasets (millions+ vectors)
- Faster index building
- Memory-constrained environments

> [!NOTE]
> **HNSW vs IVF:** HNSW is faster for queries but slower to build. IVF is faster to build but slightly slower for queries. Use HNSW for most cases.

---

## 🔎 Search & Retrieval

### 6. Vector Similarity Search

**Standard vector search query pattern.**

```sql
-- Basic similarity search
SELECT 
  id, 
  content, 
  embedding <=> query_vector AS distance
FROM documents
ORDER BY embedding <=> query_vector
LIMIT 10;

-- With query embedding
WITH query AS (
  SELECT embed_text('machine learning', 'all-MiniLM-L6-v2') AS q_vec
)
SELECT 
  id,
  content,
  embedding <=> q.q_vec AS distance
FROM documents, query q
ORDER BY embedding <=> q.q_vec
LIMIT 10;
```

**Pattern:**
1. Generate query embedding (or use existing)
2. Calculate distance to all vectors
3. Order by distance
4. Limit results

**Use cases:**
- Semantic search
- Recommendation systems
- Similarity matching

---

### 7. `mmr_rerank(table, column, query_vector, k, lambda)`

**Maximal Marginal Relevance reranking for diverse results.**

```sql
-- MMR reranking
SELECT * FROM neurondb.mmr_rerank(
  'documents',           -- table name
  'embedding',           -- column name
  '[0.1, 0.2, 0.3]'::vector,  -- query vector
  10,                    -- top k
  0.7                    -- lambda (diversity vs relevance)
);
```

**Parameters:**
- `table` - Table name
- `column` - Vector column name
- `query_vector` - Query embedding
- `k` - Number of results
- `lambda` - Diversity factor (0.0 = relevance only, 1.0 = diversity only)

**Returns:** Table with reranked results

**Use cases:**
- Diverse search results
- Avoiding duplicate content
- Recommendation systems

> [!TIP]
> **Lambda values:** Use 0.5-0.7 for balanced results. Lower values prioritize relevance, higher values prioritize diversity.

---

### 8. `rerank_cross_encoder(query, candidates, model, top_k)`

**Neural reranking using cross-encoder models.**

```sql
-- Cross-encoder reranking
SELECT * FROM neurondb.rerank_cross_encoder(
  'query text',                    -- query
  ARRAY['candidate1', 'candidate2', 'candidate3'],  -- candidates
  NULL,                            -- use default model
  5                                -- top k
);
```

**Parameters:**
- `query` - Query text
- `candidates` - Array of candidate texts
- `model` - Reranking model (NULL = default)
- `top_k` - Number of results

**Returns:** Reranked candidates with scores

**Use cases:**
- High-precision reranking
- Small candidate sets (< 100)
- Final ranking step

> [!NOTE]
> **Performance:** Cross-encoders are slower but more accurate than bi-encoders. Use for final reranking of small candidate sets.

---

## 🔗 Hybrid Search

### 9. Hybrid Vector + Full-Text Search

**Combine vector similarity with PostgreSQL full-text search.**

```sql
-- Hybrid search
WITH query AS (
  SELECT 
    embed_text('machine learning', 'all-MiniLM-L6-v2') AS q_vec,
    to_tsquery('english', 'machine & learning') AS q_tsquery
)
SELECT 
  id,
  content,
  -- Combined score: 70% vector, 30% full-text
  (embedding <=> q.q_vec) * 0.7 + 
  (ts_rank(to_tsvector('english', content), q.q_tsquery) * 0.3) AS combined_score
FROM documents, query q
WHERE to_tsvector('english', content) @@ q.q_tsquery
ORDER BY combined_score DESC
LIMIT 10;
```

**Pattern:**
1. Generate vector embedding
2. Create full-text search query
3. Combine scores with weights
4. Filter and rank

**Use cases:**
- Better search quality
- Combining semantic and keyword search
- Production search systems

> [!TIP]
> **Score weights:** Adjust weights (0.7, 0.3) based on your data. More text-heavy content benefits from higher full-text weight.

---

### 10. `rrf_rerank(vector_results, text_results, k, rrf_k)`

**Reciprocal Rank Fusion for combining multiple result sets.**

```sql
-- RRF reranking
SELECT * FROM neurondb.rrf_rerank(
  vector_results,  -- vector search results
  text_results,     -- full-text search results
  10,               -- top k
  60                -- rrf_k parameter
);
```

**Parameters:**
- `vector_results` - Vector search result set
- `text_results` - Full-text search result set
- `k` - Number of results
- `rrf_k` - RRF constant (default: 60)

**Returns:** Combined and reranked results

**Use cases:**
- Combining different search methods
- Multi-stage retrieval
- Ensemble search

---

## 📄 RAG Pipeline

### 11. `chunk_text(text, chunk_size, overlap)`

**Split documents into chunks for RAG.**

```sql
-- Basic chunking
SELECT neurondb.chunk_text(
  'Long document text...',
  500,  -- chunk size in tokens
  50    -- overlap in tokens
);

-- Chunk and embed
WITH chunks AS (
  SELECT neurondb.chunk_text(content, 500, 50) AS chunk
  FROM documents
)
SELECT 
  chunk,
  embed_text(chunk, 'all-MiniLM-L6-v2') AS embedding
FROM chunks;
```

**Parameters:**
- `text` - Text to chunk
- `chunk_size` - Size of each chunk (in tokens)
- `overlap` - Overlap between chunks (in tokens)

**Returns:** Array of text chunks

**Use cases:**
- RAG pipeline preparation
- Document preprocessing
- Context window management

> [!TIP]
> **Chunk size:** Use 200-500 tokens for most cases. Larger chunks = more context but less precise retrieval.

---

### 12. `retrieve_context(query_vector, table, column, k, filters)`

**Retrieve context for RAG queries.**

```sql
-- Basic context retrieval
SELECT * FROM neurondb.retrieve_context(
  (SELECT embed_text('query', 'all-MiniLM-L6-v2')),
  'documents', 'embedding',
  10,    -- top k
  NULL   -- optional filters
);

-- With filters
SELECT * FROM neurondb.retrieve_context(
  query_vector,
  'documents', 'embedding',
  10,
  'category = ''tech'' AND created_at > ''2024-01-01'''
);
```

**Parameters:**
- `query_vector` - Query embedding
- `table` - Table name
- `column` - Vector column name
- `k` - Number of results
- `filters` - Optional SQL WHERE clause

**Returns:** Context documents

**Use cases:**
- RAG context retrieval
- Question answering
- Document Q&A systems

---

### 13. `rag_evaluate(query, retrieved_docs, generated_answer)`

**Evaluate RAG pipeline quality using RAGAS metrics.**

```sql
-- RAG evaluation
SELECT * FROM neurondb.rag_evaluate(
  'What is machine learning?',           -- query
  ARRAY['doc1', 'doc2', 'doc3'],         -- retrieved docs
  'Machine learning is...'                -- generated answer
);
```

**Parameters:**
- `query` - Original query
- `retrieved_docs` - Retrieved documents
- `generated_answer` - Generated answer

**Returns:** Evaluation metrics (faithfulness, answer relevancy, context precision, etc.)

**Use cases:**
- RAG pipeline evaluation
- Quality monitoring
- A/B testing

---

## 🤖 ML Algorithms

### 14. `kmeans_cluster(vectors, k, max_iter)`

**K-Means clustering for vector data.**

```sql
-- K-Means clustering
SELECT * FROM neurondb.kmeans_cluster(
  (SELECT array_agg(embedding) FROM documents),
  5,        -- number of clusters
  100       -- max iterations
);
```

**Use cases:**
- Document clustering
- Customer segmentation
- Data exploration

---

### 15. `train_classifier(features, labels, algorithm)`

**Train a classification model.**

```sql
-- Train classifier
SELECT neurondb.train_classifier(
  features_array,
  labels_array,
  'random_forest'  -- algorithm
);
```

**Use cases:**
- Classification tasks
- Predictive modeling
- ML pipelines

---

## 🛠️ Utility Functions

### 16. `neurondb.version()`

**Get NeuronDB version.**

```sql
SELECT neurondb.version();
-- Returns: '2.0'
```

**Use cases:**
- Version checking
- Compatibility verification
- Debugging

---

### 17. `vector_dim(vector)`

**Get vector dimension.**

```sql
SELECT vector_dim(embedding) FROM documents LIMIT 1;
-- Returns: 384 (for all-MiniLM-L6-v2)
```

**Use cases:**
- Schema validation
- Data verification
- Debugging

---

### 18. `normalize(vector)`

**Normalize vector to unit length.**

```sql
-- Normalize vector
SELECT normalize(embedding) FROM documents;
```

**Use cases:**
- Preprocessing
- Distance calculations
- Normalization requirements

---

### 19. `vector_add(vector1, vector2)`

**Add two vectors element-wise.**

```sql
-- Vector addition
SELECT vector_add(embedding1, embedding2) FROM documents;
```

**Use cases:**
- Vector arithmetic
- Feature engineering
- Data transformations

---

### 20. `vector_subtract(vector1, vector2)`

**Subtract two vectors element-wise.**

```sql
-- Vector subtraction
SELECT vector_subtract(embedding1, embedding2) FROM documents;
```

**Use cases:**
- Vector arithmetic
- Difference calculations
- Feature engineering

---

## 📚 Complete API Reference

For all 520+ functions, see the [Complete SQL API Reference](../../NeuronDB/docs/sql-api.md).

---

## 🎓 Learning Resources

| Resource | Description |
|----------|-------------|
| **[SQL Recipes](../../examples/sql-recipes/)** | Ready-to-run examples |
| **[Getting Started Guide](../getting-started/quickstart.md)** | Step-by-step tutorial |
| **[Architecture Guide](../getting-started/architecture.md)** | System overview |
| **[Complete Documentation](../../documentation.md)** | Full documentation index |

---

<div align="center">

[⬆ Back to Top](#-top-20-neurondb-functions-you-need) · [📚 Main Documentation](../../documentation.md) · [🔍 Complete API](../../NeuronDB/docs/sql-api.md)

</div>
