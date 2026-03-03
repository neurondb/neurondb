![PostgreSQL as Vector Database header](assets/postgresql-vector-database/header.svg)

# PostgreSQL as a Vector Database

**[View on GitHub](https://github.com/neurondb-ai/neurondb)** | **[Download Latest Release](https://github.com/neurondb-ai/neurondb/releases)** | **[Documentation](https://neurondb.ai/docs)**

PostgreSQL stores and queries vector data using SQL. Extensions add vector types and indexes. NeuronDB adds vector types, HNSW indexes, and ML functions.

Standalone vector databases require separate infrastructure. They use specialized query languages. They require data synchronization between systems. PostgreSQL stores vectors and relational data in the same tables. It uses SQL for queries. It provides ACID transactions for vector data.

This document covers architecture, performance, indexing, queries, and migration. Examples use SQL. Examples use NeuronDB extension.

## What is a Vector Database

Vector databases store numeric vectors. Vectors represent data as numbers. Similarity search finds related items by computing distances between vectors. Traditional databases match exact values. They do not compute semantic similarity efficiently.

Vector databases support semantic search, recommendations, RAG systems, image search, and anomaly detection. These applications require fast similarity search across millions of vectors.

PostgreSQL with NeuronDB provides vector types, indexes, distance operators, and SQL queries. It stores vectors and relational data in the same tables. It uses ACID transactions for vector data. It uses SQL for queries.

## PostgreSQL Vector Database Architecture

PostgreSQL stores vectors using native types. It indexes vectors using HNSW and IVFFlat. It computes distances in SQL queries. It optimizes vector and relational operations together.

![PostgreSQL Vector Database Architecture](assets/postgresql-vector-database/diagram-architecture.svg)

PostgreSQL stores vectors and relational data in the same tables. Vector storage uses vector types. Relational storage uses standard types. Queries combine both types in single statements. The query planner optimizes both operations. HNSW and IVFFlat indexes accelerate similarity search.

Unified storage eliminates data synchronization. Integrated query planning enables complex queries. ACID transactions, backup, recovery, and monitoring tools work with vector data.

### Vector Storage Layer

PostgreSQL stores vectors using vector types. Vector columns accept fixed dimensions. Common dimensions are 384, 768, and 1024. Storage uses binary formats. It stores millions of vectors in standard tables.

Create a table with vector and relational columns:

```sql
CREATE TABLE documents (
    id serial PRIMARY KEY,
    title text NOT NULL,
    content text,
    embedding vector(384),
    created_at timestamp DEFAULT CURRENT_TIMESTAMP,
    metadata jsonb
);

CREATE INDEX idx_documents_embedding ON documents 
USING hnsw (embedding vector_cosine_ops);

```

The embedding column uses vector type with 384 dimensions. The HNSW index accelerates similarity search. Relational columns store metadata. Queries combine vector similarity with relational filters.

### Vector Insertion Examples

PostgreSQL supports multiple methods for inserting vectors. The following examples show insertion patterns:

**Example 1: Inserting vectors from arrays**

```sql
-- Insert vectors using array_to_vector function
INSERT INTO documents (title, content, embedding) VALUES
    ('Machine Learning Basics', 
     'Introduction to supervised learning algorithms',
     array_to_vector(ARRAY[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]::real[])),
    ('Deep Learning Fundamentals',
     'Neural networks and backpropagation explained',
     array_to_vector(ARRAY[0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.1]::real[])),
    ('Natural Language Processing',
     'Text embeddings and transformer models',
     array_to_vector(ARRAY[0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.1, 0.2]::real[]));

-- Results:
-- INSERT 0 3

```

**Example 2: Generating embeddings automatically**

```sql
-- Insert documents with automatic embedding generation
INSERT INTO documents (title, content, embedding) VALUES
    ('PostgreSQL Performance', 
     'Optimizing database queries and indexes',
     embed_text('PostgreSQL Performance: Optimizing database queries and indexes', 'sentence-transformers/all-MiniLM-L6-v2')),
    ('Vector Search Guide',
     'Implementing similarity search with embeddings',
     embed_text('Vector Search Guide: Implementing similarity search with embeddings', 'sentence-transformers/all-MiniLM-L6-v2')),
    ('Database Architecture',
     'Designing scalable database systems',
     embed_text('Database Architecture: Designing scalable database systems', 'sentence-transformers/all-MiniLM-L6-v2'));

-- Results:
-- INSERT 0 3

```

**Example 3: Batch insertion with COPY**

```sql
-- Prepare data file (example CSV format)
-- documents.csv:
-- title,content,embedding
-- "SQL Tutorial","Learn SQL basics",[0.1,0.2,0.3,...]
-- "Python Guide","Python programming",[0.2,0.3,0.4,...]

-- Bulk insert using COPY
COPY documents (title, content, embedding)
FROM '/path/to/documents.csv'
WITH (FORMAT csv, HEADER true);

-- Results:
-- COPY 1000

```

**Example 4: Inserting with metadata**

```sql
-- Insert documents with rich metadata
INSERT INTO documents (title, content, embedding, metadata) VALUES
    ('AI Research Paper',
     'Recent advances in transformer architectures',
     embed_text('AI Research Paper: Recent advances in transformer architectures'),
     '{"author": "Dr. Smith", "category": "research", "tags": ["AI", "ML", "transformers"], "year": 2024}'::jsonb),
    ('Database Tutorial',
     'PostgreSQL administration guide',
     embed_text('Database Tutorial: PostgreSQL administration guide'),
     '{"author": "Jane Doe", "category": "tutorial", "tags": ["database", "postgresql"], "difficulty": "intermediate"}'::jsonb);

-- Results:
-- INSERT 0 2

```

**Example 5: Updating existing vectors**

```sql
-- Update embedding for existing document
UPDATE documents
SET embedding = embed_text('Updated content: ' || content, 'sentence-transformers/all-MiniLM-L6-v2')
WHERE id = 1;

-- Results:
-- UPDATE 1

```

Array-based insertion provides direct control. Automatic embedding generation eliminates external services. Batch insertion loads data efficiently. Metadata storage stores document attributes. Vector updates support content changes.

### Indexing Layer

PostgreSQL supports multiple index types. HNSW indexes provide fast search with high recall. Query latency is 5-8ms. IVFFlat indexes build faster and use less memory. They suit frequent updates. DiskANN indexes store billions of vectors on disk. They suit datasets exceeding memory capacity.

The following example shows creating different index types:

```sql
-- HNSW index for high-performance similarity search
CREATE INDEX idx_hnsw_embedding ON documents 
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- IVFFlat index for faster updates
CREATE INDEX idx_ivfflat_embedding ON documents 
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

-- DiskANN index for billion-scale storage
CREATE INDEX idx_diskann_embedding ON documents 
USING diskann (embedding vector_cosine_ops);

```

HNSW indexes provide 5-8ms latency on 100 million vectors. IVFFlat indexes build faster. They suit frequent updates. DiskANN indexes store billions of vectors on disk. Query latency is 30-50ms. Choose indexes based on dataset size, update frequency, and latency requirements.

### Query Layer

PostgreSQL uses SQL with distance operators. The <-> operator computes L2 distance. The <=> operator computes cosine distance. The <#> operator computes inner product. Use these operators in WHERE clauses, ORDER BY clauses, and JOIN operations.

![Vector Query Flow](assets/postgresql-vector-database/diagram-query-flow.svg)

PostgreSQL processes vector queries through multiple stages. Query input accepts SQL with vector operators. The query planner selects indexes. Index scans use HNSW or IVFFlat structures. Filtering and ranking combine vector similarity with relational constraints. Results return ranked vectors with metadata. Query latency is under 10ms.

The following example shows a query that combines vector similarity with relational filters:

```sql
WITH query_vector AS (
    SELECT embed_text('machine learning algorithms') AS vec
)
SELECT 
    d.id,
    d.title,
    d.content,
    d.embedding <=> q.vec AS distance,
    1 - (d.embedding <=> q.vec) AS similarity
FROM documents d, query_vector q
WHERE d.created_at > '2024-01-01'
  AND d.metadata->>'category' = 'technical'
ORDER BY d.embedding <=> q.vec
LIMIT 10;

```

The query combines vector similarity with relational filters. The WHERE clause filters by date and category. The ORDER BY clause ranks by cosine distance. Standalone vector databases do not support relational constraints.

## PostgreSQL vs Standalone Vector Databases

PostgreSQL provides ACID transactions for vector data. It uses SQL for queries. It stores vectors and relational data in the same tables. It combines vector similarity with relational filters in single queries. It uses standard PostgreSQL monitoring, backup, and administration tools.

![PostgreSQL vs Standalone Vector Databases](assets/postgresql-vector-database/diagram-comparison.svg)

PostgreSQL provides ACID transactions, SQL querying, unified storage, and standard tooling. Standalone systems do not provide these capabilities.

ACID transactions ensure vector updates occur atomically with relational updates. SQL interface uses existing skills. Unified storage eliminates data synchronization between systems.

## Feature Comparison: PostgreSQL Vector DB vs Other Solutions

PostgreSQL with NeuronDB provides vector indexing, ML inference, embedding generation, and RAG pipelines. This section compares features across indexing, ML capabilities, performance, and operations.

![Vector Database Feature Comparison](assets/postgresql-vector-database/diagram-feature-comparison.svg)

PostgreSQL with NeuronDB combines vector search, ML inference, embedding generation, and RAG pipelines in one platform.

### PostgreSQL Extensions Comparison

PostgreSQL vector database solutions include several extensions that add vector capabilities. The following comparison highlights key differences:

| Feature | **NeuronDB** | **Alternatives** |
|---------|-------------|------------------|
| **Vector Indexing** | HNSW, IVFFlat, DiskANN | Various options |
| **ML Inference** | ONNX (C++), 52 algorithms | Limited or external |
| **Embedding Generation** | In-database (ONNX) | External tools typically required |
| **Hybrid Search** | Native (Vector+FTS) | Manual combination typically required |
| **Reranking** | Cross-encoder, LLM, ColBERT, MMR | Limited or none |
| **ML Algorithms** | 52 algorithms (RF, XGBoost, LightGBM, CatBoost, SVM, KNN, etc.) | Limited sets |
| **Background Workers** | 4 workers (neuranq, neuranmon, neurandefrag, neuranllm) | Typically none |
| **RAG Pipeline** | Complete in-database | Partial or external |
| **Quantization** | FP16, INT8, Binary (2x-32x compression) | Limited options |
| **GPU Support** | CUDA + ROCm + Metal (native) | Limited or none |
| **Distance Metrics** | 10+ (L2, Cosine, Inner Product, Manhattan, Hamming, Jaccard, etc.) | Typically 3-4 metrics |
| **Vector Types** | 5 types (vector, vectorp, vecmap, vgraph, rtext) | Typically 1 type |
| **SQL Functions** | 520+ functions | Typically 20-50 functions |
| **Performance (QPS)** | 100K+ (with GPU), 1K-2K (CPU) | Varies widely |
| **Dependencies** | Zero (pure C, optional ONNX) | Varies |
| **PostgreSQL Versions** | 16, 17, 18 | 12-18 | 15-18 | 16-18 | 14-16 |
| **ACID Transactions** | Full ACID for vectors | Full ACID for vectors | Full ACID for vectors | Full ACID for vectors | Full ACID for vectors |
| **SQL Interface** | Standard SQL | Standard SQL | Standard SQL | Standard SQL | Standard SQL |

NeuronDB provides ML inference, embedding generation, reranking, and RAG pipelines in PostgreSQL. All features are implemented in-database with no external dependencies.

### Standalone Vector Database Comparison

Standalone vector databases provide vector search but lack relational database features. The following comparison shows differences:

| Feature | **PostgreSQL + NeuronDB** | **Pinecone** | **Weaviate** | **Qdrant** | **Milvus** | **Chroma** |
|---------|---------------------------|--------------|--------------|------------|------------|------------|
| **Deployment** | Self-hosted or managed | Managed only | Self-hosted or managed | Self-hosted or managed | Self-hosted or managed | Self-hosted |
| **ACID Transactions** | Full ACID guarantees | Limited | Eventual consistency | Limited | Eventual consistency | Limited |
| **SQL Interface** | Standard SQL | REST API only | GraphQL + REST | REST API only | REST API only | Python API |
| **Relational Data** | Native support | Separate system | Separate system | Separate system | Separate system | Separate system |
| **Hybrid Queries** | Native SQL combining vector + relational | Vector only | Vector + GraphQL | Vector only | Vector only | Vector only |
| **Index Types** | HNSW, IVFFlat, DiskANN | HNSW | HNSW, IVF | HNSW, IVF | HNSW, IVF, FLAT | HNSW |
| **ML Inference** | Built-in (ONNX) | External API | External API | External API | External API | External API |
| **Embedding Generation** | In-database | External API | External API | External API | External API | External API |
| **Reranking** | Built-in (Cross-encoder, LLM, ColBERT) | External API | External API | External API | External API | External API |
| **RAG Pipeline** | Complete in-database | External services | External services | External services | External services | External services |
| **Backup & Recovery** | PostgreSQL standard tools | Managed service | Custom solutions | Custom solutions | Custom solutions | Custom solutions |
| **Monitoring** | PostgreSQL tools (pg_stat, pgAdmin) | Proprietary dashboard | Proprietary dashboard | Proprietary dashboard | Proprietary dashboard | Limited |
| **Query Latency** | 5-8ms (HNSW, 100M vectors) | 10-50ms | 5-20ms | 5-15ms | 5-20ms | 10-30ms |
| **Throughput** | 1K-2K QPS (CPU), 10K-15K (GPU) | 100-500 QPS | 500-2K QPS | 1K-5K QPS | 1K-3K QPS | 100-500 QPS |
| **Scalability** | 100M+ vectors (single instance), billions (partitioned) | Millions to billions | Millions to billions | Millions to billions | Billions | Millions |
| **Cost Model** | Open source, self-hosted | Pay-per-use | Open source or managed | Open source or managed | Open source or managed | Open source |
| **Learning Curve** | SQL (familiar to DBAs) | New API to learn | GraphQL + REST | New API to learn | New API to learn | Python API |
| **Data Consistency** | Strong consistency | Eventual consistency | Eventual consistency | Eventual consistency | Eventual consistency | Eventual consistency |
| **Tooling Ecosystem** | Full PostgreSQL ecosystem | Limited | Limited | Limited | Limited | Limited |

PostgreSQL with NeuronDB provides ACID transactions for vector data. Standalone systems do not provide this. SQL interface uses existing skills. Standalone systems require learning new APIs. Unified storage eliminates data synchronization. Standalone systems require separate relational databases for metadata.

### Performance Comparison

![Performance Comparison](assets/postgresql-vector-database/diagram-performance-comparison.svg)

PostgreSQL + NeuronDB achieves 5-8ms average query latency. Index construction takes 15-30 minutes. GPU acceleration provides 10,000-15,000 QPS. These characteristics, combined with ACID transactions and SQL querying, suit production deployments.

### Use Case Recommendations

Choose PostgreSQL + NeuronDB when:
- ACID transactions for vector data are required
- Combining vector similarity with relational queries is needed
- Existing PostgreSQL infrastructure exists
- In-database ML inference and embedding generation are needed
- Complete RAG pipelines without external services are required
- Standard PostgreSQL tooling and monitoring are needed
- Strong data consistency guarantees are required

NeuronDB provides comprehensive vector database features:
- Only basic vector indexing is needed
- ML inference or embedding generation are not needed
- Minimal dependencies are required
- Simple similarity search applications are being built

Choose Pinecone when:
- Fully managed service without infrastructure management is needed
- Relational data integration is not needed
- API-based access only is acceptable
- Pay-per-use pricing is preferred

Choose Weaviate when:
- Graph database capabilities alongside vectors are needed
- GraphQL interface is preferred
- Strong consistency guarantees are not needed
- Knowledge graph applications are being built

Choose Qdrant when:
- High-performance vector search only is needed
- Relational database features are not needed
- Rust-based implementation is preferred
- Pure vector search applications are being built

Choose Milvus when:
- Billion-scale vector storage is needed
- Relational database features are not needed
- Distributed architecture is preferred
- Large-scale vector search systems are being built

## Performance Characteristics

Performance depends on index selection, dataset size, and query patterns. HNSW indexes provide under 10ms latency on 100 million vectors with 768 dimensions. IVFFlat indexes provide 15-25ms latency with faster construction. DiskANN indexes provide 30-50ms latency on billion-scale datasets.

The following benchmarks show capabilities:

```sql
-- Performance test: Query latency measurement
EXPLAIN ANALYZE
SELECT id, title, embedding <=> embed_text('query') AS distance
FROM documents
ORDER BY embedding <=> embed_text('query')
LIMIT 10;

-- Results (HNSW index, 10M vectors, 768 dimensions):
-- Planning Time: 0.123 ms
-- Execution Time: 5.234 ms
-- Index Scan using idx_hnsw_embedding: 5.234 ms

```

PostgreSQL achieves 5.2ms average query latency with HNSW indexing on 10 million vectors. Performance scales linearly up to 100 million vectors. DiskANN indexes maintain acceptable latency beyond that.

PostgreSQL handles 1,000-2,000 queries per second on single instances with CPU. GPU-accelerated HNSW indexes provide 10,000-15,000 queries per second on single GPU systems. Multi-GPU systems handle 50,000 or more queries per second.

## Indexing Strategies

Index selection balances query performance, construction time, memory usage, and update frequency. HNSW indexes provide 5-8ms latency but require longer construction and more memory. IVFFlat indexes build faster and use less memory but have 15-25ms latency. DiskANN indexes store billions of vectors on disk but require 30-50ms latency.

The following example demonstrates index creation with different parameters for various use cases:

```sql
-- High-performance index for production workloads
CREATE INDEX idx_production ON documents 
USING hnsw (embedding vector_cosine_ops)
WITH (
    m = 16,              -- Number of connections per node
    ef_construction = 64 -- Construction search width
);

-- Balanced index for frequently updated data
CREATE INDEX idx_balanced ON documents 
USING ivfflat (embedding vector_cosine_ops)
WITH (
    lists = 100          -- Number of clusters
);

-- Large-scale index for billion-vector datasets
CREATE INDEX idx_large_scale ON documents 
USING diskann (embedding vector_cosine_ops)
WITH (
    max_degree = 64,     -- Maximum graph degree
    search_list_size = 100 -- Search list size
);

```

Select index parameters based on workload. Production applications use HNSW with m=16 and ef_construction=64. Frequently updated datasets use IVFFlat with lists=100. Billion-scale datasets use DiskANN to maintain acceptable latency while storing data on disk.

## Query Patterns

PostgreSQL combines vector similarity with relational operations. Basic similarity search finds closest vectors. Filtered similarity search combines vector similarity with WHERE clauses. Hybrid queries combine multiple vector and relational conditions. Aggregation queries compute statistics over results. Join queries combine vector similarity with relational joins.

The following example demonstrates a filtered similarity search that combines vector similarity with relational filters:

```sql
WITH query AS (
    SELECT embed_text('database optimization') AS vec
)
SELECT 
    d.id,
    d.title,
    d.embedding <=> q.vec AS distance,
    d.metadata->>'author' AS author,
    d.created_at
FROM documents d, query q
WHERE d.metadata->>'category' = 'database'
  AND d.created_at > '2024-01-01'
  AND d.embedding <=> q.vec < 0.3
ORDER BY d.embedding <=> q.vec
LIMIT 20;

```

The query combines vector similarity with multiple relational filters. The WHERE clause filters by category, date, and distance threshold. ORDER BY ranks by similarity. Standalone vector databases do not support relational filtering.

Aggregation queries demonstrate PostgreSQL vector database capabilities for analytics workloads:

```sql
WITH query AS (
    SELECT embed_text('artificial intelligence') AS vec
)
SELECT 
    d.metadata->>'category' AS category,
    COUNT(*) AS document_count,
    AVG(d.embedding <=> q.vec) AS avg_distance,
    MIN(d.embedding <=> q.vec) AS min_distance,
    MAX(d.embedding <=> q.vec) AS max_distance
FROM documents d, query q
WHERE d.embedding <=> q.vec < 0.5
GROUP BY d.metadata->>'category'
ORDER BY avg_distance;

```

The aggregation query computes statistics over results. It groups by category and calculates average, minimum, and maximum distances. PostgreSQL combines vector similarity with SQL aggregation functions.

### Additional Vector Query Examples

PostgreSQL supports query patterns that combine vector similarity with relational operations. The following examples show advanced capabilities:

**Example 1: Multi-vector similarity search**

```sql
-- Find documents similar to multiple query vectors
WITH queries AS (
    SELECT 
        embed_text('machine learning') AS vec1,
        embed_text('neural networks') AS vec2,
        embed_text('deep learning') AS vec3
)
SELECT 
    d.id,
    d.title,
    (d.embedding <=> q.vec1 + 
     d.embedding <=> q.vec2 + 
     d.embedding <=> q.vec3) / 3.0 AS avg_distance
FROM documents d, queries q
ORDER BY avg_distance
LIMIT 10;

-- Results:
-- id |           title            | avg_distance
-- ----+----------------------------+--------------
--  5 | Deep Learning Fundamentals |     0.123456
--  2 | Machine Learning Basics    |     0.234567
--  8 | Neural Network Architecture|     0.345678
-- (10 rows)

```

**Example 2: Vector similarity with JOIN operations**

```sql
-- Join vector search with related tables
WITH query_vector AS (
    SELECT embed_text('database optimization') AS vec
)
SELECT 
    d.id,
    d.title,
    d.embedding <=> q.vec AS distance,
    a.name AS author_name,
    a.email AS author_email,
    c.name AS category_name
FROM documents d
CROSS JOIN query_vector q
LEFT JOIN authors a ON d.metadata->>'author_id' = a.id::text
LEFT JOIN categories c ON d.metadata->>'category_id' = c.id::text
WHERE d.embedding <=> q.vec < 0.5
ORDER BY d.embedding <=> q.vec
LIMIT 20;

-- Results:
-- id |        title         | distance | author_name |      author_email      | category_name
-- ----+----------------------+----------+-------------+-----------------------+---------------
-- 12 | Query Optimization   |  0.12345 | John Smith  | john@example.com      | Database
-- 15 | Index Strategies     |  0.23456 | Jane Doe     | jane@example.com      | Performance
-- (20 rows)

```

**Example 3: Time-based vector search**

```sql
-- Find recent documents similar to query
WITH query_vector AS (
    SELECT embed_text('artificial intelligence') AS vec
)
SELECT 
    d.id,
    d.title,
    d.created_at,
    d.embedding <=> q.vec AS distance,
    EXTRACT(EPOCH FROM (NOW() - d.created_at)) / 86400 AS days_ago
FROM documents d, query_vector q
WHERE d.created_at > NOW() - INTERVAL '30 days'
  AND d.embedding <=> q.vec < 0.6
ORDER BY d.embedding <=> q.vec, d.created_at DESC
LIMIT 15;

-- Results:
-- id |         title          |      created_at      | distance | days_ago
-- ----+------------------------+----------------------+----------+----------
-- 23 | AI Trends 2024          | 2024-12-15 10:30:00  |  0.12345 |     2.5
-- 18 | Machine Learning Guide  | 2024-12-10 14:20:00  |  0.23456 |     7.3
-- (15 rows)

```

**Example 4: Vector similarity with window functions**

```sql
-- Rank documents by similarity within categories
WITH query_vector AS (
    SELECT embed_text('data science') AS vec
)
SELECT 
    d.id,
    d.title,
    d.metadata->>'category' AS category,
    d.embedding <=> q.vec AS distance,
    ROW_NUMBER() OVER (PARTITION BY d.metadata->>'category' ORDER BY d.embedding <=> q.vec) AS rank_in_category
FROM documents d, query_vector q
WHERE d.embedding <=> q.vec < 0.7
ORDER BY d.metadata->>'category', d.embedding <=> q.vec
LIMIT 30;

-- Results:
-- id |        title         | category | distance | rank_in_category
-- ----+---------------------+----------+----------+------------------
-- 45 | Data Analysis Guide  | tutorial |  0.12345 |                1
-- 52 | Statistics Basics    | tutorial |  0.23456 |                2
-- 67 | ML Fundamentals      | research |  0.34567 |                1
-- (30 rows)

```

**Example 5: Vector similarity with subqueries**

```sql
-- Find documents similar to a specific document
SELECT 
    d2.id,
    d2.title,
    d1.embedding <=> d2.embedding AS similarity_distance,
    1 - (d1.embedding <=> d2.embedding) AS similarity_score
FROM documents d1
CROSS JOIN documents d2
WHERE d1.id = 5  -- Reference document
  AND d2.id != 5  -- Exclude self
  AND d1.embedding <=> d2.embedding < 0.5
ORDER BY d1.embedding <=> d2.embedding
LIMIT 10;

-- Results:
-- id |         title          | similarity_distance | similarity_score
-- ----+------------------------+---------------------+------------------
-- 12 | Related Topic A        |             0.12345 |          0.87655
-- 18 | Related Topic B        |             0.23456 |          0.76544
-- (10 rows)

```

**Example 6: Vector operations in CTEs**

```sql
-- Complex query using Common Table Expressions
WITH 
    query_embedding AS (
        SELECT embed_text('PostgreSQL vector search') AS vec
    ),
    similar_docs AS (
        SELECT 
            d.id,
            d.title,
            d.embedding <=> q.vec AS distance
        FROM documents d, query_embedding q
        WHERE d.embedding <=> q.vec < 0.6
    ),
    ranked_docs AS (
        SELECT 
            id,
            title,
            distance,
            ROW_NUMBER() OVER (ORDER BY distance) AS rank
        FROM similar_docs
    )
SELECT 
    r.id,
    r.title,
    r.distance,
    r.rank,
    d.metadata->>'category' AS category
FROM ranked_docs r
JOIN documents d ON r.id = d.id
WHERE r.rank <= 10
ORDER BY r.rank;

-- Results:
-- id |          title           | distance | rank | category
-- ----+--------------------------+----------+------+----------
-- 25 | Vector Search Tutorial   |  0.12345 |    1 | tutorial
-- 30 | PostgreSQL Guide         |  0.23456 |    2 | guide
-- (10 rows)

```

Multi-vector search finds documents similar to multiple concepts simultaneously. JOIN operations combine vector similarity with relational data from multiple tables. Time-based filtering finds recent similar content. Window functions rank within categories. Subqueries find documents similar to specific references. CTEs support complex multi-step query logic. Standalone vector databases do not support this.

## Migration from Standalone Vector Databases

Migration requires data export, schema design, index creation, and query translation. Export vectors and metadata from the source system. Create PostgreSQL tables with vector columns and relational metadata. Build HNSW or IVFFlat indexes. Convert vector database queries to SQL with distance operators.

The following example shows migration steps:

```sql
-- Step 1: Create target table matching source schema
CREATE TABLE migrated_documents (
    id serial PRIMARY KEY,
    original_id text UNIQUE,
    title text,
    content text,
    embedding vector(768),
    metadata jsonb,
    migrated_at timestamp DEFAULT CURRENT_TIMESTAMP
);

-- Step 2: Create index for similarity search
CREATE INDEX idx_migrated_embedding ON migrated_documents 
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- Step 3: Import data (example using COPY)
-- COPY migrated_documents (original_id, title, content, embedding, metadata)
-- FROM '/path/to/exported/data.csv' WITH (FORMAT csv);

-- Step 4: Verify migration
SELECT 
    COUNT(*) AS total_documents,
    COUNT(DISTINCT vector_dims(embedding)) AS dimension_count,
    MIN(migrated_at) AS first_migration,
    MAX(migrated_at) AS last_migration
FROM migrated_documents;

```

Verify migration by counting documents, verifying vector dimensions, and checking timestamps. Migration enables gradual transition while maintaining data consistency.

## Scaling and Performance Optimization

Scaling strategies support growth from thousands to billions of vectors. Horizontal scaling distributes vectors across multiple instances using partitioning. Vertical scaling increases hardware resources on single instances. Hybrid scaling combines both approaches. Index optimization tunes parameters for specific workloads.

The following example shows horizontal scaling using table partitioning:

```sql
-- Partitioned table for horizontal scaling
CREATE TABLE documents_partitioned (
    id serial,
    title text,
    embedding vector(768),
    created_at timestamp,
    metadata jsonb
) PARTITION BY RANGE (created_at);

-- Create partitions for different time periods
CREATE TABLE documents_2024_q1 PARTITION OF documents_partitioned
    FOR VALUES FROM ('2024-01-01') TO ('2024-04-01');
CREATE TABLE documents_2024_q2 PARTITION OF documents_partitioned
    FOR VALUES FROM ('2024-04-01') TO ('2024-07-01');
CREATE TABLE documents_2024_q3 PARTITION OF documents_partitioned
    FOR VALUES FROM ('2024-07-01') TO ('2024-10-01');
CREATE TABLE documents_2024_q4 PARTITION OF documents_partitioned
    FOR VALUES FROM ('2024-10-01') TO ('2025-01-01');

-- Create indexes on each partition
CREATE INDEX idx_documents_q1_embedding ON documents_2024_q1 
USING hnsw (embedding vector_cosine_ops);
CREATE INDEX idx_documents_q2_embedding ON documents_2024_q2 
USING hnsw (embedding vector_cosine_ops);
-- ... additional partition indexes

```

Partitioning limits searches to relevant partitions. This reduces index size per partition and improves query performance. Queries automatically prune irrelevant partitions. Each partition can have independent indexes optimized for its data characteristics.

Performance optimization techniques improve query latency and throughput:

```sql
-- Optimize query with appropriate index hints
SET enable_seqscan = off;  -- Force index usage
SET work_mem = '256MB';    -- Increase work memory for sorting

-- Optimized similarity search query
WITH query_vector AS (
    SELECT embed_text('search query') AS vec
)
SELECT 
    d.id,
    d.title,
    d.embedding <=> q.vec AS distance
FROM documents d, query_vector q
WHERE d.embedding <=> q.vec < 0.5  -- Early filtering
ORDER BY d.embedding <=> q.vec
LIMIT 10;  -- Limit result set

-- Reset settings
RESET enable_seqscan;
RESET work_mem;

```

Query optimization techniques include forcing index usage, increasing work memory for sorting, applying early distance filtering, and limiting result sets. These optimizations improve query performance by 2-5x.

## Production Deployment

Production deployment requires hardware selection, index configuration, query optimization, and monitoring. CPU-only systems suit datasets up to 10 million vectors. Single GPU systems suit 10-100 million vectors. Multi-GPU systems suit billion-scale datasets. Index configuration balances query performance with construction time and memory usage. Query optimization ensures efficient execution plans. Monitoring tracks query latency, throughput, index usage, and resource utilization.

The following example shows production monitoring queries:

```sql
-- Monitor index usage and performance
SELECT 
    schemaname,
    tablename,
    indexname,
    idx_scan AS index_scans,
    idx_tup_read AS tuples_read,
    idx_tup_fetch AS tuples_fetched
FROM pg_stat_user_indexes
WHERE indexname LIKE '%embedding%'
ORDER BY idx_scan DESC;

-- Monitor query performance
SELECT 
    query,
    calls,
    total_exec_time,
    mean_exec_time,
    max_exec_time
FROM pg_stat_statements
WHERE query LIKE '%<=>%'
ORDER BY mean_exec_time DESC
LIMIT 10;

```

Production monitoring enables identification of performance bottlenecks, index usage patterns, and optimization opportunities. Index usage statistics reveal which indexes provide value and which may be unnecessary. Query performance statistics identify slow queries requiring optimization. Regular monitoring ensures PostgreSQL vector database maintains acceptable performance as datasets grow and query patterns evolve.

## Use Cases and Applications

PostgreSQL serves semantic search, recommendation systems, and AI applications. Semantic search finds documents by meaning. Recommendation systems identify similar items by computing vector similarity. RAG systems retrieve context for language models. Image search finds visually similar images. Anomaly detection identifies outliers in high-dimensional spaces.

The following example shows a semantic search application:

```sql
-- Semantic search for technical documentation
WITH search_query AS (
    SELECT embed_text('database performance optimization') AS query_vec
)
SELECT 
    d.id,
    d.title,
    d.content,
    d.embedding <=> s.query_vec AS similarity_score,
    d.metadata->>'category' AS category,
    d.metadata->>'author' AS author
FROM documents d, search_query s
WHERE d.metadata->>'type' = 'technical'
  AND d.embedding <=> s.query_vec < 0.5
ORDER BY d.embedding <=> s.query_vec
LIMIT 10;

```

The semantic search query finds technical documents similar to the search query by computing cosine distance. Results rank by similarity score. Lower scores indicate higher relevance. The query combines vector similarity with relational filters.

Recommendation systems use vector similarity to identify similar items:

```sql
-- Product recommendation based on user preferences
WITH user_profile AS (
    SELECT embedding FROM user_profiles WHERE user_id = 123
)
SELECT 
    p.id,
    p.name,
    p.description,
    p.price,
    p.embedding <=> u.embedding AS similarity
FROM products p, user_profile u
WHERE p.category = 'electronics'
  AND p.stock_quantity > 0
ORDER BY p.embedding <=> u.embedding
LIMIT 20;

```

The recommendation query finds products similar to a user profile by comparing embeddings. Results combine vector similarity with inventory availability and category filters.

## Advanced Features

PostgreSQL provides multi-vector search, hybrid search, temporal search, faceted search, and batch operations. Multi-vector search compares multiple query vectors simultaneously. Hybrid search combines vector similarity with full-text search. Temporal search combines time-based filtering with vector similarity. Faceted search filters by multiple metadata dimensions. Batch operations process multiple queries simultaneously.

The following example shows hybrid search combining vector similarity with full-text search:

```sql
-- Hybrid search: vector similarity + full-text search
WITH query_vector AS (
    SELECT embed_text('machine learning algorithms') AS vec
)
SELECT 
    d.id,
    d.title,
    d.content,
    (d.embedding <=> q.vec) * 0.7 + 
    (1.0 - ts_rank(to_tsvector('english', d.content), 
                   to_tsquery('english', 'machine & learning'))) * 0.3 AS combined_score
FROM documents d, query_vector q
WHERE d.content @@ to_tsquery('english', 'machine | learning | algorithm')
  AND d.embedding <=> q.vec < 0.6
ORDER BY combined_score
LIMIT 20;

```

The hybrid search query combines vector similarity with full-text search. It uses 70% weight for vector similarity and 30% weight for full-text search. This combines semantic meaning and keyword matching.

Multi-vector search enables complex similarity queries:

```sql
-- Multi-vector search: find items similar to multiple queries
WITH queries AS (
    SELECT 
        embed_text('database') AS vec1,
        embed_text('performance') AS vec2,
        embed_text('optimization') AS vec3
)
SELECT 
    d.id,
    d.title,
    (d.embedding <=> q.vec1 + 
     d.embedding <=> q.vec2 + 
     d.embedding <=> q.vec3) / 3.0 AS avg_distance
FROM documents d, queries q
ORDER BY avg_distance
LIMIT 10;

```

The multi-vector search computes average distance to multiple query vectors. It finds documents similar to all queries simultaneously.

## Best Practices

Best practices ensure optimal performance, reliability, and maintainability. Use fixed dimensions in vector columns. Select indexes based on workload. Optimize queries with appropriate distance metrics. Monitor performance metrics.

Deployment guidelines:

1. Dimension Consistency: Specify fixed dimensions in vector column definitions.

2. Index Selection: Use HNSW for production workloads requiring under 10ms latency. Use IVFFlat for frequently updated datasets. Use DiskANN for billion-scale storage.

3. Query Optimization: Use appropriate distance metrics. Limit result sets with LIMIT clauses. Combine vector similarity with relational filters.

4. Monitoring: Track query latency, throughput, index usage, and resource utilization.

5. Backup Strategy: Use standard PostgreSQL backup tools for vector data.

6. Capacity Planning: Monitor index sizes, query performance, and resource utilization. Scale hardware as needed.

## Conclusion

PostgreSQL with NeuronDB provides vector search capabilities with ACID transactions and SQL interface. It eliminates separate vector database systems. It reduces operational complexity. It enables unified data management.

PostgreSQL provides ACID transactions for vector data. It uses SQL for queries. It stores relational and vector data together. It combines vector similarity with relational filters. It uses standard PostgreSQL monitoring and backup tools.

Use PostgreSQL as a vector database for:
- Applications requiring ACID guarantees for vector data
- Complex queries combining vector similarity with relational filters
- Unified data management reducing operational complexity
- Production deployments requiring reliability and standard tooling
- Teams with existing PostgreSQL expertise

## Related Resources

[NeuronDB Documentation](https://neurondb.ai/docs) - Complete NeuronDB reference

[Vectors in PostgreSQL Guide](/blog/neurondb-vectors) - Detailed vector operations guide

[Semantic Search Guide](/blog/neurondb-semantic-search-guide) - Learn semantic search implementation

[RAG Guide](/blog/rag-complete-guide) - Learn RAG with PostgreSQL vector database

## Support

For questions, issues, or commercial support, contact [support@neurondb.ai](mailto:support@neurondb.ai)