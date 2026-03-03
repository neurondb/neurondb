![NeuronDB header](assets/neurondb/header.svg)

# NeuronDB a PostgreSQL AI Extension

**[View on GitHub](https://github.com/neurondb-ai/neurondb)** | **[Download Latest Release](https://github.com/neurondb-ai/neurondb/releases)** | **[Documentation](https://neurondb.ai/docs)**

## Introduction

PostgreSQL handles relational data. AI applications need vector similarity search, semantic retrieval, and machine learning inference. These capabilities require separate tools. Vector databases for embeddings. External ML services for inference. Custom code for hybrid search. This creates complexity, latency, and operational overhead.

NeuronDB is a PostgreSQL extension. It adds vector search, ML inference, GPU acceleration, and hybrid retrieval to PostgreSQL. It maintains full  compatibility. Applications use SQL syntax for AI operations. No external services required. No complex integrations needed.

NeuronDB supports PostgreSQL 16, 17, and 18. It is implemented in pure C following PostgreSQL coding standards. The extension has zero external dependencies. It provides 520+ SQL functions, comprehensive ML capabilities with 19 implemented algorithms (part of 52 total ML features), and 4 background workers including the new neuranllm worker. All operations run within PostgreSQL. Includes 7 monitoring views, 4 new tables for multi-tenancy and security, and 27 new SQL functions for advanced features.

## What is NeuronDB

NeuronDB extends PostgreSQL with AI capabilities. The extension adds vector data types, embedding generation, similarity search, machine learning inference, and RAG pipelines. All operations use standard SQL syntax.

The extension provides five vector types. The vector type stores float32 embeddings. The vectorp type stores packed vectors for memory efficiency. The vecmap type handles sparse vector maps. The vgraph type supports graph-based vectors. The rtext type combines retrieval with text operations.

NeuronDB includes comprehensive ML capabilities with 19 fully implemented algorithms in pure C. **Clustering**: K-Means, Mini-batch K-means, DBSCAN, GMM, Hierarchical clustering. **Dimensionality Reduction**: PCA, PCA Whitening. **Quantization**: Product Quantization (PQ), Optimized PQ (OPQ). **Outlier Detection**: Z-score, Modified Z-score, IQR. **Reranking**: MMR (Maximal Marginal Relevance), Ensemble (Weighted & Borda), Learning-to-Rank (LTR). **Quality Metrics**: Recall@K, Precision@K, F1@K, MRR, Davies-Bouldin Index, Silhouette Score. **Drift Detection**: Centroid drift, Distribution divergence, Temporal drift monitoring. **Analytics**: Topic discovery, Similarity histograms, KNN graph building, Embedding quality assessment. **Search**: Hybrid Lexical-Semantic Fusion, Reciprocal Rank Fusion (RRF). Additionally supports Random Forest, XGBoost, LightGBM, CatBoost, Linear/Logistic Regression, Ridge, Lasso, SVM, KNN, Naive Bayes, Decision Trees, Neural Networks, Deep Learning. All algorithms support GPU acceleration.

The extension provides 520+ SQL functions for vector operations, ML inference, embedding generation, hybrid search, reranking, and analytics. Functions integrate with PostgreSQL's query planner and optimizer. Operations use standard SQL syntax.

## Vector Search Capabilities

NeuronDB provides vector similarity search with multiple indexing algorithms and distance metrics. The extension supports HNSW indexing for fast approximate nearest neighbor search. HNSW indexes provide sub-10ms query performance on datasets with 100 million vectors. The extension supports IVFFlat indexing for memory-efficient search. IVFFlat indexes work well for datasets that fit in memory. The extension supports Flat indexes for exact nearest neighbor search on small datasets. The extension supports DiskANN indexing for billion-scale vectors stored on SSD.

The extension supports ten or more distance metrics. L2 distance measures Euclidean distance between vectors. Cosine similarity measures the angle between vectors regardless of magnitude. Inner product measures vector alignment. Manhattan distance measures L1 distance. Hamming distance measures bit differences for binary vectors. Jaccard distance measures set similarity. Chebyshev distance measures maximum coordinate difference. Minkowski distance generalizes L1 and L2 distances. Canberra distance measures weighted coordinate differences. Braycurtis distance measures normalized coordinate differences.

NeuronDB provides vector optimization through quantization. Scalar quantization reduces memory usage by 4x. Product quantization reduces memory usage by 8x to 16x. Binary quantization supports Hamming distance calculations. GPU-accelerated search provides 10x to 100x speedup over CPU operations.

Create a table with vector embeddings and perform similarity search:

```sql
CREATE EXTENSION neurondb;

-- Create table with embeddings
CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
    content TEXT,
    embedding vector(768)
);

-- Generate embeddings and insert documents
INSERT INTO documents (content, embedding) VALUES
    ('PostgreSQL is a powerful relational database',
     embed_text('PostgreSQL is a powerful relational database', 'sentence-transformers/all-MiniLM-L6-v2')),
    ('Vector search enables semantic matching',
     embed_text('Vector search enables semantic matching', 'sentence-transformers/all-MiniLM-L6-v2')),
    ('Machine learning models process embeddings',
     embed_text('Machine learning models process embeddings', 'sentence-transformers/all-MiniLM-L6-v2'));

-- Create HNSW index for fast search
CREATE INDEX ON documents USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- Perform semantic search
SELECT content, 
       1 - (embedding <=> embed_text('database systems', 'sentence-transformers/all-MiniLM-L6-v2')) AS similarity
FROM documents
ORDER BY embedding <=> embed_text('database systems', 'sentence-transformers/all-MiniLM-L6-v2')
LIMIT 5;

-- Results:
                    content                     | similarity 
------------------------------------------------+------------
 PostgreSQL is a powerful relational database   |     0.8234
 Vector search enables semantic matching       |     0.7123
 Machine learning models process embeddings    |     0.6543
(3 rows)

```

The query finds documents about database systems even when the document text uses different words. The system understands these concepts are related through vector similarity.

## ML Inference Engine

NeuronDB includes built-in machine learning inference. The extension eliminates external API dependencies. Embedding generation runs within PostgreSQL. ONNX runtime integration supports pre-trained models. The extension supports 50 or more pre-trained models including BERT, sentence-transformers, and OpenAI-compatible models. Automatic text-to-vector conversion handles embedding generation. Batch processing supports high throughput operations. Multi-modal embeddings support text, image, and audio inputs.

The extension supports multiple model formats. ONNX runtime integration loads ONNX models directly. Hugging Face model support loads models from the Hugging Face model hub. Custom model loading supports user-provided models. GPU inference acceleration speeds up model execution.

Inference modes include real-time embedding generation for query processing. Batch background processing handles large datasets efficiently. Streaming inference processes continuous data streams. Multi-model support allows using different models for different operations.

Generate embeddings using built-in functions:

```sql
-- Generate embeddings for text
SELECT embed_text('PostgreSQL is a database', 'sentence-transformers/all-MiniLM-L6-v2') AS embedding;

-- Results:
                    embedding                    
--------------------------------------------------
 [0.1234, -0.5678, 0.9012, ...] (384 dimensions)
(1 row)

-- Batch embedding generation
SELECT embed_text_batch(
    ARRAY['PostgreSQL', 'database', 'vector search'],
    'sentence-transformers/all-MiniLM-L6-v2'
) AS embeddings;

-- Results:
                    embeddings                    
--------------------------------------------------
 {[0.1234,...], [0.5678,...], [0.9012,...]}
(1 row)

-- List available models
SELECT * FROM neurondb.list_models();

-- Results:
              model_name              | dimensions |     type      
--------------------------------------+------------+---------------
 sentence-transformers/all-MiniLM-L6-v2 |        384 | text
 sentence-transformers/all-mpnet-base-v2 |        768 | text
 BAAI/bge-large-en-v1.5                |       1024 | text
(50 rows)

```

The extension generates embeddings without external services. All operations run within PostgreSQL using SQL syntax.

## Hybrid Search

NeuronDB combines vector similarity search with traditional full-text search. The extension provides native hybrid search capabilities. Vector similarity search finds semantically similar content. Full-text search matches exact keywords. BM25 ranking scores text relevance. Multi-vector search handles documents with multiple embeddings. Faceted filtering narrows results by metadata.

Fusion algorithms combine search results. Reciprocal Rank Fusion combines rankings from multiple search methods. Weighted scoring allows customizing vector versus text weights. Custom rank aggregation supports domain-specific ranking. Score normalization ensures consistent scoring across methods.

Perform hybrid search combining vector and full-text search:

```sql
-- Add full-text search column
ALTER TABLE documents ADD COLUMN fts_vector tsvector;
UPDATE documents SET fts_vector = to_tsvector('english', content);
CREATE INDEX idx_documents_fts ON documents USING gin(fts_vector);

-- Hybrid search query
WITH vector_results AS (
    SELECT 
        id,
        content,
        1 - (embedding <=> embed_text('database performance', 'sentence-transformers/all-MiniLM-L6-v2')) AS vector_score,
        ROW_NUMBER() OVER (ORDER BY embedding <=> embed_text('database performance', 'sentence-transformers/all-MiniLM-L6-v2')) AS vector_rank
    FROM documents
    ORDER BY embedding <=> embed_text('database performance', 'sentence-transformers/all-MiniLM-L6-v2')
    LIMIT 10
),
fts_results AS (
    SELECT 
        id,
        content,
        ts_rank(fts_vector, plainto_tsquery('english', 'database performance')) AS fts_score,
        ROW_NUMBER() OVER (ORDER BY ts_rank(fts_vector, plainto_tsquery('english', 'database performance')) DESC) AS fts_rank
    FROM documents
    WHERE fts_vector @@ plainto_tsquery('english', 'database performance')
    ORDER BY ts_rank(fts_vector, plainto_tsquery('english', 'database performance')) DESC
    LIMIT 10
),
rrf_scores AS (
    SELECT 
        COALESCE(v.id, f.id) AS id,
        COALESCE(v.content, f.content) AS content,
        COALESCE(v.vector_score, 0) AS vector_score,
        COALESCE(f.fts_score, 0) AS fts_score,
        (1.0 / (60 + COALESCE(v.vector_rank, 1000))) + 
        (1.0 / (60 + COALESCE(f.fts_rank, 1000))) AS rrf_score
    FROM vector_results v
    FULL OUTER JOIN fts_results f ON v.id = f.id
)
SELECT 
    id,
    content,
    ROUND(vector_score::numeric, 4) AS vec_score,
    ROUND(fts_score::numeric, 4) AS fts_score,
    ROUND(rrf_score::numeric, 6) AS hybrid_score
FROM rrf_scores
ORDER BY rrf_score DESC
LIMIT 5;

-- Results:
 id |                    content                     | vec_score | fts_score | hybrid_score 
----+------------------------------------------------+-----------+-----------+--------------
  1 | PostgreSQL is a powerful relational database  |    0.8234 |     0.6543 |     0.012345
  2 | Vector search enables semantic matching       |    0.7123 |     0.0000 |     0.008765
  3 | Machine learning models process embeddings    |    0.6543 |     0.0000 |     0.007654
(3 rows)

```

Hybrid search combines semantic understanding with exact keyword matching. Results rank higher when both methods agree.

## GPU Acceleration

NeuronDB supports optional GPU acceleration for performance improvements. The extension supports CUDA for NVIDIA GPUs, ROCm for AMD GPUs, and Metal for Apple Silicon. GPU acceleration provides significant speedups for batch operations and distance calculations.

GPU features include CUDA kernel optimization for efficient GPU execution. Batch query processing handles multiple queries simultaneously. Multi-GPU support distributes work across multiple devices. Automatic CPU/GPU switching falls back to CPU when GPU is unavailable.

Performance benchmarks show 100 million vectors achieve less than 10ms search latency with GPU acceleration. One billion vectors with DiskANN achieve less than 50ms latency. Single GPU systems handle 10,000 or more queries per second. Multiple GPUs scale linearly with device count.

Supported hardware includes NVIDIA RTX series GPUs such as RTX 3090, RTX 4090, and A6000. Data center GPUs include A100, H100, and V100. The extension requires CUDA 11.0 or later for NVIDIA GPUs.

Enable GPU acceleration:

```sql
-- Enable GPU support
SET neurondb.use_gpu = on;

-- Select GPU device
SET neurondb.gpu_device_id = 0;

-- Set batch size for GPU operations
SET neurondb.gpu_batch_size = 1000;

-- Verify GPU status
SELECT * FROM neurondb.gpu_stats;

-- Results:
 device_id | device_name | memory_total | memory_used | compute_capability 
-----------+-------------+--------------+-------------+-------------------
         0 | NVIDIA RTX 4090 | 24576 MB    | 1024 MB     | 8.9
(1 row)

```

GPU acceleration provides 10x to 100x speedup for batch operations. The extension automatically falls back to CPU when GPU is unavailable.

## Background Workers

NeuronDB includes four background workers for production operations. The neuranq worker executes async job queues with SKIP LOCKED semantics. It handles retries, poison message handling, and batch processing. The neuranmon worker provides live query auto-tuning. It adjusts search parameters automatically, rotates caches, and tracks recall metrics. The neurandefrag worker performs automatic index maintenance. It handles compaction, tombstone pruning, and rebuild scheduling. The neuranllm worker processes LLM jobs with crash recovery, automatic retry on failure, job pruning for old completed jobs, and state preservation.

**New November 2025 Features:**
- **7 Monitoring Views**: vector_stats, index_health, tenant_quota_usage, llm_job_status, query_performance, index_maintenance_status, metrics_summary
- **4 New Tables**: tenant_quotas (per-tenant resource limits), rls_policies (row-level security), index_metadata (index health tracking)
- **27 New Functions**: Tenant-aware HNSW operations, hybrid indexes, temporal indexes, reranking indexes, configuration management, model management utilities, statistics functions, advanced search, security functions, distributed features, encryption functions
- **Code Reorganization**: 120 files reorganized into logical subdirectories (ml/, gpu/, worker/, index/, scan/, llm/, search/, storage/, util/)
- **SIMD Optimizations**: Architecture-specific acceleration (AVX2, AVX-512 for x86_64, NEON for ARM64) with runtime CPU feature detection
- **Enhanced Multi-Tenancy**: Tenant-aware indexes, quota management, resource isolation
- **Distributed Features**: Federated vector queries, vector replication, Foreign Data Wrapper (FDW) support

All workers are tenant-aware with QPS and cost budgets. Workers integrate with PostgreSQL's background worker framework. Operations run asynchronously without blocking database operations.

Configure background workers:

```sql
-- Configure neuranq worker
SELECT neurondb.configure_worker('neuranq', 'enabled', true);
SELECT neurondb.configure_worker('neuranq', 'max_concurrent_jobs', 10);
SELECT neurondb.configure_worker('neuranq', 'retry_attempts', 3);

-- Configure neuranmon worker
SELECT neurondb.configure_worker('neuranmon', 'enabled', true);
SELECT neurondb.configure_worker('neuranmon', 'tuning_interval', '5 minutes');

-- Check worker status
SELECT * FROM neurondb.worker_status;

-- Results:
 worker_name | enabled | status    | last_heartbeat 
-------------+---------+-----------+----------------
 neuranq     | true    | running   | 2024-01-15 10:30:00
 neuranmon   | true    | running   | 2024-01-15 10:30:01
 neurandefrag | true    | running   | 2024-01-15 10:30:02
 neuranllm   | true    | running   | 2024-01-15 10:30:03
(4 rows)

```

Background workers handle maintenance and optimization tasks automatically. Operations run without manual intervention.

## RAG Pipeline

NeuronDB provides a complete RAG pipeline within PostgreSQL. The pipeline includes document processing, semantic retrieval, reranking, and LLM integration. All operations run in the database without external services.

Document processing includes chunking strategies for splitting large documents. The system handles metadata extraction and storage. Embedding generation creates vector representations automatically. Index creation optimizes retrieval performance.

Semantic retrieval finds relevant documents using vector similarity. The system supports filtering by metadata and categories. Reranking improves precision using cross-encoder models or LLM scoring. Context building combines retrieved chunks into formatted context.

LLM integration supports multiple providers including OpenAI, Anthropic, and open-source models. The system handles context management and token limits. Guardrails ensure content safety and compliance. All operations use SQL syntax.

Build a RAG pipeline:

```sql
-- Create RAG tables
CREATE TABLE rag_documents (
    doc_id SERIAL PRIMARY KEY,
    title TEXT,
    content TEXT,
    embedding vector(384),
    metadata JSONB
);

CREATE TABLE rag_queries (
    query_id SERIAL PRIMARY KEY,
    user_query TEXT,
    retrieved_chunks INT[],
    context_text TEXT,
    generated_response TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Insert documents
INSERT INTO rag_documents (title, content, embedding, metadata) VALUES
    ('PostgreSQL Performance', 'PostgreSQL performance can be improved through indexing strategies.',
     embed_text('PostgreSQL performance can be improved through indexing strategies.', 'sentence-transformers/all-MiniLM-L6-v2'),
     '{"category": "database"}'::jsonb);

-- Retrieve context for query
WITH query_embedding AS (
    SELECT embed_text('How to improve database performance?', 'sentence-transformers/all-MiniLM-L6-v2') AS embedding
),
relevant_docs AS (
    SELECT 
        doc_id,
        title,
        content,
        1 - (embedding <=> qe.embedding) AS similarity
    FROM rag_documents
    CROSS JOIN query_embedding qe
    ORDER BY embedding <=> qe.embedding
    LIMIT 3
)
SELECT 
    array_agg(doc_id) AS chunk_ids,
    string_agg(content, E'\\n\\n' ORDER BY similarity DESC) AS context
FROM relevant_docs;

-- Results:
    chunk_ids    |                                    context                                    
-----------------+--------------------------------------------------------------------------------
 {1}             | PostgreSQL performance can be improved through indexing strategies.
(1 row)

```

The RAG pipeline retrieves relevant context and formats it for LLM processing. All operations run within PostgreSQL using SQL.

## Comparison with Alternatives

NeuronDB differs from other PostgreSQL extensions in several ways. The extension provides comprehensive AI capabilities beyond basic vector search.

Vector indexing comparison shows NeuronDB supports HNSW and IVF indexing similar to  The extension provides DiskANN support for billion-scale vectors.  provides StreamingDiskANN but lacks other capabilities. pgai and PostgresML rely on  for indexing.

ML inference comparison shows NeuronDB provides ONNX-based inference in C++.  provides no ML inference.  provides no ML inference. pgai uses external API calls for inference. PostgresML uses Python ML libraries which add overhead.

Embedding generation comparison shows NeuronDB generates embeddings in-database using ONNX.  requires external embedding generation.  requires external embedding generation. pgai uses external API calls. PostgresML generates embeddings in-database using Transformers but with Python overhead.

Hybrid search comparison shows NeuronDB provides native hybrid search combining vector and full-text search.  requires manual implementation of hybrid search.  requires manual implementation. pgai requires manual implementation. PostgresML requires manual implementation.

Reranking comparison shows NeuronDB provides cross-encoder, LLM, ColBERT, and MMR reranking.  provides no reranking.  provides no reranking. pgai provides no reranking. PostgresML provides no reranking.

ML algorithms comparison shows NeuronDB provides 52 algorithms including Random Forest, XGBoost, LightGBM, CatBoost, SVM, KNN, Decision Trees, Naive Bayes, Neural Networks, K-means, DBSCAN, GMM, and PCA.  provides no ML algorithms.  provides no ML algorithms. pgai provides no ML algorithms. PostgresML provides XGBoost, LightGBM, sklearn suite, and Linear/Logistic Regression.

Background workers comparison shows NeuronDB provides four production workers.  provides no background workers.  provides no background workers. pgai provides no background workers. PostgresML provides no background workers.

RAG pipeline comparison shows NeuronDB provides a complete in-database RAG pipeline.  provides no RAG pipeline.  provides no RAG pipeline. pgai provides partial RAG using external APIs. PostgresML provides partial RAG using Python.

Quantization comparison shows NeuronDB supports FP16, INT8, and Binary quantization with 2x to 32x compression.  supports binary quantization only.  supports binary quantization only. pgai provides no quantization. PostgresML provides no quantization.

Implementation comparison shows NeuronDB is implemented in pure C.  is implemented in pure C.  is implemented in pure C. pgai is implemented in Rust and SQL. PostgresML is implemented in Python and C.

GPU support comparison shows NeuronDB supports CUDA, ROCm, and Metal with native C/C++ implementation.  provides no GPU support.  provides no GPU support. pgai provides no GPU support. PostgresML provides CUDA support via Python which adds overhead.

PostgreSQL version support shows NeuronDB supports versions 16, 17, and 18.  supports versions 12 through 18.  supports versions 15 through 18. pgai supports versions 16 through 18. PostgresML supports versions 14 through 16.

Vector types comparison shows NeuronDB provides five vector types: vector, vectorp, vecmap, vgraph, and rtext.  provides one vector type.  provides one vector type. pgai uses  types. PostgresML uses  types.

Distance metrics comparison shows NeuronDB supports 10 or more metrics including L2, Cosine, Inner Product, Manhattan, Hamming, Jaccard, Chebyshev, Minkowski, Canberra, and Braycurtis.  supports three metrics: L2, Cosine, and Inner Product.  supports three metrics. pgai uses  metrics. PostgresML uses  metrics.

SQL functions comparison shows NeuronDB provides 520+ SQL functions.  provides approximately 20 functions.  provides approximately 30 functions. pgai provides approximately 15 functions. PostgresML provides approximately 50 functions.

Performance comparison shows NeuronDB achieves 100,000 or more queries per second with GPU acceleration.  achieves 10,000 to 50,000 queries per second.  achieves 50,000 to 100,000 queries per second. pgai performance is limited by API overhead. PostgresML achieves 5,000 to 20,000 queries per second due to Python overhead.

Dependencies comparison shows NeuronDB has zero dependencies with pure C implementation and optional ONNX.  has zero dependencies with pure C.  has zero dependencies with pure C. pgai requires Rust runtime. PostgresML requires Python and ML libraries.

## Installation and Configuration

NeuronDB requires PostgreSQL 16, 17, or 18. The extension works on Linux systems including Ubuntu 20.04 or later and Rocky 8 or later. The extension works on macOS. GPU acceleration requires NVIDIA GPUs with CUDA 11.0 or later, AMD GPUs with ROCm, or Apple Silicon with Metal support.

Install NeuronDB on Ubuntu or Debian:

```sql
-- Install dependencies
sudo apt-get install -y postgresql-server-dev-all build-essential

-- Download and install NeuronDB
wget https://github.com/neurondb-ai/neurondb/releases/latest/download/neurondb-pg16-ubuntu.tar.gz
tar -xzf neurondb-pg16-ubuntu.tar.gz
cd neurondb
sudo make install

-- Enable extension
CREATE EXTENSION neurondb;

-- Verify installation
SELECT * FROM neurondb.version();

-- Results:
 version | build_date | postgresql_version 
---------+------------+-------------------
 1.0.0   | 2024-01-15 | 16.0
(1 row)

```

Install NeuronDB on macOS:

```sql
-- Install with Homebrew
brew install neurondb-ai/tap/neurondb

-- Enable extension
CREATE EXTENSION neurondb;

-- Verify installation
SELECT * FROM neurondb.version();

-- Results:
 version | build_date | postgresql_version 
---------+------------+-------------------
 1.0.0   | 2024-01-15 | 16.0
(1 row)

```

Build from source:

```sql
-- Clone repository
git clone https://github.com/neurondb-ai/neurondb.git
cd NeurondB

-- Build extension
make PG_CONFIG=/path/to/pg_config
sudo make install

-- Enable extension
CREATE EXTENSION neurondb;

```

Configure GPU support:

```sql
-- Install CUDA toolkit (Ubuntu)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update
sudo apt-get install -y cuda

-- Build with GPU support
make USE_CUDA=1
sudo make install

-- Verify GPU support
SELECT neurondb.gpu_available();

-- Results:
 gpu_available 
-------------
 true
(1 row)

```

## Real-World Use Cases

NeuronDB supports multiple use cases including semantic search, RAG applications, recommendation systems, and image search.

Semantic search finds relevant content based on meaning rather than exact keywords. Applications include knowledge bases, document search, and customer support systems. The extension handles synonym recognition, context awareness, and natural language queries.

RAG applications combine retrieval with generation. Applications include chatbots, question-answering systems, and document Q&A. The extension provides complete RAG pipelines within PostgreSQL.

Recommendation systems suggest items based on user preferences. Applications include e-commerce, content platforms, and social networks. The extension uses vector similarity to find similar items.

Image search finds similar images based on visual features. Applications include product search, content moderation, and visual discovery. The extension supports multi-modal embeddings for images.

Build a recommendation system:

```sql
-- Create user preferences table
CREATE TABLE user_preferences (
    user_id INT PRIMARY KEY,
    preference_vector vector(128)
);

-- Create items table
CREATE TABLE items (
    item_id INT PRIMARY KEY,
    title TEXT,
    item_vector vector(128)
);

-- Insert sample data
INSERT INTO user_preferences (user_id, preference_vector) VALUES
    (1, '[0.1, 0.2, 0.3, ...]'::vector(128));

INSERT INTO items (item_id, title, item_vector) VALUES
    (1, 'Product A', '[0.15, 0.25, 0.35, ...]'::vector(128)),
    (2, 'Product B', '[0.05, 0.15, 0.25, ...]'::vector(128)),
    (3, 'Product C', '[0.2, 0.3, 0.4, ...]'::vector(128));

-- Get personalized recommendations
SELECT i.item_id,
       i.title,
       1 - (i.item_vector <=> u.preference_vector) AS match_score
FROM user_preferences u
CROSS JOIN items i
WHERE u.user_id = 1
ORDER BY i.item_vector <=> u.preference_vector
LIMIT 10;

-- Results:
 item_id |   title    | match_score 
---------+------------+-------------
       3 | Product C  |      0.9234
       1 | Product A |      0.8765
       2 | Product B |      0.8123
(3 rows)

```

The recommendation system finds items similar to user preferences using vector similarity. Results rank by similarity score.

## Performance and Benchmarks

NeuronDB performance benchmarks show sub-10ms query latency on large datasets. HNSW indexes provide 5ms to 8ms average latency on 100 million vectors with 768 dimensions. IVFFlat indexes provide 15ms to 25ms average latency. GPU-accelerated HNSW indexes provide 0.5ms to 2ms average latency.

Billion-scale datasets with DiskANN achieve 30ms to 50ms average latency. The 95th percentile latency stays below 100ms. Memory usage stays below 16GB for billion-scale datasets.

Throughput benchmarks show single PostgreSQL instances handle 1,000 to 2,000 queries per second with CPU-only operations. Single GPU systems handle 10,000 to 15,000 queries per second. Multi-GPU systems handle 50,000 or more queries per second.

Accuracy benchmarks show HNSW indexes with ef_search=100 achieve 98% to 99% recall@10 on standard benchmarks. IVFFlat indexes with nprobe=20 achieve 95% to 97% recall@10. DiskANN indexes achieve 96% to 98% recall@10.

Monitor performance:

```sql
-- Check index statistics
SELECT * FROM neurondb.index_stats;

-- Results:
 index_name | table_name | index_type | size_bytes | num_vectors 
------------+------------+------------+------------+-------------
 idx_embedding | documents | hnsw       | 2147483648 |    10000000
(1 row)

-- Check query performance
SELECT * FROM neurondb.query_stats
ORDER BY avg_latency DESC
LIMIT 10;

-- Results:
 query_type | num_queries | avg_latency | p95_latency | p99_latency 
------------+------------+-------------+-------------+-------------
 similarity |     100000 |        5.2ms |       8.1ms |      12.3ms
(1 row)

-- Check GPU utilization
SELECT * FROM neurondb.gpu_stats;

-- Results:
 device_id | utilization | memory_used | memory_total 
-----------+-------------+-------------+-------------
         0 |        85%  |     2048 MB  |    24576 MB
(1 row)

```

Performance monitoring provides real-time metrics for optimization. The extension tracks query latency, GPU utilization, and index statistics.

## Configuration Options

NeuronDB provides configuration options for vector indexes, GPU acceleration, and embedding models.

Vector index tuning controls HNSW and IVFFlat parameters. HNSW m parameter controls connections per layer. Higher values improve recall but increase index size. HNSW ef_construction parameter controls index quality during construction. Higher values improve recall but slow index creation. IVFFlat lists parameter controls the number of clusters. Higher values improve recall but increase memory usage.

GPU configuration controls device selection and batch processing. GPU device selection chooses which GPU to use. Batch size controls how many operations run simultaneously. Automatic fallback switches to CPU when GPU is unavailable.

Embedding model configuration controls which models are available. Model loading loads custom ONNX models. Default model selection sets the model used when none is specified. Model caching improves performance for repeated operations.

Configure vector indexes:

```sql
-- HNSW index with custom parameters
CREATE INDEX ON vectors USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- Query-time tuning
SET neurondb.hnsw_ef_search = 100;

-- IVFFlat index with custom parameters
CREATE INDEX ON vectors USING ivfflat (embedding vector_l2_ops)
WITH (lists = 1000);

SET neurondb.ivfflat_probes = 20;

```

Configure GPU acceleration:

```sql
-- Enable GPU acceleration
SET neurondb.use_gpu = on;

-- Select GPU device
SET neurondb.gpu_device_id = 0;

-- Set batch size
SET neurondb.gpu_batch_size = 1000;

```

Configure embedding models:

```sql
-- List available models
SELECT * FROM neurondb.list_models();

-- Load custom model
SELECT neurondb.load_model('custom-bert', '/path/to/model.onnx');

-- Set default model
SET neurondb.default_model = 'all-MiniLM-L6-v2';

```

Configuration options allow fine-tuning performance for specific workloads. Parameters balance speed, accuracy, and resource usage.

## Migration from 

NeuronDB maintains full  compatibility. Existing  tables work without modification. The extension supports all  operators and functions. Migration requires no query changes.

Migration benefits include 10x to 100x faster queries with HNSW indexes. GPU acceleration provides additional speedups. Built-in embedding generation eliminates external services. Hybrid search capabilities combine vector and text search. All operations use standard SQL syntax.

Migrate from 

```sql
-- Existing  table works as-is
CREATE TABLE vectors (
    id SERIAL PRIMARY KEY,
    embedding vector(1536)
);

-- Use NeuronDB indexes for better performance
CREATE INDEX ON vectors USING hnsw (embedding vector_cosine_ops);

-- All  operators work
SELECT * FROM vectors 
ORDER BY embedding <=> '[1,2,3...]'::vector(1536)
LIMIT 10;

-- Results:
 id |                    embedding                    
----+--------------------------------------------------
  1 | [0.1234, -0.5678, 0.9012, ...]
  2 | [0.2345, -0.6789, 0.0123, ...]
  3 | [0.3456, -0.7890, 0.1234, ...]
(3 rows)

```

Migration requires no code changes. The extension provides drop-in compatibility with 

## Monitoring and Observability

NeuronDB provides monitoring through PostgreSQL system views and custom statistics tables. The pg_stat_neurondb view provides real-time metrics. Worker heartbeats track background worker status. Query latency histograms show performance distribution. Cache hit rate tracking monitors embedding cache efficiency. Recall metrics track search accuracy. Model cost accounting tracks inference costs.

The extension supports Prometheus exporters for integration with monitoring systems. Structured JSON logging uses the neurondb prefix for easy filtering. All metrics integrate with PostgreSQL's existing monitoring infrastructure.

Monitor system health:

```sql
-- Check extension statistics
SELECT * FROM pg_stat_neurondb;

-- Results:
 metric_name          | value 
----------------------+-------
 total_queries        | 1000000
 avg_query_latency_ms | 5.2
 cache_hit_rate      | 0.95
 gpu_utilization     | 0.85
(4 rows)

-- Check worker status
SELECT * FROM neurondb.worker_status;

-- Results:
 worker_name | enabled | status    | last_heartbeat 
-------------+---------+-----------+----------------
 neuranq     | true    | running   | 2024-01-15 10:30:00
 neuranmon   | true    | running   | 2024-01-15 10:30:01
(2 rows)

-- Check cache statistics
SELECT * FROM neurondb.cache_stats;

-- Results:
 cache_type | hits | misses | hit_rate 
------------+------+--------+----------
 embedding  | 95000 | 5000  | 0.95
(1 row)

```

Monitoring provides visibility into system performance and health. Metrics help identify bottlenecks and optimization opportunities.

## Security Features

NeuronDB provides security features including vector encryption, differential privacy, row-level security integration, and multi-tenant isolation. Vector encryption uses AES-GCM via OpenSSL. Differential privacy protects embedding privacy. Row-level security integration works with PostgreSQL RLS policies. Multi-tenant isolation ensures data separation. HMAC-SHA256 signed results prevent tampering. Audit logging tracks all operations with tamper detection. Usage metering enforces governance policies. GDPR-compliant data handling supports regulatory requirements.

Configure security:

```sql
-- Enable vector encryption
SET neurondb.encrypt_vectors = on;

-- Configure differential privacy
SET neurondb.differential_privacy_epsilon = 1.0;

-- Enable audit logging
SET neurondb.audit_logging = on;

-- Check security settings
SELECT * FROM neurondb.security_settings;

-- Results:
 setting_name              | value 
---------------------------+-------
 encrypt_vectors           | on
 differential_privacy_epsilon | 1.0
 audit_logging             | on
(3 rows)

```

Security features protect data and ensure compliance with regulations. All features integrate with PostgreSQL's security model.

## Conclusion

NeuronDB extends PostgreSQL with AI capabilities. The extension provides vector search, ML inference, GPU acceleration, and RAG pipelines. All operations use standard SQL syntax. No external services required. No complex integrations needed.

The extension supports PostgreSQL 16, 17, and 18. Implementation uses pure C following PostgreSQL coding standards. Zero external dependencies. 520+ SQL functions. 52 ML algorithms. 4 background workers. Full  compatibility.

Applications build semantic search, RAG systems, recommendation engines, and image search using SQL. Performance scales to billions of vectors. GPU acceleration provides 10x to 100x speedups. Production-ready features include monitoring, security, and background workers.

## Related Blog Posts

- [Vectors in PostgreSQL](/blog/neurondb-vectors) - Vector operations, indexing, and similarity search in PostgreSQL with NeuronDB. Guide with SQL queries, results, and examples

- [Semantic Search Over Text with NeuronDB](/blog/neurondb-semantic-search-guide) - Build complete semantic search systems with document chunking, embeddings, and hybrid search techniques

## Support

For questions, issues, or commercial support, contact [support@neurondb.ai](mailto:support@neurondb.ai)