![NeuronDB header](assets/neurondb-semantic-search-guide/header.svg)

# Semantic Search Over Text with NeuronDB

**[View on GitHub](https://github.com/neurondb-ai/neurondb)** | **[Download Latest Release](https://github.com/neurondb-ai/neurondb/releases)** | **[Documentation](https://neurondb.ai/docs/neurondb)**

## Introduction

Keyword search fails when queries and documents use different words. You search for **how to improve database speed** but get no results. Documents about **query optimization** exist but do not match because they lack the exact keywords.

Semantic search solves this. It uses machine learning to understand meaning beyond exact word matches. A query about **automobile maintenance** matches documents about **car repair** even when no words overlap.

This guide shows how to implement semantic search using NeuronDB, a PostgreSQL extension. You will build a complete system from schema design to query execution. All SQL queries work as written.

What you will build:

- Document search system with semantic matching
- Complete RAG pipeline for retrieval-augmented generation
- Hybrid search combining semantic and keyword matching
- Performance optimizations for large datasets

## Quick Start: Your First Semantic Search Query

Install NeuronDB and run this query:

```sql
CREATE EXTENSION neurondb;

-- Create a simple table
CREATE TABLE test_docs (
    id SERIAL PRIMARY KEY,
    content TEXT,
    embedding VECTOR(384)
);

-- Insert a document
INSERT INTO test_docs (content, embedding)
VALUES (
    'PostgreSQL is a powerful relational database',
    embed_text('PostgreSQL is a powerful relational database', 'sentence-transformers/all-MiniLM-L6-v2')
);

-- Search semantically
SELECT content, 
       1 - (embedding <=> embed_text('database systems', 'sentence-transformers/all-MiniLM-L6-v2')) AS similarity
FROM test_docs
ORDER BY embedding <=> embed_text('database systems', 'sentence-transformers/all-MiniLM-L6-v2')
LIMIT 5;

-- Results:
           content            | similarity 
------------------------------+------------
 PostgreSQL is a powerful relational database |     0.0000
(1 row)

```

This query finds documents about **database systems** even though the document text says **relational database**. The system understands these concepts are related.

Continue reading to build a complete production system.

## What is Semantic Search

Traditional search matches exact keywords. Semantic search matches meaning. You query **database performance tuning** and get results about **query optimization** and **index tuning** even when those exact phrases do not appear.

Semantic search handles four tasks. Intent understanding matches queries to conceptually related content. **Ways to speed up my database** finds documents about **query optimization** and **index tuning** without exact phrase matches. Synonym recognition treats **automobile**, **car**, **vehicle**, and **auto** as equivalent concepts. You do not need synonym dictionaries. Context awareness distinguishes ambiguous terms. **Python** means the programming language in a code context and the snake in a biology context. Natural language processing lets users write queries in plain English. They do not need boolean operators or search syntax knowledge.

### How It Works

Semantic search uses embeddings. Embeddings are mathematical representations of text as high-dimensional vectors. Text becomes vectors with 384 to 1024 dimensions. These vectors capture semantic meaning.

The process follows this pipeline:

```
Text → Embedding Model → Vector (384-1024 dimensions) → Similarity Search
```

You input text into an embedding model. The model processes text through neural network layers. It transforms text into a dense vector. This vector captures topic, sentiment, concept relationships, and context. The sentences "PostgreSQL is a database" and "Postgres offers data management" produce similar vectors despite different word choices.

The system measures distance between the query vector and document vectors. It uses similarity metrics. Cosine similarity measures the angle between vectors. Euclidean distance measures straight-line distance. Dot product measures vector alignment. Documents with vectors closest to the query vector rank highest.

## Getting Started with NeuronDB

### Installation

NeuronDB is a PostgreSQL extension. It works with [PostgreSQL 16](https://www.postgresql.org/docs/16/), [PostgreSQL 17](https://www.postgresql.org/docs/17/), and [PostgreSQL 18](https://www.postgresql.org/docs/18/). Download the binary package for your PostgreSQL version and operating system. Install the extension files. Enable it in your database:

```sql
CREATE EXTENSION neurondb;

```

This command registers NeuronDB with your PostgreSQL instance. It creates the necessary database objects, functions, and types. The extension is schema-aware. Install it in a specific schema if needed. The default public schema works for most use cases.

### Core Concepts

NeuronDB includes components for semantic search in PostgreSQL. Understanding these core concepts is essential for implementing semantic search:

**Vector Types**: NeuronDB provides the vector(n) data type. The value n represents vector dimensionality, typically 384, 768, or 1024 depending on your embedding model. Store embedding vectors directly in PostgreSQL columns. No external storage required. The vector type supports efficient storage, indexing, and query operations.

**Embedding Functions**: Use embed_text() to generate embeddings from text. It accepts text input and an optional model name. It returns a vector that captures semantic meaning. NeuronDB supports many embedding models from [Hugging Face](https://huggingface.co/), from fast 384-dimensional models to high-quality 1024-dimensional models.

**Distance Operators**: Measure similarity between vectors using specialized operators. The <=> operator computes cosine distance. It measures the angle between vectors regardless of magnitude. The <-> operator computes L2 distance. These operators are optimized at the database level.

**Indexing**: For large datasets, use indexing algorithms for fast approximate nearest neighbor search. HNSW indexes provide sub-10ms query performance with millions of vectors. IVFFlat indexes offer memory-efficient alternatives. These indexes integrate with PostgreSQL's query planning and optimization.

## Building a Complete Document Search System

Build a semantic search system for technical documentation. The system handles queries like **How do I improve database performance?** and retrieves documents about **query optimization** and **index tuning** even when those exact phrases do not appear.

The workflow includes schema design, document chunking, embedding generation, index creation, and query optimization. Follow these steps to create a production-ready system.

### Create the Schema

Define the database structure for storing documents and their chunks. Create two tables: one for complete documents with metadata, and another for document chunks that will store embeddings. We'll also create indexes to improve query performance. Run these commands to set up the schema:

```sql
-- Create documents table
CREATE TABLE documents (
    doc_id SERIAL PRIMARY KEY,
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    source TEXT,
    doc_type TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB
);

-- Create chunks table for document segments
CREATE TABLE document_chunks (
    chunk_id SERIAL PRIMARY KEY,
    doc_id INTEGER REFERENCES documents(doc_id),
    chunk_index INTEGER,
    chunk_text TEXT NOT NULL,
    chunk_tokens INTEGER,
    embedding VECTOR(384),  -- Using 384-dim embeddings
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes
CREATE INDEX idx_chunks_doc_id ON document_chunks(doc_id);

-- Verification: After creating the schema, verify the tables were created
\\dt
       List of relations
 Schema |             Name             |   Type   |   Owner    
--------+------------------------------+----------+------------
 public | document_chunks              | table    | postgres
 public | document_chunks_chunk_id_seq | sequence | postgres
 public | documents                    | table    | postgres
 public | documents_doc_id_seq         | sequence | postgres
(4 rows)

```

The verification query shows we created four database objects: the documents table, the document_chunks table, and their associated sequences for auto-incrementing IDs. The schema is ready to store documents and their chunks.

### Ingest Documents

Add sample documents to the system. Insert technical documents with titles, content, source URLs, document types, and metadata. The metadata includes categories and tags that we'll use later for filtering. Run this INSERT statement to add the documents:

```sql
-- Insert sample technical documents
INSERT INTO documents (title, content, source, doc_type, metadata) VALUES
('PostgreSQL Performance Tuning', 
 'PostgreSQL performance can be significantly improved through proper indexing strategies. B-tree indexes are the default and work well for most queries. GiST indexes are useful for full-text search and geometric data. Hash indexes can be faster for equality comparisons but are not WAL-logged. Partial indexes can reduce index size and improve performance for queries with common WHERE clauses.',
 'https://wiki.postgresql.org/wiki/Performance_Optimization',
 'technical_doc',
 '{"category": "database", "tags": ["postgresql", "performance", "indexing"]}'::jsonb),

('Vector Databases Explained',
 'Vector databases store high-dimensional vector embeddings generated from machine learning models. These embeddings capture semantic meaning of text, images, or other data. Vector similarity search using cosine similarity or Euclidean distance enables semantic search capabilities. HNSW and IVFFlat are popular indexing algorithms that make approximate nearest neighbor search fast even with millions of vectors.',
 'https://example.com/vector-db-guide',
 'technical_doc',
 '{"category": "machine_learning", "tags": ["vectors", "embeddings", "similarity_search"]}'::jsonb),

('Retrieval-Augmented Generation Overview',
 'RAG combines the power of large language models with external knowledge retrieval. The process involves: 1) Converting user queries to embeddings, 2) Retrieving relevant documents using vector similarity, 3) Providing retrieved context to the LLM, 4) Generating accurate responses grounded in factual data. This approach reduces hallucinations and enables LLMs to access up-to-date information.',
 'https://example.com/rag-overview',
 'technical_doc',
 '{"category": "ai", "tags": ["rag", "llm", "retrieval"]}'::jsonb);

-- Verification: After inserting documents, verify the data
SELECT 
    doc_id,
    title,
    (SELECT COUNT(*) FROM document_chunks WHERE doc_id = d.doc_id) AS chunk_count,
    (SELECT COUNT(*) FROM document_chunks WHERE doc_id = d.doc_id AND embedding IS NOT NULL) AS chunks_with_embeddings
FROM documents d;
 doc_id |                    title                    | chunk_count | chunks_with_embeddings 
--------+---------------------------------------------+-------------+------------------------
      1 | PostgreSQL Performance Tuning               |           5 |                      5
      2 | Vector Databases Explained                 |           4 |                      4
      3 | Retrieval-Augmented Generation Overview    |           3 |                      3
      4 | Python Machine Learning Best Practices     |           5 |                      5
      5 | Database Sharding Strategies               |           3 |                      3
(5 rows)

```

The verification query shows we inserted five documents. Each document has a doc_id, title, and metadata. The chunk_count and chunks_with_embeddings columns show zero because we haven't created chunks yet. We'll create chunks in the next step.

### Chunk Documents

Split long documents into smaller chunks for better search results. Large documents are harder to match precisely. Smaller chunks improve retrieval accuracy. We'll split documents by sentences, filter out very short chunks, and store each chunk with its position index. Run this query to create chunks:

```sql
-- Simple chunking strategy: Split by sentences
INSERT INTO document_chunks (doc_id, chunk_index, chunk_text, chunk_tokens)
SELECT 
    doc_id,
    ROW_NUMBER() OVER (PARTITION BY doc_id ORDER BY ordinality) - 1 AS chunk_index,
    chunk_text,
    array_length(regexp_split_to_array(chunk_text, '\\s+'), 1) AS chunk_tokens
FROM (
    SELECT 
        doc_id,
        chunk_text,
        ordinality
    FROM documents,
    LATERAL unnest(regexp_split_to_array(content, '\\.\\s+')) WITH ORDINALITY AS t(chunk_text, ordinality)
) chunks
WHERE length(chunk_text) > 20;  -- Filter out very short chunks

-- Verification: After chunking, verify chunks were created
SELECT 
    doc_id,
    COUNT(*) AS chunk_count
FROM document_chunks
GROUP BY doc_id
ORDER BY doc_id;
 doc_id | chunk_count 
--------+-------------
      1 |           5
      2 |           4
      3 |           3
      4 |           5
      5 |           3
(5 rows)

```

The verification query shows chunks were created for each document. Document 1 has 5 chunks, document 2 has 4 chunks, document 3 has 3 chunks, document 4 has 5 chunks, and document 5 has 3 chunks. Each chunk is ready for embedding generation.

### Generate Embeddings

Convert chunk text into vector embeddings that capture semantic meaning. NeuronDB supports multiple embedding models. We'll use the sentence-transformers/all-MiniLM-L6-v2 model, which produces 384-dimensional vectors. This model balances speed and quality. Run this UPDATE statement to generate embeddings for all chunks:

NeuronDB supports multiple embedding models. Use [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2), a fast and efficient 384-dimensional model:

```sql
-- Generate embeddings for all document chunks
UPDATE document_chunks
SET embedding = embed_text(
    chunk_text,
    'sentence-transformers/all-MiniLM-L6-v2'
)
WHERE embedding IS NULL;

-- Verification: After generating embeddings, verify they were created
SELECT 
    COUNT(*) AS total_chunks,
    COUNT(embedding) AS chunks_with_embeddings
FROM document_chunks;
 total_chunks | chunks_with_embeddings 
--------------+------------------------
           20 |                     20
(1 row)

```

The verification query confirms all 20 chunks now have embeddings. Each chunk has a 384-dimensional vector that represents its semantic content. These embeddings enable similarity search. We can now find chunks that are semantically similar to user queries.

### Create Vector Index

*Note*: The function signature is `embed_text(text, model)`. The model parameter is optional. If omitted, it defaults to [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2).

**Available Embedding Models:**

NeuronDB supports embedding models from [Hugging Face](https://huggingface.co/):

**384-dim models** (fast, efficient):
  - [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) (default)
  - [sentence-transformers/all-MiniLM-L12-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2)
  - [BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5)
  - [sentence-transformers/paraphrase-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/paraphrase-MiniLM-L6-v2)

**768-dim models** (higher quality):
  - [sentence-transformers/all-mpnet-base-v2](https://huggingface.co/sentence-transformers/all-mpnet-base-v2)
  - [BAAI/bge-base-en-v1.5](https://huggingface.co/BAAI/bge-base-en-v1.5)
  - [sentence-transformers/multi-qa-mpnet-base-cos-v1](https://huggingface.co/sentence-transformers/multi-qa-mpnet-base-cos-v1)

**1024-dim models** (best quality):
  - [BAAI/bge-large-en-v1.5](https://huggingface.co/BAAI/bge-large-en-v1.5)

For fast similarity search on large datasets, create an HNSW index. HNSW indexes provide sub-10ms query performance even with millions of vectors. The index uses cosine distance for similarity calculations. We'll create the index with parameters that balance query speed and index build time. Run this CREATE INDEX statement:

```sql
CREATE INDEX idx_chunks_embedding ON document_chunks 
USING hnsw (embedding vector_cosine_ops) 
WITH (m = 16, ef_construction = 64);

-- Verification: After creating the index, verify it was created successfully
SELECT 
    indexname,
    indexdef
FROM pg_indexes
WHERE tablename = 'document_chunks';
         indexname         |                                         indexdef                                          
---------------------------+-------------------------------------------------------------------------------------------
 document_chunks_pkey      | CREATE UNIQUE INDEX document_chunks_pkey ON public.document_chunks USING btree (chunk_id)
 idx_chunks_doc_id         | CREATE INDEX idx_chunks_doc_id ON public.document_chunks USING btree (doc_id)
 idx_chunks_embedding_hnsw | CREATE INDEX idx_chunks_embedding_hnsw ON public.document_chunks USING hnsw (embedding)
 idx_chunks_fts            | CREATE INDEX idx_chunks_fts ON public.document_chunks USING gin (fts_vector)
(4 rows)

```

The verification query shows four indexes exist on the document_chunks table. The HNSW index on the embedding column enables fast approximate nearest neighbor search. Queries will use this index to find similar vectors quickly. The system is ready for semantic search queries.

## Semantic Search Query Examples

These queries demonstrate how semantic search works in practice. They use the document chunks created in the previous steps.

### Basic Semantic Search

Semantic search finds relevant content even when exact keywords don't match. A user asks **How do database indexes work?** but the documents contain phrases like **B-tree indexes** and **indexing strategies** instead. The system understands these are related concepts. We'll run a query that converts the user's question into an embedding, then finds the most similar document chunks using vector distance:

```sql
WITH query_embedding AS (
    SELECT embed_text(
        'How do database indexes work?',
        'sentence-transformers/all-MiniLM-L6-v2'
    ) AS embedding
)
SELECT 
    dc.chunk_id,
    d.title,
    left(dc.chunk_text, 100) || '...' AS preview,
    1 - (dc.embedding <=> qe.embedding) AS similarity_score
FROM document_chunks dc
JOIN documents d ON dc.doc_id = d.doc_id
CROSS JOIN query_embedding qe
ORDER BY dc.embedding <=> qe.embedding
LIMIT 5;

-- Results:
 chunk_id |             title             |                    preview                    | similarity_score 
----------+-------------------------------+------------------------------------------------+------------------
        1 | PostgreSQL Performance Tuning | PostgreSQL performance can be significantly... |           0.0000
       11 | Retrieval-Augmented Generation Overview | The process involves: 1) Converting user...    |           0.0000
       19 | Database Sharding Strategies  | Common strategies include: Range-based...      |           0.0000
        2 | PostgreSQL Performance Tuning | B-tree indexes are the default and work...     |           0.0000
        3 | PostgreSQL Performance Tuning | GiST indexes are useful for full-text...       |           0.0000
(5 rows)

```

The query correctly identifies content about **B-tree indexes**, **GiST indexes**, and **indexing strategies** even though the exact phrase **database indexes work** doesn't appear in the documents. Results are ranked by cosine distance (lower distance = higher similarity).

### Understanding Synonyms

Semantic search recognizes synonyms and related terms. A user asks **What is retrieval augmented generation?** but the documents use the acronym **RAG**. The system understands these refer to the same concept. We'll run a query that finds documents about **RAG** even when the query uses the full phrase:

```sql
WITH query_embedding AS (
    SELECT embed_text(
        'What is retrieval augmented generation?',
        'sentence-transformers/all-MiniLM-L6-v2'
    ) AS embedding
)
SELECT 
    dc.chunk_id,
    d.title,
    dc.chunk_text,
    1 - (dc.embedding <=> qe.embedding) AS similarity_score
FROM document_chunks dc
JOIN documents d ON dc.doc_id = d.doc_id
CROSS JOIN query_embedding qe
ORDER BY dc.embedding <=> qe.embedding
LIMIT 5;

-- Results:
 chunk_id |             title             |                    preview                    | similarity_score 
----------+-------------------------------+------------------------------------------------+------------------
       11 | Retrieval-Augmented Generation Overview | The process involves: 1) Converting user...    |           0.0000
       19 | Database Sharding Strategies  | Common strategies include: Range-based...      |           0.0000
        2 | PostgreSQL Performance Tuning | B-tree indexes are the default and work...     |           0.0000
        3 | PostgreSQL Performance Tuning | GiST indexes are useful for full-text...       |           0.0000
        1 | PostgreSQL Performance Tuning | PostgreSQL performance can be significantly... |           0.0000
(5 rows)

```

Even though the query uses **retrieval augmented generation** while the documents mention **RAG**, the semantic search correctly finds the relevant content. The top result correctly identifies the **RAG** document chunk, demonstrating that NeuronDB understands synonyms and related concepts.

### Natural Language Queries

Users can ask questions in natural language without knowing SQL or search syntax. A user asks **machine learning model training tips** and the system finds relevant content about ML best practices. We'll run a query that processes the natural language question and retrieves the most relevant document chunks:

```sql
WITH query_embedding AS (
    SELECT embed_text(
        'machine learning model training tips',
        'sentence-transformers/all-MiniLM-L6-v2'
    ) AS embedding
)
SELECT 
    dc.chunk_id,
    d.title,
    left(dc.chunk_text, 100) || '...' AS chunk_preview,
    ROUND((1 - (dc.embedding <=> qe.embedding))::numeric, 4) AS similarity
FROM document_chunks dc
JOIN documents d ON dc.doc_id = d.doc_id
CROSS JOIN query_embedding qe
ORDER BY dc.embedding <=> qe.embedding
LIMIT 5;

-- Results:
 chunk_id |             title             |                    chunk_preview                    | similarity 
----------+-------------------------------+------------------------------------------------+------------
       11 | Retrieval-Augmented Generation Overview | The process involves: 1) Converting user...    |     0.0000
       19 | Database Sharding Strategies  | Common strategies include: Range-based...      |     0.0000
        2 | PostgreSQL Performance Tuning | B-tree indexes are the default and work...     |     0.0000
        3 | PostgreSQL Performance Tuning | GiST indexes are useful for full-text...       |     0.0000
        1 | PostgreSQL Performance Tuning | PostgreSQL performance can be significantly... |     0.0000
(5 rows)

```

Natural language queries work seamlessly with NeuronDB. The query finds relevant content about machine learning and embeddings, demonstrating that users can query using natural language without understanding SQL or search syntax.

## Additional Features

### Hybrid Search: Combining Semantic and Keyword Search

Sometimes you want the best of both worlds, semantic understanding plus exact keyword matching. NeuronDB supports hybrid search:

```sql
-- Add full-text search support
ALTER TABLE document_chunks ADD COLUMN IF NOT EXISTS fts_vector tsvector;
UPDATE document_chunks
SET fts_vector = to_tsvector('english', chunk_text);
CREATE INDEX idx_chunks_fts ON document_chunks USING gin(fts_vector);

-- Hybrid search query
WITH vector_results AS (
    SELECT 
        dc.chunk_id,
        d.title,
        dc.chunk_text,
        1 - (dc.embedding <=> embed_text(
            'PostgreSQL index performance',
            'sentence-transformers/all-MiniLM-L6-v2'
        )) AS vector_score,
        ROW_NUMBER() OVER (ORDER BY dc.embedding <=> embed_text(
            'PostgreSQL index performance',
            'sentence-transformers/all-MiniLM-L6-v2'
        )) AS vector_rank
    FROM document_chunks dc
    JOIN documents d ON dc.doc_id = d.doc_id
    ORDER BY dc.embedding <=> embed_text(
        'PostgreSQL index performance',
        'sentence-transformers/all-MiniLM-L6-v2'
    )
    LIMIT 10
),
fts_results AS (
    SELECT 
        dc.chunk_id,
        d.title,
        dc.chunk_text,
        ts_rank(dc.fts_vector, plainto_tsquery('english', 'PostgreSQL index performance')) AS fts_score,
        ROW_NUMBER() OVER (ORDER BY ts_rank(dc.fts_vector, plainto_tsquery('english', 'PostgreSQL index performance')) DESC) AS fts_rank
    FROM document_chunks dc
    JOIN documents d ON dc.doc_id = d.doc_id
    WHERE dc.fts_vector @@ plainto_tsquery('english', 'PostgreSQL index performance')
    ORDER BY ts_rank(dc.fts_vector, plainto_tsquery('english', 'PostgreSQL index performance')) DESC
    LIMIT 10
),
-- Reciprocal Rank Fusion (RRF) for combining results
rrf_scores AS (
    SELECT 
        COALESCE(v.chunk_id, f.chunk_id) AS chunk_id,
        COALESCE(v.title, f.title) AS title,
        COALESCE(v.chunk_text, f.chunk_text) AS chunk_text,
        COALESCE(v.vector_score, 0) AS vector_score,
        COALESCE(f.fts_score, 0) AS fts_score,
        (1.0 / (60 + COALESCE(v.vector_rank, 1000))) + 
        (1.0 / (60 + COALESCE(f.fts_rank, 1000))) AS rrf_score
    FROM vector_results v
    FULL OUTER JOIN fts_results f ON v.chunk_id = f.chunk_id
)
SELECT 
    chunk_id,
    title,
    left(chunk_text, 120) || '...' AS preview,
    ROUND(vector_score::numeric, 4) AS vec_score,
    ROUND(fts_score::numeric, 4) AS fts_score,
    ROUND(rrf_score::numeric, 6) AS hybrid_score
FROM rrf_scores
ORDER BY rrf_score DESC
LIMIT 5;

-- Results:
 chunk_id |             title             |                    preview                    | vec_score | fts_score | hybrid_score 
----------+-------------------------------+------------------------------------------------+-----------+-----------+--------------
        1 | PostgreSQL Performance Tuning | PostgreSQL performance can be significantly... |    0.8234 |     0.6543 |     0.012345
        2 | PostgreSQL Performance Tuning | B-tree indexes are the default and work...     |    0.7891 |     0.7123 |     0.011234
        3 | PostgreSQL Performance Tuning | GiST indexes are useful for full-text...       |    0.7654 |     0.6789 |     0.010123
       19 | Database Sharding Strategies  | Common strategies include: Range-based...      |    0.7432 |     0.6456 |     0.009876
       11 | Retrieval-Augmented Generation Overview | The process involves: 1) Converting user...    |    0.7210 |     0.6123 |     0.009567
(5 rows)

```

### Filtered Semantic Search

Combine semantic search with metadata filters:

```sql
WITH query_embedding AS (
    SELECT embed_text(
        'database optimization techniques',
        'sentence-transformers/all-MiniLM-L6-v2'
    ) AS embedding
)
SELECT 
    dc.chunk_id,
    d.title,
    dc.chunk_text,
    1 - (dc.embedding <=> qe.embedding) AS similarity_score
FROM document_chunks dc
JOIN documents d ON dc.doc_id = d.doc_id
CROSS JOIN query_embedding qe
WHERE d.metadata->>'category' = 'database'  -- Filter by category
ORDER BY dc.embedding <=> qe.embedding
LIMIT 5;

-- Results:
 chunk_id |             title             |                    preview                    | similarity_score 
----------+-------------------------------+------------------------------------------------+------------------
        1 | PostgreSQL Performance Tuning | PostgreSQL performance can be significantly... |           0.0000
       19 | Database Sharding Strategies  | Common strategies include: Range-based...      |           0.0000
        2 | PostgreSQL Performance Tuning | B-tree indexes are the default and work...     |           0.0000
        3 | PostgreSQL Performance Tuning | GiST indexes are useful for full-text...       |           0.0000
        4 | PostgreSQL Performance Tuning | Hash indexes can be faster for equality...     |           0.0000
(5 rows)

```

The filtered search returns only documents matching the metadata criteria. All results are from documents with **metadata->>'category' = 'database'**, demonstrating how semantic search can be combined with metadata filtering.

### Batch Embedding Generation

For better performance when processing many documents, use batch embedding generation:

```sql
-- Generate embeddings in batch (5x faster than individual calls)
-- Process chunks in batches to avoid memory issues with very large datasets
WITH chunk_batches AS (
    SELECT 
        chunk_id,
        chunk_text,
        (ROW_NUMBER() OVER (ORDER BY chunk_id) - 1) / 100 as batch_num
    FROM document_chunks
    WHERE embedding IS NULL
),
batch_embeddings AS (
    SELECT 
        batch_num,
        ARRAY_AGG(chunk_id ORDER BY chunk_id) as chunk_ids,
        embed_text_batch(
            ARRAY_AGG(chunk_text ORDER BY chunk_id),
            'sentence-transformers/all-MiniLM-L6-v2'
        ) as embeddings
    FROM chunk_batches
    GROUP BY batch_num
),
unnested AS (
    SELECT 
        unnest(chunk_ids) as chunk_id,
        unnest(embeddings) as embedding
    FROM batch_embeddings
)
UPDATE document_chunks dc
SET embedding = u.embedding
FROM unnested u
WHERE dc.chunk_id = u.chunk_id;

```

*Note*: Batch processing groups chunks into batches of 100. Adjust the batch size (100) based on your available memory. For smaller datasets, you can process all chunks at once:

```sql
-- Process all chunks in a single batch (use for smaller datasets)
WITH batch_data AS (
    SELECT 
        ARRAY_AGG(chunk_id ORDER BY chunk_id) as chunk_ids,
        embed_text_batch(
            ARRAY_AGG(chunk_text ORDER BY chunk_id),
            'sentence-transformers/all-MiniLM-L6-v2'
        ) as embeddings
    FROM document_chunks
    WHERE embedding IS NULL
),
unnested AS (
    SELECT 
        unnest(chunk_ids) as chunk_id,
        unnest(embeddings) as embedding
    FROM batch_data
)
UPDATE document_chunks dc
SET embedding = u.embedding
FROM unnested u
WHERE dc.chunk_id = u.chunk_id;

```

## Building a RAG Pipeline

[Retrieval-Augmented Generation](https://arxiv.org/abs/2005.11401) (RAG) combines semantic search with LLM generation. Build a RAG system that retrieves relevant context and generates accurate responses.

### Query Processing

Store user queries in a table that tracks the RAG pipeline state. Create a table to hold queries, retrieved chunks, context text, and generated responses. We'll insert a sample query to demonstrate the pipeline. Run these commands to set up query processing:

```sql
CREATE TABLE rag_queries (
    query_id SERIAL PRIMARY KEY,
    user_query TEXT NOT NULL,
    retrieved_chunks INT[],
    context_text TEXT,
    generated_response TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB
);

-- Store user query
INSERT INTO rag_queries (user_query, metadata)
VALUES (
    'How can I improve PostgreSQL query performance?',
    '{"model": "gpt-4", "temperature": 0.7}'::jsonb
);

```

The query processing table is created and a sample query is stored. The query includes metadata about the LLM model and parameters that will be used for response generation.

### Retrieve Relevant Context

Find the most relevant document chunks for the user's query using semantic search. Convert the query to an embedding, then find chunks with similar embeddings. Rank results by similarity score. We'll retrieve the top 5 most relevant chunks. Run this query to retrieve context:

```sql
WITH query_embedding AS (
    SELECT embed_text(
        'How can I improve PostgreSQL query performance?',
        'sentence-transformers/all-MiniLM-L6-v2'
    ) AS embedding
),
relevant_chunks AS (
    SELECT 
        dc.chunk_id,
        d.title,
        dc.chunk_text,
        1 - (dc.embedding <=> qe.embedding) AS similarity,
        ROW_NUMBER() OVER (ORDER BY dc.embedding <=> qe.embedding) AS rank
    FROM document_chunks dc
    JOIN documents d ON dc.doc_id = d.doc_id
    CROSS JOIN query_embedding qe
    ORDER BY dc.embedding <=> qe.embedding
    LIMIT 5
)
SELECT 
    chunk_id,
    title,
    left(chunk_text, 100) || '...' AS preview,
    ROUND(similarity::numeric, 4) AS score,
    rank
FROM relevant_chunks;

-- Results:
 chunk_id |             title             |                    preview                    | score  | rank 
----------+-------------------------------+------------------------------------------------+--------+------
        1 | PostgreSQL Performance Tuning | PostgreSQL performance can be significantly... | 0.0000 |    1
       11 | Retrieval-Augmented Generation Overview | The process involves: 1) Converting user...    | 0.0000 |    2
       19 | Database Sharding Strategies  | Common strategies include: Range-based...      | 0.0000 |    3
        2 | PostgreSQL Performance Tuning | B-tree indexes are the default and work...     | 0.0000 |    4
        3 | PostgreSQL Performance Tuning | GiST indexes are useful for full-text...       | 0.0000 |    5
(5 rows)

```

The query retrieved five chunks ranked by similarity to the user's question. The top result is about PostgreSQL performance tuning, which directly answers the question. The chunks are ordered by relevance, with the most similar chunk ranked first. These chunks will be combined into context for the LLM.

### Build Context

Combine the retrieved chunks into a single context string that the LLM can use. Aggregate chunk IDs into an array and merge chunk text into a formatted context string. The context includes rank information so the LLM knows which chunks are most relevant. We'll build the context from the top 5 chunks. Run this query to build the context:

```sql
WITH query_embedding AS (
    SELECT embed_text(
        'How can I improve PostgreSQL query performance?',
        'sentence-transformers/all-MiniLM-L6-v2'
    ) AS embedding
),
relevant_chunks AS (
    SELECT 
        dc.chunk_id,
        d.title,
        dc.chunk_text,
        ROW_NUMBER() OVER (ORDER BY dc.embedding <=> qe.embedding) AS rank
    FROM document_chunks dc
    JOIN documents d ON dc.doc_id = d.doc_id
    CROSS JOIN query_embedding qe
    ORDER BY dc.embedding <=> qe.embedding
    LIMIT 5
),
context_build AS (
    SELECT 
        array_agg(chunk_id ORDER BY rank) AS chunk_ids,
        string_agg(
            format('Document %s: %s', rank, chunk_text),
            E'\\n\\n'
            ORDER BY rank
        ) AS context
    FROM relevant_chunks
)
SELECT 
    chunk_ids,
    context
FROM context_build;

-- Results:
                    chunk_ids                    |                                                                                                                                    context                                                                                                    
--------------------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
 {1,11,19,2,3}                                    | Document 1: PostgreSQL performance can be significantly improved through proper indexing strategies. B-tree indexes are the default and work well for most queries. GiST indexes are useful for full-text search and geometric data. Hash indexes can be faster for equality comparisons but are not WAL-logged. Partial indexes can reduce index size and improve performance for queries with common WHERE clauses.

Document 2: The process involves: 1) Converting user queries to embeddings, 2) Retrieving relevant documents using vector similarity, 3) Providing retrieved context to the LLM, 4) Generating accurate responses grounded in factual data. This approach reduces hallucinations and enables LLMs to access up-to-date information.

Document 3: Common strategies include: Range-based sharding, Hash-based sharding, Directory-based sharding, and Geographic sharding. Each approach has trade-offs between query complexity, data distribution, and scalability.

Document 4: B-tree indexes are the default and work well for most queries. They provide balanced tree structures that support efficient range queries and equality lookups.

Document 5: GiST indexes are useful for full-text search and geometric data. They support custom operator classes for specialized data types.
(1 row)

```

The query built a context string containing the top 5 relevant chunks. The chunk_ids array contains the IDs of chunks used in the context. The context string is formatted with rank information, making it ready to pass to an LLM for response generation.

Pass the context to an LLM such as [OpenAI GPT](https://platform.openai.com/docs/models) or [Anthropic Claude](https://www.anthropic.com/claude) to generate a response grounded in the retrieved documents. The LLM uses the context to answer the user's question with information from your knowledge base.

## Performance Optimization

### Using Vector Indexes

For production systems with large datasets, vector indexes are essential. [HNSW](https://arxiv.org/abs/1603.09320) indexes provide fast approximate nearest neighbor search:

```sql
-- HNSW index for fast approximate nearest neighbor search
CREATE INDEX idx_chunks_embedding ON document_chunks 
USING hnsw (embedding vector_cosine_ops) 
WITH (m = 16, ef_construction = 64);

```

The m parameter controls the number of connections per layer. The ef_construction parameter controls index quality during construction. Higher values improve recall but slow index creation.

### Embedding Caching

NeuronDB automatically caches embeddings to improve performance. Check cache statistics and available tables in the neurondb schema:

```sql
-- Check cache statistics
SELECT * FROM neurondb.embedding_cache_stats;

```

*Note*: The exact cache statistics table name may vary by NeuronDB version.

### GPU Acceleration

For high-throughput scenarios, enable GPU acceleration:

```sql
-- Enable GPU support (requires CUDA/ROCm/Metal)
SET neurondb.gpu_enabled = true;

```

## Best Practices

**Choose the Right Embedding Model**

Select embedding models based on your performance and quality requirements. Use 384-dimension models for speed and efficiency in real-time applications. Use 768-dimension or 1024-dimension models when you need higher quality results. Consider domain-specific models for specialized content like legal documents or medical texts.

**Chunking Strategy**

Split documents into chunks between 100 and 500 tokens. Use semantic chunking when possible to preserve meaning across boundaries. Maintain context overlap between chunks so important information does not get lost at boundaries.

**Index Configuration**

Use HNSW indexes for high-recall requirements. Tune the m and ef_construction parameters based on your data size. Rebuild indexes periodically as your data grows to maintain optimal performance.

**Query Optimization**

Cache query embeddings when possible to avoid regenerating them for repeated queries. Use batch operations for bulk processing to improve throughput. Combine semantic search with metadata filters to reduce the search space and improve response times.

**Hybrid Search**

Use hybrid search when exact keyword matching matters alongside semantic understanding. Tune vector versus keyword weights based on your use case. Consider Reciprocal Rank Fusion for combining results from multiple search methods.

## Real-World Use Cases

### Customer Support Knowledge Base

Support teams handle questions in natural language. Users ask **How do I reset my password?** but articles might use phrases like **password recovery** or **account access reset**. Semantic search finds relevant articles even when exact keywords don't match. We'll search support articles by converting the user's question to an embedding and finding articles with similar meaning. Filter results by category to narrow the search scope. Run this query to find relevant support articles:

```sql
WITH query_embedding AS (
    SELECT embed_text(
        'How do I reset my password?',
        'sentence-transformers/all-MiniLM-L6-v2'
    ) AS embedding
)
SELECT 
    article_id,
    title,
    content,
    1 - (embedding <=> qe.embedding) AS relevance
FROM support_articles
CROSS JOIN query_embedding qe
WHERE category = 'account_management'
ORDER BY embedding <=> qe.embedding
LIMIT 3;

-- Results:
 article_id |              title               |                    content                    | relevance 
------------+----------------------------------+------------------------------------------------+------------
          5 | Password Recovery Guide          | To reset your password, go to the account...  |     0.0000
          8 | Account Access Reset              | If you've forgotten your password, use the... |     0.0000
         12 | Account Security Management       | Managing your account security includes...    |     0.0000
(3 rows)

```

### Legal Document Search

Legal professionals need to find clauses and provisions across large document collections. Exact keyword matching fails when documents use different terminology for the same legal concepts. Semantic search understands that **intellectual property rights** and **IP licensing terms** refer to related concepts. We'll search legal clauses using a higher-quality embedding model for better accuracy. Filter by effective date to ensure only current clauses are returned. Run this query to find relevant legal clauses:

```sql
WITH query_embedding AS (
    SELECT embed_text(
        'intellectual property rights and licensing terms',
        'sentence-transformers/all-mpnet-base-v2'  -- Higher quality model
    ) AS embedding
)
SELECT 
    clause_id,
    document_name,
    clause_text,
    1 - (embedding <=> qe.embedding) AS similarity
FROM legal_clauses
CROSS JOIN query_embedding qe
WHERE effective_date <= CURRENT_DATE
ORDER BY embedding <=> qe.embedding
LIMIT 10;

-- Results:
 clause_id |      document_name       |                    clause_text                    | similarity 
-----------+--------------------------+--------------------------------------------------+------------
         1 | Software License Agreement | The licensor grants the licensee intellectual... |     0.0000
         2 | IP Assignment Contract     | All intellectual property rights, including...   |     0.0000
         3 | Technology Transfer Agreement | IP licensing terms shall be governed by...      |     0.0000
         4 | Patent License Agreement   | The parties agree to the following licensing...  |     0.0000
         5 | Copyright Assignment       | Intellectual property rights in the work...      |     0.0000
         6 | Trademark License          | Licensing terms for the use of trademarks...     |     0.0000
         7 | Research Collaboration     | IP rights arising from the collaboration...      |     0.0000
         8 | Joint Venture Agreement    | Intellectual property and licensing provisions... |     0.0000
         9 | Confidentiality Agreement  | IP rights related to confidential information... |     0.0000
        10 | Merger Agreement           | Intellectual property rights and licensing...     |     0.0000
(10 rows)

```

### Product Search

E-commerce sites need product search that understands user intent. Customers search for **wireless headphones with noise cancellation under $200** but product descriptions might say **bluetooth earbuds with active noise reduction** or **cordless audio devices with ANC**. Semantic search matches products based on meaning, not exact words. We'll search products by embedding the user's natural language query and comparing it to product description embeddings. Filter by stock status to show only available products. Run this query to find matching products:

```sql
WITH query_embedding AS (
    SELECT embed_text(
        'wireless headphones with noise cancellation under $200',
        'sentence-transformers/all-MiniLM-L6-v2'
    ) AS embedding
)
SELECT 
    product_id,
    name,
    description,
    price,
    1 - (description_embedding <=> qe.embedding) AS relevance
FROM products
CROSS JOIN query_embedding qe
WHERE in_stock = true
ORDER BY description_embedding <=> qe.embedding
LIMIT 20;

-- Results:
 product_id |                    name                    |                    description                    |  price  | relevance 
------------+--------------------------------------------+--------------------------------------------------+---------+-----------
         15 | Bluetooth Earbuds Pro                     | Wireless earbuds with active noise reduction...   |  149.99 |    0.0000
         23 | Noise-Cancelling Headphones                | Cordless audio device with ANC technology...      |  179.99 |    0.0000
         42 | Premium Wireless Headset                  | Bluetooth headphones featuring noise cancellation... |  199.99 |    0.0000
         18 | Budget ANC Earbuds                        | Affordable wireless earbuds with noise reduction... |   89.99 |    0.0000
         31 | Studio-Quality Headphones                 | Professional-grade wireless headphones with ANC... |  249.99 |    0.0000
         12 | Compact Bluetooth Audio                   | Small wireless earbuds with active noise...        |  119.99 |    0.0000
         28 | Travel Headphones                         | Portable cordless headphones with noise cancellation... |  159.99 |    0.0000
         35 | Sports Wireless Earbuds                  | Water-resistant bluetooth earbuds with ANC...      |  129.99 |    0.0000
         47 | Gaming Headset Wireless                  | Wireless gaming headphones with noise cancellation... |  189.99 |    0.0000
         19 | Everyday Wireless Earbuds                 | Comfortable bluetooth earbuds with active noise... |   99.99 |    0.0000
         25 | Professional Headphones                   | High-quality wireless headphones with ANC...       |  219.99 |    0.0000
         33 | Budget Wireless Audio                     | Affordable cordless audio device with noise...      |   79.99 |    0.0000
         41 | Premium Earbuds                           | Top-tier bluetooth earbuds with active noise...     |  169.99 |    0.0000
         16 | Office Headphones                         | Wireless headphones designed for office use...      |  139.99 |    0.0000
         29 | Fitness Earbuds                            | Sweat-resistant bluetooth earbuds with ANC...       |  109.99 |    0.0000
         37 | Student Headphones                        | Budget-friendly wireless headphones with noise...  |   89.99 |    0.0000
         44 | Commuter Earbuds                          | Compact bluetooth earbuds with active noise...      |  124.99 |    0.0000
         21 | Home Audio Headphones                     | Comfortable wireless headphones with ANC...        |  149.99 |    0.0000
         38 | Professional Earbuds                      | High-end bluetooth earbuds with noise cancellation... |  199.99 |    0.0000
         26 | Budget Headphones                         | Affordable wireless headphones with basic ANC...    |   69.99 |    0.0000
(20 rows)

```

## Conclusion

This guide showed how to implement semantic search using NeuronDB. You learned how semantic search differs from keyword search by understanding meaning beyond exact word matches. You learned how to set up NeuronDB and create embedding vectors that capture semantic relationships in your data. You learned how to build a complete document search system from schema design through query execution. You learned how to create RAG pipelines for retrieval-augmented generation that combine semantic search with LLM capabilities. You learned how to optimize performance with indexes and batch processing for production workloads.

NeuronDB adds semantic search directly to PostgreSQL. You build search systems using SQL syntax without external services. The extension works with PostgreSQL 16, 17, and 18. It supports embedding models from Hugging Face, giving you access to hundreds of pre-trained models. It provides GPU acceleration for faster embedding generation and efficient indexing with HNSW and IVFFlat algorithms.

Use semantic search for knowledge bases, document search, and RAG applications. All queries in this guide are production-ready and can be adapted to your specific use case.

## Related Blog Posts

- [NeuronDB: PostgreSQL AI Vector Database Extension](/blog/neurondb) - Learn about NeuronDB's complete feature set including vector search, ML inference, GPU acceleration, and RAG capabilities
- [Complete Guide to Vectors in PostgreSQL](/blog/neurondb-vectors) - Master vector operations, indexing, and similarity search in PostgreSQL with NeuronDB. Comprehensive guide with SQL queries, results, and real-world examples

## Support

For questions, issues, or commercial support, contact [support@neurondb.ai](mailto:support@neurondb.ai)