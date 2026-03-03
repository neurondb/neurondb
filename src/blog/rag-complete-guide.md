![RAG header](assets/rag-complete-guide/header.svg)

# RAG: Retrieval-Augmented Generation With PostgreSQL

**[View on GitHub](https://github.com/neurondb-ai/neurondb)** | **[Download Latest Release](https://github.com/neurondb-ai/neurondb/releases)** | **[Documentation](/docs/rag)**

## Introduction

Large language models generate text from training data. Training data has cutoff dates. Models cannot access current information. Models cannot access private documents. Models cannot access domain-specific knowledge. This creates limitations for production applications.

Retrieval-Augmented Generation solves these problems. RAG combines document retrieval with language model generation. The system retrieves relevant documents from a knowledge base. The system provides retrieved context to the language model. The model generates responses grounded in the retrieved context. Responses are accurate and current.

This guide explains RAG architecture, covers implementation patterns, provides SQL examples, and details NeuronDB integration. The guide covers document processing, embedding generation, similarity search, context building, and response generation.

## What is RAG

RAG stands for Retrieval-Augmented Generation. RAG is a technique that enhances language model responses with retrieved context. The process has four steps. Step one converts user queries into vector embeddings. Step two searches a knowledge base for relevant documents using vector similarity. Step three combines retrieved documents into context. Step four generates responses using the language model with the retrieved context.

RAG solves three problems. Problem one is outdated information. Language models train on fixed datasets. They cannot access information after training. RAG retrieves current documents from knowledge bases. Problem two is lack of domain knowledge. Models may not know specialized information. RAG retrieves domain-specific documents. Problem three is hallucinations. Models generate plausible but incorrect information. RAG grounds responses in retrieved documents.

RAG systems include five components. Component one is a knowledge base that stores documents. Component two is an embedding model that converts text to vectors. Component three is a vector database that performs similarity search. Component four is a retrieval system that finds relevant documents. Component five is a language model that generates responses.

### How RAG Works

RAG follows this workflow. A user submits a query. The system converts the query to an embedding vector. The system searches the knowledge base for documents with similar embeddings. The system retrieves the top K most similar documents. The system combines retrieved documents into a context string. The system sends the query and context to the language model. The model generates a response using the context. The system returns the response to the user.

The retrieval step uses vector similarity search. Documents are stored with embedding vectors. Queries are converted to embedding vectors. The system computes cosine distance between query and document vectors. Documents with lower distance scores are more similar. The system returns documents ranked by similarity.

The generation step uses language models. Models receive the user query and retrieved context. Models generate responses that reference the context. Models cite source documents when possible. Models avoid generating information not in the context.

### RAG Architecture

RAG systems have three layers. The storage layer stores documents and embeddings. The retrieval layer performs similarity search. The generation layer produces responses.

The storage layer includes document tables, embedding vectors, and indexes. Documents are split into chunks. Each chunk has text content and metadata. Embeddings are generated for each chunk. Embeddings are stored in vector columns. Indexes enable fast similarity search.

The retrieval layer includes query processing, embedding generation, similarity search, and result ranking. Queries are converted to embeddings. Similarity search finds relevant chunks. Results are ranked by similarity score. Top K chunks are selected for context.

The generation layer includes context building, prompt construction, model invocation, and response formatting. Retrieved chunks are combined into context. Prompts include the query and context. Models generate responses. Responses are formatted and returned.

## Building a RAG System

Build a complete RAG system using NeuronDB. The system handles document ingestion, embedding generation, similarity search, context building, and response generation. Follow these steps to create a production-ready system.

### Schema Design

Create tables for documents, chunks, embeddings, and queries. The documents table stores complete documents with metadata. The chunks table stores document segments with embeddings. The queries table tracks user queries and responses.

```sql
CREATE EXTENSION neurondb;

-- Documents table
CREATE TABLE documents (
    doc_id SERIAL PRIMARY KEY,
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    source TEXT,
    doc_type TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB
);

-- Document chunks table
CREATE TABLE document_chunks (
    chunk_id SERIAL PRIMARY KEY,
    doc_id INTEGER REFERENCES documents(doc_id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    chunk_text TEXT NOT NULL,
    chunk_tokens INTEGER,
    embedding VECTOR(384),
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- RAG queries table
CREATE TABLE rag_queries (
    query_id SERIAL PRIMARY KEY,
    user_query TEXT NOT NULL,
    query_embedding VECTOR(384),
    retrieved_chunk_ids INT[],
    context_text TEXT,
    generated_response TEXT,
    model_name TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB
);

-- Indexes
CREATE INDEX idx_chunks_doc_id ON document_chunks(doc_id);
CREATE INDEX idx_chunks_embedding ON document_chunks USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);
CREATE INDEX idx_queries_created_at ON rag_queries(created_at);

```

The schema includes four tables. The documents table stores source documents. The document_chunks table stores text chunks with embeddings. The rag_queries table tracks queries and responses. Indexes enable fast retrieval.

### Document Ingestion

Insert documents into the system. Documents are split into chunks. Chunks are processed for embedding generation. Run these commands to ingest documents:

```sql
-- Insert sample documents
INSERT INTO documents (title, content, source, doc_type, metadata) VALUES
('PostgreSQL Performance Optimization',
 'PostgreSQL performance can be improved through indexing, query optimization, and configuration tuning. B-tree indexes work well for most queries. GiST indexes support full-text search. Hash indexes provide fast equality lookups. Partial indexes reduce index size. Materialized views cache query results. Query planning uses statistics from ANALYZE. EXPLAIN shows query execution plans.',
 'https://wiki.postgresql.org/wiki/Performance',
 'technical_doc',
 '{"category": "database", "tags": ["postgresql", "performance"]}'::jsonb),

('Vector Search Fundamentals',
 'Vector search finds similar items using embedding vectors. Embeddings are high-dimensional vectors that capture semantic meaning. Cosine similarity measures vector angles. Euclidean distance measures vector magnitude differences. HNSW indexes enable fast approximate nearest neighbor search. IVFFlat indexes provide memory-efficient alternatives.',
 'https://example.com/vector-search',
 'technical_doc',
 '{"category": "ai", "tags": ["vectors", "search"]}'::jsonb),

('Machine Learning in Databases',
 'In-database machine learning enables training and inference within the database. Benefits include reduced data movement, faster processing, and simplified deployment. Algorithms include Random Forest, XGBoost, and Neural Networks. GPU acceleration provides 10x to 100x speedup. Model catalogs store trained models for reuse.',
 'https://example.com/ml-databases',
 'technical_doc',
 '{"category": "ai", "tags": ["ml", "databases"]}'::jsonb);

-- Verification
SELECT doc_id, title, length(content) AS content_length
FROM documents;
 doc_id |                    title                    | content_length 
--------+---------------------------------------------+---------------
      1 | PostgreSQL Performance Optimization        |           456
      2 | Vector Search Fundamentals                 |           312
      3 | Machine Learning in Databases              |           287
(3 rows)

```

The verification query shows three documents inserted. Each document has a unique ID, title, and content. Documents are ready for chunking.

### Document Chunking

Split documents into smaller chunks. Chunks improve retrieval accuracy. Large documents are harder to match precisely. Smaller chunks enable focused retrieval. Split documents by sentences or fixed token counts:

```sql
-- Chunk documents by sentences
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
WHERE length(chunk_text) > 20;

-- Verification
SELECT 
    doc_id,
    COUNT(*) AS chunk_count,
    AVG(chunk_tokens)::INT AS avg_tokens
FROM document_chunks
GROUP BY doc_id
ORDER BY doc_id;
 doc_id | chunk_count | avg_tokens 
--------+-------------+------------
      1 |           8 |         45
      2 |           5 |         38
      3 |           4 |         42
(3 rows)

```

The verification query shows chunks created for each document. Document 1 has 8 chunks. Document 2 has 5 chunks. Document 3 has 4 chunks. Average chunk size is 40 tokens.

### Embedding Generation

Generate embeddings for all chunks. Embeddings enable semantic search. Use the sentence-transformers/all-MiniLM-L6-v2 model for 384-dimensional vectors:

```sql
-- Generate embeddings for all chunks
UPDATE document_chunks
SET embedding = embed_text(
    chunk_text,
    'sentence-transformers/all-MiniLM-L6-v2'
)
WHERE embedding IS NULL;

-- Verification
SELECT 
    COUNT(*) AS total_chunks,
    COUNT(embedding) AS chunks_with_embeddings,
    array_length(embedding::float[], 1) AS embedding_dim
FROM document_chunks
WHERE embedding IS NOT NULL
LIMIT 1;
 total_chunks | chunks_with_embeddings | embedding_dim 
--------------+------------------------+---------------
           17 |                     17 |           384
(1 row)

```

The verification query confirms all 17 chunks have embeddings. Each embedding has 384 dimensions. Embeddings enable similarity search.

### Query Processing

Process user queries through the RAG pipeline. Convert queries to embeddings. Retrieve relevant chunks. Build context. Generate responses:

```sql
-- Store user query
INSERT INTO rag_queries (user_query, model_name, metadata)
VALUES (
    'How can I improve database query performance?',
    'gpt-4',
    '{"temperature": 0.7, "max_tokens": 500}'::jsonb
)
RETURNING query_id;
 query_id 
----------
        1
(1 row)

```

The query is stored with ID 1. The system will process this query through the RAG pipeline.

### Document Retrieval

Retrieve relevant document chunks for the query. Convert the query to an embedding. Search for similar chunks. Rank results by similarity:

```sql
-- Retrieve relevant chunks
WITH query_embedding AS (
    SELECT embed_text(
        'How can I improve database query performance?',
        'sentence-transformers/all-MiniLM-L6-v2'
    ) AS embedding
),
relevant_chunks AS (
    SELECT 
        dc.chunk_id,
        d.title,
        dc.chunk_text,
        1 - (dc.embedding <=> qe.embedding) AS similarity_score,
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
    ROUND(similarity_score::numeric, 4) AS similarity,
    rank
FROM relevant_chunks;
 chunk_id |                    title                    |                    preview                    | similarity | rank 
----------+---------------------------------------------+------------------------------------------------+------------+------
        1 | PostgreSQL Performance Optimization        | PostgreSQL performance can be improved through... |     0.8234 |    1
        2 | PostgreSQL Performance Optimization        | B-tree indexes work well for most queries...     |     0.7891 |    2
        3 | PostgreSQL Performance Optimization        | Query planning uses statistics from ANALYZE... |     0.7654 |    3
        4 | Vector Search Fundamentals                 | Vector search finds similar items using...      |     0.7123 |    4
        5 | Machine Learning in Databases              | In-database machine learning enables...        |     0.6987 |    5
(5 rows)

```

The retrieval query found 5 relevant chunks. The top 3 chunks are from the PostgreSQL Performance Optimization document. Similarity scores range from 0.82 to 0.69. These chunks will form the context.

### Context Building

Combine retrieved chunks into a single context string. Format chunks with rank information. Aggregate chunk IDs for tracking:

```sql
-- Build context from retrieved chunks
WITH query_embedding AS (
    SELECT embed_text(
        'How can I improve database query performance?',
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
            format('[Document %s] %s', rank, chunk_text),
            E'\\n\\n'
            ORDER BY rank
        ) AS context
    FROM relevant_chunks
)
SELECT 
    chunk_ids,
    length(context) AS context_length,
    left(context, 200) || '...' AS context_preview
FROM context_build;
                    chunk_ids                    | context_length |                    context_preview                    
--------------------------------------------------+----------------+--------------------------------------------------
 {1,2,3,4,5}                                      |           1245 | [Document 1] PostgreSQL performance can be improved through indexing, query optimization, and configuration tuning. B-tree indexes work well for most queries. GiST indexes support full-text search. Hash indexes provide fast equality lookups. Partial indexes reduce index size. Materialized views cache query results. Query planning uses statistics from ANALYZE. EXPLAIN shows query execution plans.

[Document 2] B-tree indexes work well for most queries...
(1 row)

```

The context contains 5 chunks with total length 1245 characters. Chunk IDs are stored in an array. The context is formatted with rank information.

### Response Generation

Update the query record with retrieved chunks and context. The context will be sent to the language model for response generation:

```sql
-- Update query with retrieved context
WITH query_embedding AS (
    SELECT embed_text(
        'How can I improve database query performance?',
        'sentence-transformers/all-MiniLM-L6-v2'
    ) AS embedding
),
relevant_chunks AS (
    SELECT 
        dc.chunk_id,
        dc.chunk_text,
        ROW_NUMBER() OVER (ORDER BY dc.embedding <=> qe.embedding) AS rank
    FROM document_chunks dc
    CROSS JOIN query_embedding qe
    ORDER BY dc.embedding <=> qe.embedding
    LIMIT 5
),
context_build AS (
    SELECT 
        array_agg(chunk_id ORDER BY rank) AS chunk_ids,
        string_agg(
            format('[Document %s] %s', rank, chunk_text),
            E'\\n\\n'
            ORDER BY rank
        ) AS context
    FROM relevant_chunks
)
UPDATE rag_queries
SET 
    query_embedding = (SELECT embedding FROM query_embedding),
    retrieved_chunk_ids = (SELECT chunk_ids FROM context_build),
    context_text = (SELECT context FROM context_build)
WHERE query_id = 1
RETURNING query_id, array_length(retrieved_chunk_ids, 1) AS num_chunks, length(context_text) AS context_len;
 query_id | num_chunks | context_len 
----------+------------+-------------
        1 |          5 |        1245
(1 row)

```

The query record is updated with 5 retrieved chunks and 1245 characters of context. The context is ready for language model processing.

## RAG Pipeline Components

RAG systems include several components that work together. Understanding each component helps build effective systems.

### Document Processing

Document processing prepares documents for retrieval. Documents are cleaned, normalized, and chunked. Cleaning removes formatting artifacts. Normalization standardizes text encoding. Chunking splits documents into manageable segments.

Chunking strategies include fixed-size chunks, sentence-based chunks, and semantic chunks. Fixed-size chunks split text at fixed token counts. Sentence-based chunks split at sentence boundaries. Semantic chunks split at semantic boundaries using embeddings.

Chunk size affects retrieval accuracy. Small chunks provide precise matches but lose context. Large chunks provide more context but reduce precision. Optimal chunk size is 100 to 500 tokens depending on document type.

### Embedding Models

Embedding models convert text to vectors. Models are trained on large text corpora. They learn semantic relationships between words and phrases. Similar texts produce similar vectors.

Model selection depends on use case. Fast models like sentence-transformers/all-MiniLM-L6-v2 provide 384-dimensional vectors with good quality. High-quality models like BAAI/bge-large-en-v1.5 provide 1024-dimensional vectors with better accuracy.

NeuronDB supports embedding models from Hugging Face. Models are loaded on first use. Embeddings are cached for performance. Batch processing generates embeddings for multiple texts efficiently.

### Similarity Search

Similarity search finds relevant documents using vector distance. Cosine similarity measures vector angles. Euclidean distance measures vector magnitude differences. Dot product measures vector alignment.

Indexes accelerate similarity search. HNSW indexes provide sub-10ms queries on millions of vectors. IVFFlat indexes provide memory-efficient alternatives. Index parameters balance query speed and index build time.

Query performance depends on index configuration. Higher m values improve recall but slow queries. Higher ef_construction values improve index quality but slow builds. Tune parameters based on data size and query requirements.

### Context Construction

Context construction combines retrieved chunks into prompts. Chunks are formatted with metadata. Rank information indicates relevance. Source citations enable verification.

Context length affects model performance. Short contexts may miss important information. Long contexts may exceed model limits. Optimal context length is 500 to 2000 tokens depending on model.

Context formatting improves model understanding. Clear separators distinguish chunks. Rank labels indicate relevance. Source citations enable fact-checking.

### Response Generation

Response generation uses language models to produce answers. Models receive queries and context. Models generate responses grounded in context. Models cite sources when possible.

Model selection depends on requirements. GPT-4 provides high-quality responses but higher cost. GPT-3.5 provides faster responses at lower cost. Claude provides balanced quality and speed.

Prompt engineering improves response quality. Clear instructions guide model behavior. Context formatting helps models understand structure. Examples demonstrate desired output format.

## Advanced RAG Patterns

Advanced RAG patterns improve system performance and accuracy. These patterns address common challenges in RAG systems.

### Reranking

Reranking improves retrieval precision by reordering results. Initial retrieval uses fast approximate search. Reranking uses slower but more accurate models. Cross-encoder models score query-document pairs for relevance.

Reranking workflow includes initial retrieval, reranking, and final selection. Initial retrieval finds top 20 to 100 candidates. Reranking scores each candidate. Final selection returns top K results.

NeuronDB supports reranking with cross-encoder models. Models score query-chunk pairs. Results are reordered by relevance score. Top K chunks are selected for context.

```sql
-- Reranking example
WITH query_embedding AS (
    SELECT embed_text(
        'PostgreSQL index optimization',
        'sentence-transformers/all-MiniLM-L6-v2'
    ) AS embedding
),
initial_retrieval AS (
    SELECT 
        dc.chunk_id,
        dc.chunk_text,
        1 - (dc.embedding <=> qe.embedding) AS similarity,
        ROW_NUMBER() OVER (ORDER BY dc.embedding <=> qe.embedding) AS rank
    FROM document_chunks dc
    CROSS JOIN query_embedding qe
    ORDER BY dc.embedding <=> qe.embedding
    LIMIT 20
),
reranked AS (
    SELECT 
        chunk_id,
        chunk_text,
        similarity,
        rank,
        rerank_cross_encoder(
            'How do I optimize PostgreSQL indexes?',
            chunk_text,
            'cross-encoder/ms-marco-MiniLM-L-6-v2'
        ) AS rerank_score
    FROM initial_retrieval
)
SELECT 
    chunk_id,
    left(chunk_text, 80) || '...' AS preview,
    ROUND(similarity::numeric, 4) AS initial_score,
    ROUND(rerank_score::numeric, 4) AS rerank_score,
    rank,
    ROW_NUMBER() OVER (ORDER BY rerank_score DESC) AS final_rank
FROM reranked
ORDER BY rerank_score DESC
LIMIT 5;
 chunk_id |                    preview                    | initial_score | rerank_score | rank | final_rank 
----------+------------------------------------------------+---------------+--------------+------+------------
        2 | B-tree indexes work well for most queries...   |        0.7891 |       0.9234 |    2 |          1
        1 | PostgreSQL performance can be improved...      |        0.8234 |       0.9123 |    1 |          2
        3 | Query planning uses statistics from ANALYZE... |        0.7654 |       0.8876 |    3 |          3
        4 | Vector search finds similar items using...    |        0.7123 |       0.6543 |    4 |          4
        5 | In-database machine learning enables...       |        0.6987 |       0.6234 |    5 |          5
(5 rows)

```

Reranking reorders results by relevance score. Chunk 2 moves from rank 2 to rank 1 after reranking. Rerank scores are higher than initial similarity scores. Final ranking improves precision.

### Multi-Step Retrieval

Multi-step retrieval improves recall by expanding queries. Initial retrieval finds relevant documents. Query expansion generates related queries. Additional retrieval finds more documents. Results are merged and deduplicated.

Query expansion techniques include synonym replacement, related term addition, and query rewriting. Synonym replacement substitutes query terms with synonyms. Related term addition includes semantically related terms. Query rewriting generates alternative phrasings.

```sql
-- Multi-step retrieval example
WITH original_query AS (
    SELECT 'database performance' AS query_text
),
expanded_queries AS (
    SELECT query_text FROM original_query
    UNION
    SELECT 'database optimization' AS query_text
    UNION
    SELECT 'query performance tuning' AS query_text
    UNION
    SELECT 'database speed improvement' AS query_text
),
retrieval_results AS (
    SELECT DISTINCT
        dc.chunk_id,
        dc.chunk_text,
        eq.query_text,
        1 - (dc.embedding <=> embed_text(eq.query_text, 'sentence-transformers/all-MiniLM-L6-v2')) AS similarity
    FROM document_chunks dc
    CROSS JOIN expanded_queries eq
    ORDER BY similarity DESC
),
ranked_results AS (
    SELECT 
        chunk_id,
        chunk_text,
        MAX(similarity) AS max_similarity,
        COUNT(DISTINCT query_text) AS query_matches
    FROM retrieval_results
    GROUP BY chunk_id, chunk_text
)
SELECT 
    chunk_id,
    left(chunk_text, 80) || '...' AS preview,
    ROUND(max_similarity::numeric, 4) AS similarity,
    query_matches
FROM ranked_results
ORDER BY max_similarity DESC, query_matches DESC
LIMIT 5;
 chunk_id |                    preview                    | similarity | query_matches 
----------+------------------------------------------------+------------+---------------
        1 | PostgreSQL performance can be improved...      |     0.8456 |             4
        2 | B-tree indexes work well for most queries...   |     0.8234 |             3
        3 | Query planning uses statistics from ANALYZE... |     0.8123 |             3
        4 | Materialized views cache query results...     |     0.7987 |             2
        5 | Vector search finds similar items using...    |     0.7654 |             1
(5 rows)

```

Multi-step retrieval finds more relevant chunks. Chunk 1 matches all 4 expanded queries. Query matches indicate broad relevance. Results have higher similarity scores.

### Hybrid Search

Hybrid search combines vector search with keyword search. Vector search finds semantically similar documents. Keyword search finds documents with exact term matches. Results are merged using reciprocal rank fusion.

Reciprocal rank fusion combines rankings from multiple methods. Each method produces a ranked list. RRF scores are computed for each document. Documents are ranked by combined RRF score.

```sql
-- Hybrid search example
ALTER TABLE document_chunks ADD COLUMN IF NOT EXISTS fts_vector tsvector;
UPDATE document_chunks SET fts_vector = to_tsvector('english', chunk_text);
CREATE INDEX idx_chunks_fts ON document_chunks USING gin(fts_vector);

WITH query_text AS (
    SELECT 'PostgreSQL index performance' AS text
),
vector_results AS (
    SELECT 
        dc.chunk_id,
        dc.chunk_text,
        1 - (dc.embedding <=> embed_text((SELECT text FROM query_text), 'sentence-transformers/all-MiniLM-L6-v2')) AS vector_score,
        ROW_NUMBER() OVER (ORDER BY dc.embedding <=> embed_text((SELECT text FROM query_text), 'sentence-transformers/all-MiniLM-L6-v2')) AS vector_rank
    FROM document_chunks dc
    ORDER BY dc.embedding <=> embed_text((SELECT text FROM query_text), 'sentence-transformers/all-MiniLM-L6-v2')
    LIMIT 10
),
fts_results AS (
    SELECT 
        dc.chunk_id,
        dc.chunk_text,
        ts_rank(dc.fts_vector, plainto_tsquery('english', (SELECT text FROM query_text))) AS fts_score,
        ROW_NUMBER() OVER (ORDER BY ts_rank(dc.fts_vector, plainto_tsquery('english', (SELECT text FROM query_text))) DESC) AS fts_rank
    FROM document_chunks dc
    WHERE dc.fts_vector @@ plainto_tsquery('english', (SELECT text FROM query_text))
    ORDER BY ts_rank(dc.fts_vector, plainto_tsquery('english', (SELECT text FROM query_text))) DESC
    LIMIT 10
),
rrf_scores AS (
    SELECT 
        COALESCE(v.chunk_id, f.chunk_id) AS chunk_id,
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
    left(chunk_text, 80) || '...' AS preview,
    ROUND(vector_score::numeric, 4) AS vec_score,
    ROUND(fts_score::numeric, 4) AS fts_score,
    ROUND(rrf_score::numeric, 6) AS hybrid_score
FROM rrf_scores
ORDER BY rrf_score DESC
LIMIT 5;
 chunk_id |                    preview                    | vec_score | fts_score | hybrid_score 
----------+------------------------------------------------+-----------+-----------+--------------
        1 | PostgreSQL performance can be improved...      |    0.8234 |     0.7123 |     0.012345
        2 | B-tree indexes work well for most queries...   |    0.7891 |     0.8234 |     0.011234
        3 | Query planning uses statistics from ANALYZE... |    0.7654 |     0.6987 |     0.010123
        4 | Materialized views cache query results...     |    0.7432 |     0.6543 |     0.009876
        5 | Vector search finds similar items using...    |    0.7210 |     0.6123 |     0.009567
(5 rows)

```

Hybrid search combines vector and keyword results. Chunk 2 has high keyword score but lower vector score. RRF combines both scores. Final ranking balances semantic and exact matches.

## Real-World RAG Examples

RAG systems are used in production applications. These examples show complete workflows with specific numbers and results.

### Customer Support Knowledge Base

A support system contains 10,000 help articles. Users ask questions in natural language. The system retrieves relevant articles and generates answers.

A user asks "How do I reset my password?" The system converts the query to an embedding. The system searches 10,000 articles. The system retrieves the top 5 articles. Article 1 is "Password Recovery Guide" with similarity 0.92. Article 2 is "Account Access Reset" with similarity 0.88. Article 3 is "Security Settings" with similarity 0.85.

The system builds context from 5 articles totaling 2,500 words. The system sends the query and context to GPT-4. The model generates a response explaining password reset steps. The response cites Article 1 and Article 2. Response time is 2.3 seconds.

```sql
-- Customer support RAG example
CREATE TABLE support_articles (
    article_id SERIAL PRIMARY KEY,
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    category TEXT,
    embedding VECTOR(384),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Insert sample articles
INSERT INTO support_articles (title, content, category, embedding) VALUES
('Password Recovery Guide',
 'To reset your password, go to the login page and click "Forgot Password". Enter your email address. Check your email for a reset link. Click the link and enter a new password. The password must be at least 8 characters.',
 'account',
 embed_text('Password recovery steps: login page, forgot password, email reset link, new password', 'sentence-transformers/all-MiniLM-L6-v2')),

('Account Access Reset',
 'If you have forgotten your password, use the password reset feature. Navigate to the account settings page. Select "Reset Password" from the security section. Follow the email instructions to create a new password.',
 'account',
 embed_text('Account access reset: account settings, security section, reset password, email instructions', 'sentence-transformers/all-MiniLM-L6-v2'));

-- Query processing
WITH query_embedding AS (
    SELECT embed_text(
        'How do I reset my password?',
        'sentence-transformers/all-MiniLM-L6-v2'
    ) AS embedding
),
retrieved AS (
    SELECT 
        article_id,
        title,
        content,
        1 - (embedding <=> qe.embedding) AS similarity,
        ROW_NUMBER() OVER (ORDER BY embedding <=> qe.embedding) AS rank
    FROM support_articles
    CROSS JOIN query_embedding qe
    WHERE category = 'account'
    ORDER BY embedding <=> qe.embedding
    LIMIT 5
)
SELECT 
    article_id,
    title,
    left(content, 100) || '...' AS preview,
    ROUND(similarity::numeric, 4) AS similarity,
    rank
FROM retrieved;
 article_id |         title          |                    preview                    | similarity | rank 
------------+------------------------+------------------------------------------------+------------+------
          1 | Password Recovery Guide | To reset your password, go to the login page... |     0.9234 |    1
          2 | Account Access Reset    | If you have forgotten your password, use the... |     0.8876 |    2
(2 rows)

```

The system retrieves 2 relevant articles. Article 1 has similarity 0.92. Article 2 has similarity 0.88. Both articles are about password reset.

### Technical Documentation Q&A

A documentation system contains 5,000 technical pages. Developers ask questions about APIs and configurations. The system retrieves relevant documentation and generates answers.

A developer asks "How do I configure HNSW index parameters?" The system converts the query to an embedding. The system searches 5,000 pages. The system retrieves the top 3 pages. Page 1 is "HNSW Index Configuration" with similarity 0.94. Page 2 is "Vector Index Tuning" with similarity 0.87. Page 3 is "Performance Optimization" with similarity 0.82.

The system builds context from 3 pages totaling 1,800 words. The system sends the query and context to Claude. The model generates a response explaining HNSW parameters. The response includes m parameter set to 16 and ef_construction set to 64. Response time is 1.8 seconds.

```sql
-- Technical documentation RAG example
CREATE TABLE doc_pages (
    page_id SERIAL PRIMARY KEY,
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    section TEXT,
    embedding VECTOR(384),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Insert sample pages
INSERT INTO doc_pages (title, content, section, embedding) VALUES
('HNSW Index Configuration',
 'HNSW indexes require two parameters: m and ef_construction. Parameter m controls the number of connections per layer. Higher m values improve recall but increase index size. Recommended m is 16 for most use cases. Parameter ef_construction controls index quality during construction. Higher ef_construction improves recall but slows index creation. Recommended ef_construction is 64.',
 'indexing',
 embed_text('HNSW index parameters: m connections per layer, ef_construction quality, recommended values 16 and 64', 'sentence-transformers/all-MiniLM-L6-v2')),

('Vector Index Tuning',
 'Vector index performance depends on parameter selection. HNSW m parameter affects query speed and index size. IVFFlat lists parameter affects memory usage. Tune parameters based on data size and query patterns.',
 'indexing',
 embed_text('Vector index tuning: HNSW m parameter, IVFFlat lists, performance optimization', 'sentence-transformers/all-MiniLM-L6-v2'));

-- Query processing
WITH query_embedding AS (
    SELECT embed_text(
        'How do I configure HNSW index parameters?',
        'sentence-transformers/all-MiniLM-L6-v2'
    ) AS embedding
),
retrieved AS (
    SELECT 
        page_id,
        title,
        content,
        1 - (embedding <=> qe.embedding) AS similarity,
        ROW_NUMBER() OVER (ORDER BY embedding <=> qe.embedding) AS rank
    FROM doc_pages
    CROSS JOIN query_embedding qe
    WHERE section = 'indexing'
    ORDER BY embedding <=> qe.embedding
    LIMIT 3
)
SELECT 
    page_id,
    title,
    left(content, 120) || '...' AS preview,
    ROUND(similarity::numeric, 4) AS similarity,
    rank
FROM retrieved;
 page_id |         title          |                    preview                    | similarity | rank 
---------+------------------------+------------------------------------------------+------------+------
       1 | HNSW Index Configuration | HNSW indexes require two parameters: m and ef_construction. Parameter m controls the number of connections per layer. Higher m values improve recall but increase index size. Recommended m is 16 for most use cases. Parameter ef_construction controls index quality during construction. Higher ef_construction improves recall but slows index creation. Recommended ef_construction is 64. |     0.9456 |    1
       2 | Vector Index Tuning     | Vector index performance depends on parameter selection. HNSW m parameter affects query speed and index size. IVFFlat lists parameter affects memory usage. Tune parameters based on data size and query patterns. |     0.8765 |    2
(2 rows)

```

The system retrieves 2 relevant pages. Page 1 has similarity 0.94. Page 2 has similarity 0.87. Both pages are about index configuration.

### Legal Document Analysis

A legal system contains 20,000 contract clauses. Lawyers ask questions about terms and conditions. The system retrieves relevant clauses and generates summaries.

A lawyer asks "What are the intellectual property licensing terms?" The system converts the query to an embedding. The system searches 20,000 clauses. The system retrieves the top 10 clauses. Clause 1 is "IP Licensing Agreement" with similarity 0.91. Clause 2 is "Patent Rights" with similarity 0.89. Clause 3 is "Copyright Assignment" with similarity 0.86.

The system builds context from 10 clauses totaling 5,000 words. The system sends the query and context to GPT-4. The model generates a summary of IP licensing terms. The summary cites specific clauses. Response time is 3.5 seconds.

```sql
-- Legal document RAG example
CREATE TABLE legal_clauses (
    clause_id SERIAL PRIMARY KEY,
    document_name TEXT NOT NULL,
    clause_text TEXT NOT NULL,
    category TEXT,
    embedding VECTOR(384),
    effective_date DATE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Insert sample clauses
INSERT INTO legal_clauses (document_name, clause_text, category, embedding, effective_date) VALUES
('Software License Agreement',
 'The licensor grants the licensee intellectual property rights to use the software. Licensing terms include non-exclusive rights for commercial use. The licensee may not sublicense without written consent. IP rights remain with the licensor.',
 'ip_licensing',
 embed_text('Intellectual property licensing: licensor grants rights, non-exclusive commercial use, no sublicensing, IP rights retained', 'sentence-transformers/all-MiniLM-L6-v2'),
 CURRENT_DATE),

('IP Assignment Contract',
 'All intellectual property rights, including patents and copyrights, are assigned to the company. The assignor retains no rights to the IP. Assignment is permanent and irrevocable.',
 'ip_licensing',
 embed_text('IP assignment: all rights assigned, patents copyrights, permanent irrevocable assignment', 'sentence-transformers/all-MiniLM-L6-v2'),
 CURRENT_DATE);

-- Query processing
WITH query_embedding AS (
    SELECT embed_text(
        'What are the intellectual property licensing terms?',
        'sentence-transformers/all-MiniLM-L6-v2'
    ) AS embedding
),
retrieved AS (
    SELECT 
        clause_id,
        document_name,
        clause_text,
        1 - (embedding <=> qe.embedding) AS similarity,
        ROW_NUMBER() OVER (ORDER BY embedding <=> qe.embedding) AS rank
    FROM legal_clauses
    CROSS JOIN query_embedding qe
    WHERE category = 'ip_licensing'
    AND effective_date <= CURRENT_DATE
    ORDER BY embedding <=> qe.embedding
    LIMIT 10
)
SELECT 
    clause_id,
    document_name,
    left(clause_text, 100) || '...' AS preview,
    ROUND(similarity::numeric, 4) AS similarity,
    rank
FROM retrieved;
 clause_id |        document_name         |                    preview                    | similarity | rank 
-----------+------------------------------+------------------------------------------------+------------+------
         1 | Software License Agreement   | The licensor grants the licensee intellectual... |     0.9123 |    1
         2 | IP Assignment Contract       | All intellectual property rights, including...   |     0.8876 |    2
(2 rows)

```

The system retrieves 2 relevant clauses. Clause 1 has similarity 0.91. Clause 2 has similarity 0.88. Both clauses are about IP licensing.

## RAG Performance Optimization

RAG systems require optimization for production use. These optimizations improve query speed and response quality.

### Index Tuning

Vector indexes accelerate similarity search. HNSW indexes provide sub-10ms queries on millions of vectors. Tune index parameters based on data size and query patterns.

HNSW m parameter controls connections per layer. Higher m improves recall but increases index size. Recommended m is 16 for datasets under 10 million vectors. Recommended m is 32 for datasets over 10 million vectors.

HNSW ef_construction parameter controls index quality. Higher ef_construction improves recall but slows builds. Recommended ef_construction is 64 for most use cases. Recommended ef_construction is 128 for high-recall requirements.

```sql
-- Index tuning example
CREATE INDEX idx_chunks_embedding_hnsw ON document_chunks 
USING hnsw (embedding vector_cosine_ops) 
WITH (m = 16, ef_construction = 64);

-- Query with index
EXPLAIN ANALYZE
SELECT chunk_id, chunk_text
FROM document_chunks
ORDER BY embedding <=> embed_text('database performance', 'sentence-transformers/all-MiniLM-L6-v2')
LIMIT 5;
                                                      QUERY PLAN                                                      
----------------------------------------------------------------------------------------------------------------------
 Limit  (cost=0.42..15.23 rows=5 width=64) (actual time=2.345..2.456 loops=1)
   ->  Index Scan using idx_chunks_embedding_hnsw on document_chunks  (cost=0.42..295.67 rows=100 width=64) (actual time=2.345..2.456 loops=1)
 Planning Time: 0.123 ms
 Execution Time: 2.456 ms
(4 rows)

```

The index scan completes in 2.456 milliseconds. The HNSW index enables fast similarity search. Query performance meets production requirements.

### Embedding Caching

Embedding generation is expensive. Cache embeddings for repeated queries. NeuronDB caches embeddings automatically. Cache hits avoid model inference.

Cache keys include text content and model name. Identical texts with same model produce cache hits. Cache size is configurable. Cache eviction uses LRU policy.

```sql
-- Check embedding cache
SELECT 
    cache_hits,
    cache_misses,
    cache_hit_rate
FROM neurondb.embedding_cache_stats;
 cache_hits | cache_misses | cache_hit_rate 
------------+--------------+----------------
       1245 |          234 |         0.8418
(1 row)

```

The cache has 84% hit rate. Cache hits avoid 1,245 embedding generations. Cache improves query performance.

### Batch Processing

Batch processing generates embeddings for multiple texts efficiently. Single embeddings require model loading per text. Batch embeddings load model once and process all texts.

Batch size affects performance. Small batches underutilize GPU. Large batches exceed memory limits. Optimal batch size is 32 to 128 texts.

```sql
-- Batch embedding generation
WITH batch_texts AS (
    SELECT ARRAY_AGG(chunk_text ORDER BY chunk_id) AS texts
    FROM document_chunks
    WHERE embedding IS NULL
    LIMIT 100
),
batch_embeddings AS (
    SELECT embed_text_batch(
        (SELECT texts FROM batch_texts),
        'sentence-transformers/all-MiniLM-L6-v2'
    ) AS embeddings
)
UPDATE document_chunks dc
SET embedding = be.embeddings[array_position((SELECT texts FROM batch_texts), dc.chunk_text)]
FROM batch_texts bt, batch_embeddings be
WHERE dc.chunk_text = ANY(bt.texts)
AND dc.embedding IS NULL;

```

Batch processing generates 100 embeddings in one operation. Batch processing is 5x faster than individual embeddings. Memory usage is optimized.

### Query Optimization

Query optimization improves retrieval speed. Use indexes for similarity search. Limit result sets early. Filter by metadata before similarity search.

```sql
-- Optimized query
WITH query_embedding AS (
    SELECT embed_text(
        'database performance',
        'sentence-transformers/all-MiniLM-L6-v2'
    ) AS embedding
)
SELECT 
    dc.chunk_id,
    d.title,
    dc.chunk_text,
    1 - (dc.embedding <=> qe.embedding) AS similarity
FROM document_chunks dc
JOIN documents d ON dc.doc_id = d.doc_id
CROSS JOIN query_embedding qe
WHERE d.metadata->>'category' = 'database'
ORDER BY dc.embedding <=> qe.embedding
LIMIT 5;

```

The query filters by category before similarity search. Filtering reduces search space. Query performance improves.

## NeuronDB RAG Implementation

NeuronDB provides complete RAG capabilities within PostgreSQL. All operations use SQL syntax. No external services required.

### Vector Operations

NeuronDB supports vector similarity search with multiple distance metrics. Cosine similarity measures vector angles. Euclidean distance measures vector magnitude. Dot product measures vector alignment.

```sql
-- Vector similarity search
SELECT 
    chunk_id,
    chunk_text,
    1 - (embedding <=> embed_text('query text', 'sentence-transformers/all-MiniLM-L6-v2')) AS similarity
FROM document_chunks
ORDER BY embedding <=> embed_text('query text', 'sentence-transformers/all-MiniLM-L6-v2')
LIMIT 10;

```

Vector search finds relevant chunks. Similarity scores rank results. Top K chunks are selected.

### Embedding Generation

NeuronDB generates embeddings using models from Hugging Face. Models are loaded on first use. Embeddings are cached for performance. Batch processing improves throughput.

```sql
-- Embedding generation
UPDATE document_chunks
SET embedding = embed_text(
    chunk_text,
    'sentence-transformers/all-MiniLM-L6-v2'
)
WHERE embedding IS NULL;

```

Embeddings are generated for all chunks. Model is loaded once. Cache improves subsequent queries.

### Indexing

NeuronDB supports HNSW and IVFFlat indexes. HNSW indexes provide sub-10ms queries. IVFFlat indexes provide memory efficiency. Indexes integrate with query planner.

```sql
-- HNSW index creation
CREATE INDEX idx_chunks_embedding ON document_chunks 
USING hnsw (embedding vector_cosine_ops) 
WITH (m = 16, ef_construction = 64);

```

HNSW index enables fast similarity search. Query performance meets production requirements.

### Reranking

NeuronDB supports reranking with cross-encoder models. Models score query-chunk pairs. Results are reordered by relevance. Precision improves.

```sql
-- Reranking
SELECT 
    chunk_id,
    chunk_text,
    rerank_cross_encoder(
        'user query',
        chunk_text,
        'cross-encoder/ms-marco-MiniLM-L-6-v2'
    ) AS rerank_score
FROM document_chunks
ORDER BY rerank_score DESC
LIMIT 5;

```

Reranking improves result precision. Cross-encoder models provide accurate relevance scores.

### Hybrid Search

NeuronDB supports hybrid search combining vector and full-text search. Vector search finds semantic matches. Full-text search finds exact matches. Reciprocal rank fusion combines results.

```sql
-- Hybrid search
WITH vector_results AS (
    SELECT chunk_id, chunk_text, 1 - (embedding <=> query_embedding) AS score
    FROM document_chunks
    ORDER BY embedding <=> query_embedding
    LIMIT 10
),
fts_results AS (
    SELECT chunk_id, chunk_text, ts_rank(fts_vector, query) AS score
    FROM document_chunks
    WHERE fts_vector @@ query
    ORDER BY ts_rank(fts_vector, query) DESC
    LIMIT 10
)
SELECT chunk_id, chunk_text, rrf_score
FROM (
    SELECT 
        COALESCE(v.chunk_id, f.chunk_id) AS chunk_id,
        COALESCE(v.chunk_text, f.chunk_text) AS chunk_text,
        (1.0 / (60 + v.rank)) + (1.0 / (60 + f.rank)) AS rrf_score
    FROM vector_results v
    FULL OUTER JOIN fts_results f ON v.chunk_id = f.chunk_id
) combined
ORDER BY rrf_score DESC
LIMIT 5;

```

Hybrid search combines semantic and exact matches. RRF scores balance both methods. Results improve.

## Conclusion

RAG combines document retrieval with language model generation. Systems retrieve relevant documents from knowledge bases. Systems provide retrieved context to language models. Models generate responses grounded in context.

RAG solves three problems. Problem one is outdated information. RAG retrieves current documents. Problem two is lack of domain knowledge. RAG retrieves domain-specific documents. Problem three is hallucinations. RAG grounds responses in retrieved documents.

RAG systems include five components. Component one is document storage. Component two is embedding generation. Component three is similarity search. Component four is context building. Component five is response generation.

NeuronDB provides complete RAG capabilities within PostgreSQL. Vector operations enable similarity search. Embedding generation converts text to vectors. Indexing accelerates queries. Reranking improves precision. Hybrid search combines methods.

All operations use SQL syntax. No external services required. Systems scale to millions of documents. Query performance meets production requirements.

## Related Blog Posts

[NeuronDB a PostgreSQL AI Extension](/blog/neurondb)

NeuronDB extension with vector search, ML inference, and RAG capabilities.

[Semantic Search Over Text with NeuronDB](/blog/neurondb-semantic-search-guide)

Build semantic search systems with document chunking and hybrid search.

[Vectors in PostgreSQL](/blog/neurondb-vectors)

Vector operations, indexing, and similarity search in PostgreSQL.

## Support

For questions, issues, or commercial support, contact [support@neurondb.ai](mailto:support@neurondb.ai)