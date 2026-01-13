-- ============================================================================
-- Recipe: Context Retrieval for RAG
-- ============================================================================
-- Purpose: Retrieve relevant context chunks for RAG queries
-- 
-- Prerequisites:
--   - document_chunks table with embeddings (from chunking recipe)
--   - HNSW index on chunk embeddings
--
-- RAG Context Retrieval:
--   1. Convert query to embedding
--   2. Find similar chunks via vector search
--   3. Optionally filter by metadata
--   4. Return top-k chunks as context
--
-- Performance Notes:
--   - Use HNSW index for fast retrieval
--   - Typical k values: 3-10 chunks
--   - Consider reranking for better quality
-- ============================================================================

-- Example 1: Basic context retrieval
-- Get top-k similar chunks for a query
WITH query_vector AS (
    SELECT embed_text('How do vector databases work?') AS query_emb
)
SELECT 
    dc.chunk_id,
    d.title AS document_title,
    dc.chunk_index,
    dc.chunk_text,
    1 - (dc.embedding <=> query_vector.query_emb) AS similarity_score,
    d.id AS doc_id
FROM document_chunks dc
JOIN quickstart_documents d ON dc.doc_id = d.id
CROSS JOIN query_vector
WHERE dc.embedding IS NOT NULL
ORDER BY dc.embedding <=> query_vector.query_emb
LIMIT 5;  -- Top 5 chunks as context

-- Example 2: Context retrieval with metadata filtering
-- Filter chunks by document category or other metadata
WITH query_vector AS (
    SELECT embed_text('database indexing strategies') AS query_emb
)
SELECT 
    dc.chunk_id,
    d.title,
    d.category,
    dc.chunk_text,
    1 - (dc.embedding <=> query_vector.query_emb) AS similarity
FROM document_chunks dc
JOIN quickstart_documents d ON dc.doc_id = d.id
CROSS JOIN query_vector
WHERE dc.embedding IS NOT NULL
  AND d.category = 'database'  -- Filter by category
ORDER BY dc.embedding <=> query_vector.query_emb
LIMIT 5;

-- Example 3: Context retrieval with diversity
-- Get chunks from different documents (avoid duplicate sources)
WITH query_vector AS (
    SELECT embed_text('machine learning best practices') AS query_emb
),
ranked_chunks AS (
    SELECT 
        dc.*,
        d.title,
        d.id AS doc_id,
        1 - (dc.embedding <=> query_vector.query_emb) AS similarity,
        ROW_NUMBER() OVER (
            PARTITION BY d.id 
            ORDER BY dc.embedding <=> query_vector.query_emb
        ) AS doc_rank
    FROM document_chunks dc
    JOIN quickstart_documents d ON dc.doc_id = d.id
    CROSS JOIN query_vector
    WHERE dc.embedding IS NOT NULL
)
SELECT 
    chunk_id,
    title,
    chunk_text,
    similarity
FROM ranked_chunks
WHERE doc_rank = 1  -- Top chunk from each document
ORDER BY similarity DESC
LIMIT 5;

-- Example 4: Context retrieval with minimum similarity threshold
-- Only return chunks above a similarity threshold
WITH query_vector AS (
    SELECT embed_text('vector similarity search') AS query_emb
)
SELECT 
    dc.chunk_id,
    d.title,
    dc.chunk_text,
    1 - (dc.embedding <=> query_vector.query_emb) AS similarity
FROM document_chunks dc
JOIN quickstart_documents d ON dc.doc_id = d.id
CROSS JOIN query_vector
WHERE dc.embedding IS NOT NULL
  AND (1 - (dc.embedding <=> query_vector.query_emb)) > 0.7  -- 70% similarity threshold
ORDER BY dc.embedding <=> query_vector.query_emb;

-- Example 5: Multi-query context retrieval
-- Use multiple query variations for better coverage
WITH query_variations AS (
    SELECT 
        query_text,
        embed_text(query_text) AS query_emb
    FROM (VALUES 
        ('How do vector databases work?'),
        ('What are vector databases?'),
        ('vector database explanation')
    ) AS q(query_text)
),
all_results AS (
    SELECT DISTINCT
        dc.chunk_id,
        d.title,
        dc.chunk_text,
        MIN(dc.embedding <=> qv.query_emb) AS min_distance,
        1 - MIN(dc.embedding <=> qv.query_emb) AS max_similarity
    FROM document_chunks dc
    JOIN quickstart_documents d ON dc.doc_id = d.id
    CROSS JOIN query_variations qv
    WHERE dc.embedding IS NOT NULL
    GROUP BY dc.chunk_id, d.title, dc.chunk_text
)
SELECT 
    chunk_id,
    title,
    chunk_text,
    ROUND(max_similarity::numeric, 4) AS similarity
FROM all_results
ORDER BY min_distance
LIMIT 5;

-- Example 6: Context retrieval with chunk ordering
-- Return chunks in document order for better narrative flow
WITH query_vector AS (
    SELECT embed_text('PostgreSQL performance tuning') AS query_emb
),
top_chunks AS (
    SELECT 
        dc.chunk_id,
        dc.doc_id,
        dc.chunk_index,
        d.title,
        dc.chunk_text,
        1 - (dc.embedding <=> query_vector.query_emb) AS similarity
    FROM document_chunks dc
    JOIN quickstart_documents d ON dc.doc_id = d.id
    CROSS JOIN query_vector
    WHERE dc.embedding IS NOT NULL
    ORDER BY dc.embedding <=> query_vector.query_emb
    LIMIT 10
)
SELECT 
    chunk_id,
    title,
    chunk_index,
    chunk_text,
    similarity
FROM top_chunks
ORDER BY doc_id, chunk_index;  -- Ordered by document position

-- Example 7: Format context for LLM
-- Prepare retrieved chunks as a formatted context string
WITH query_vector AS (
    SELECT embed_text('database indexing') AS query_emb
),
context_chunks AS (
    SELECT 
        d.title,
        dc.chunk_text,
        1 - (dc.embedding <=> query_vector.query_emb) AS similarity
    FROM document_chunks dc
    JOIN quickstart_documents d ON dc.doc_id = d.id
    CROSS JOIN query_vector
    WHERE dc.embedding IS NOT NULL
    ORDER BY dc.embedding <=> query_vector.query_emb
    LIMIT 5
)
SELECT 
    string_agg(
        format('Document: %s\nContent: %s\n---', title, chunk_text),
        E'\n\n'
    ) AS formatted_context
FROM context_chunks;



