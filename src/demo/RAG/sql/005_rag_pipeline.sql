-- ============================================================================
-- Test 005: Complete RAG Pipeline
-- ============================================================================
-- Demonstrates: End-to-end RAG flow: Query -> Retrieve -> Augment -> Generate
-- ============================================================================

\echo 'Testing complete RAG pipeline...'

-- Create RAG queries table to store Q&A pairs
CREATE TABLE IF NOT EXISTS rag_queries (
    query_id SERIAL PRIMARY KEY,
    user_query TEXT NOT NULL,
    retrieved_chunks INT[],
    context_text TEXT,
    generated_response TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB
);

\echo ''
\echo 'RAG Pipeline Step 1: Query Processing'
\echo 'User Query: "How can I improve PostgreSQL query performance?"'

-- Store the query
INSERT INTO rag_queries (user_query, metadata)
VALUES (
    'How can I improve PostgreSQL query performance?',
    '{"model": "gpt-4", "temperature": 0.7}'::jsonb
)
RETURNING query_id AS current_query_id \gset

\echo 'Query ID: ' :current_query_id

\echo ''
\echo 'RAG Pipeline Step 2: Retrieve Relevant Context'

-- Retrieve top-k most relevant chunks
WITH query_embedding AS (
    SELECT embed_text('How can I improve PostgreSQL query performance?',
        'sentence-transformers/all-MiniLM-L6-v2'::text
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

\echo ''
\echo 'RAG Pipeline Step 3: Context Augmentation'

-- Build context from retrieved chunks
WITH query_embedding AS (
    SELECT embed_text('How can I improve PostgreSQL query performance?',
        'sentence-transformers/all-MiniLM-L6-v2'::text
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
            E'\n\n'
            ORDER BY rank
        ) AS context
    FROM relevant_chunks
)
UPDATE rag_queries
SET 
    retrieved_chunks = cb.chunk_ids,
    context_text = cb.context
FROM context_build cb
WHERE query_id = :current_query_id
RETURNING 
    left(context_text, 300) || '...' AS context_preview;

\echo ''
\echo 'RAG Pipeline Step 4: Generate Response (Simulated)'
\echo 'In production, this would call an LLM API (OpenAI, Anthropic, etc.)'

-- Simulated LLM response
UPDATE rag_queries
SET generated_response = 'Based on the retrieved documentation, here are key ways to improve PostgreSQL query performance:

1. **Indexing Strategy**: Create appropriate indexes (B-tree, GiST, Hash) based on your query patterns. Use EXPLAIN ANALYZE to identify missing indexes.

2. **Partial Indexes**: For queries with common WHERE clauses, partial indexes can reduce index size and improve performance.

3. **Query Optimization**: Ensure queries use indexes effectively. Avoid SELECT *, use proper WHERE clauses, and leverage index-only scans when possible.

4. **Configuration Tuning**: Adjust shared_buffers, work_mem, and effective_cache_size based on your workload.

These recommendations are grounded in PostgreSQL best practices from the official documentation.'
WHERE query_id = :current_query_id;

\echo ''
\echo 'RAG Pipeline Complete!'
\echo ''
\echo 'Final RAG Result:'
SELECT 
    query_id,
    user_query,
    array_length(retrieved_chunks, 1) AS num_chunks_retrieved,
    left(generated_response, 400) || '...' AS response
FROM rag_queries
WHERE query_id = :current_query_id;

\echo ''
\echo 'Pipeline Metrics:'
SELECT 
    query_id,
    user_query,
    array_length(retrieved_chunks, 1) AS chunks_used,
    length(context_text) AS context_length,
    length(generated_response) AS response_length,
    created_at
FROM rag_queries
WHERE query_id = :current_query_id;

