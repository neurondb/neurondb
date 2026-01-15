-- ============================================================================
-- Test 004: Hybrid Search (Vector + Full-Text)
-- ============================================================================
-- Demonstrates: Combining semantic search with keyword search, RRF fusion
-- ============================================================================

\echo 'Creating full-text search indexes...'

-- Add tsvector column for full-text search
ALTER TABLE document_chunks ADD COLUMN IF NOT EXISTS fts_vector tsvector;

-- Populate full-text search vectors
UPDATE document_chunks
SET fts_vector = to_tsvector('english', chunk_text);

-- Create GIN index for full-text search
CREATE INDEX IF NOT EXISTS idx_chunks_fts ON document_chunks USING gin(fts_vector);

\echo ''
\echo 'Testing hybrid search: Vector + Full-Text...'

-- Hybrid Search Query 1: PostgreSQL performance
\echo ''
\echo 'Query 1: "PostgreSQL index performance"'

-- Vector search results
WITH vector_results AS (
    SELECT 
        dc.chunk_id,
        d.title,
        dc.chunk_text,
        1 - (dc.embedding <=> embed_text('PostgreSQL index performance',
            'sentence-transformers/all-MiniLM-L6-v2'::text
        )) AS vector_score,
        ROW_NUMBER() OVER (ORDER BY dc.embedding <=> embed_text('PostgreSQL index performance',
            'sentence-transformers/all-MiniLM-L6-v2'::text
        )) AS vector_rank
    FROM document_chunks dc
    JOIN documents d ON dc.doc_id = d.doc_id
    ORDER BY dc.embedding <=> embed_text('PostgreSQL index performance',
        'sentence-transformers/all-MiniLM-L6-v2'::text
    )
    LIMIT 10
),
-- Full-text search results
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
-- Reciprocal Rank Fusion (RRF)
rrf_scores AS (
    SELECT 
        COALESCE(v.chunk_id, f.chunk_id) AS chunk_id,
        COALESCE(v.title, f.title) AS title,
        COALESCE(v.chunk_text, f.chunk_text) AS chunk_text,
        COALESCE(v.vector_score, 0) AS vector_score,
        COALESCE(f.fts_score, 0) AS fts_score,
        (1.0 / (60 + COALESCE(v.vector_rank, 1000))) + (1.0 / (60 + COALESCE(f.fts_rank, 1000))) AS rrf_score
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

-- Hybrid Search Query 2: Machine learning
\echo ''
\echo 'Query 2: "machine learning embeddings"'
WITH vector_results AS (
    SELECT 
        dc.chunk_id,
        ROUND((1 - (dc.embedding <=> embed_text('machine learning embeddings',
            'sentence-transformers/all-MiniLM-L6-v2'::text
        )))::numeric, 4) AS vector_score,
        ROW_NUMBER() OVER (ORDER BY dc.embedding <=> embed_text('machine learning embeddings',
            'sentence-transformers/all-MiniLM-L6-v2'::text
        )) AS vector_rank
    FROM document_chunks dc
    LIMIT 10
),
fts_results AS (
    SELECT 
        dc.chunk_id,
        ROUND(ts_rank(dc.fts_vector, plainto_tsquery('english', 'machine learning embeddings'))::numeric, 4) AS fts_score,
        ROW_NUMBER() OVER (ORDER BY ts_rank(dc.fts_vector, plainto_tsquery('english', 'machine learning embeddings')) DESC) AS fts_rank
    FROM document_chunks dc
    WHERE dc.fts_vector @@ plainto_tsquery('english', 'machine learning embeddings')
    LIMIT 10
)
SELECT 
    v.chunk_id,
    v.vector_score,
    COALESCE(f.fts_score, 0) AS fts_score,
    ROUND((v.vector_score * 0.7 + COALESCE(f.fts_score, 0) * 0.3)::numeric, 4) AS weighted_score
FROM vector_results v
LEFT JOIN fts_results f ON v.chunk_id = f.chunk_id
ORDER BY weighted_score DESC
LIMIT 5;

\echo ''
\echo 'Hybrid search comparison complete!'

