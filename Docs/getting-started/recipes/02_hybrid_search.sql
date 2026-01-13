-- ============================================================================
-- SQL Recipe Library: Hybrid Search
-- ============================================================================
-- Ready-to-run queries for combining vector search with full-text search
-- 
-- Prerequisites:
--   - NeuronDB extension installed
--   - Quickstart data pack loaded (or your own table with embeddings)
--   - Full-text search indexes (GIN indexes on tsvector columns)
--
-- Usage:
--   psql -f 02_hybrid_search.sql
--   Or copy individual queries to run them
--
-- Table: quickstart_documents (from quickstart data pack)
-- ============================================================================

-- ============================================================================
-- Setup: Add full-text search support to quickstart_documents
-- ============================================================================
-- Run this once to enable full-text search on the quickstart table

-- Add tsvector column for full-text search
ALTER TABLE quickstart_documents ADD COLUMN IF NOT EXISTS fts_vector tsvector;

-- Populate full-text search vectors from title and content
UPDATE quickstart_documents
SET fts_vector = to_tsvector('english', COALESCE(title, '') || ' ' || COALESCE(content, ''))
WHERE fts_vector IS NULL;

-- Create GIN index for full-text search (if not exists)
CREATE INDEX IF NOT EXISTS idx_quickstart_fts 
ON quickstart_documents USING gin(fts_vector);

-- ============================================================================
-- Recipe 1: Basic Hybrid Search (Weighted Combination)
-- ============================================================================
-- Use case: Combine vector similarity with full-text search scores
-- Complexity: ⭐⭐
-- Method: Weighted linear combination

-- Search using both vector and full-text, weighted 70% vector, 30% full-text
WITH query_text AS (
    SELECT 'machine learning' AS query
),
vector_results AS (
    SELECT 
        id,
        title,
        LEFT(content, 100) AS preview,
        1 - (embedding <=> (SELECT embedding FROM quickstart_documents WHERE title ILIKE '%Machine Learning%' LIMIT 1)) AS vector_score
    FROM quickstart_documents
    ORDER BY embedding <=> (SELECT embedding FROM quickstart_documents WHERE title ILIKE '%Machine Learning%' LIMIT 1)
    LIMIT 20
),
fts_results AS (
    SELECT 
        id,
        title,
        LEFT(content, 100) AS preview,
        ts_rank(fts_vector, plainto_tsquery('english', (SELECT query FROM query_text))) AS fts_score
    FROM quickstart_documents
    WHERE fts_vector @@ plainto_tsquery('english', (SELECT query FROM query_text))
    LIMIT 20
)
SELECT 
    COALESCE(v.id, f.id) AS id,
    COALESCE(v.title, f.title) AS title,
    COALESCE(v.preview, f.preview) AS preview,
    COALESCE(v.vector_score, 0) AS vector_score,
    COALESCE(f.fts_score, 0) AS fts_score,
    (COALESCE(v.vector_score, 0) * 0.7 + COALESCE(f.fts_score, 0) * 0.3) AS hybrid_score
FROM vector_results v
FULL OUTER JOIN fts_results f ON v.id = f.id
ORDER BY hybrid_score DESC
LIMIT 10;

-- ============================================================================
-- Recipe 2: Reciprocal Rank Fusion (RRF)
-- ============================================================================
-- Use case: Combine results from vector and full-text search using RRF
-- Complexity: ⭐⭐⭐
-- Method: RRF - more robust than weighted combination

-- RRF formula: score = 1/(k + rank1) + 1/(k + rank2)
-- where k is a constant (typically 60)
WITH query_doc_id AS (
    SELECT 1 AS qid  -- Use document #1 as query
),
vector_results AS (
    SELECT 
        id,
        title,
        LEFT(content, 100) AS preview,
        embedding <=> (SELECT embedding FROM quickstart_documents WHERE id = (SELECT qid FROM query_doc_id)) AS distance,
        ROW_NUMBER() OVER (ORDER BY embedding <=> (SELECT embedding FROM quickstart_documents WHERE id = (SELECT qid FROM query_doc_id))) AS vector_rank
    FROM quickstart_documents
    WHERE id != (SELECT qid FROM query_doc_id)
    ORDER BY embedding <=> (SELECT embedding FROM quickstart_documents WHERE id = (SELECT qid FROM query_doc_id))
    LIMIT 20
),
fts_results AS (
    SELECT 
        id,
        title,
        LEFT(content, 100) AS preview,
        ts_rank(fts_vector, (SELECT to_tsvector('english', content) FROM quickstart_documents WHERE id = (SELECT qid FROM query_doc_id))) AS fts_score,
        ROW_NUMBER() OVER (ORDER BY ts_rank(fts_vector, (SELECT to_tsvector('english', content) FROM quickstart_documents WHERE id = (SELECT qid FROM query_doc_id)) DESC)) AS fts_rank
    FROM quickstart_documents
    WHERE id != (SELECT qid FROM query_doc_id)
      AND fts_vector @@ (SELECT to_tsvector('english', content) FROM quickstart_documents WHERE id = (SELECT qid FROM query_doc_id))
    ORDER BY fts_score DESC
    LIMIT 20
),
rrf_scores AS (
    SELECT 
        COALESCE(v.id, f.id) AS id,
        COALESCE(v.title, f.title) AS title,
        COALESCE(v.preview, f.preview) AS preview,
        COALESCE(v.vector_rank, 1000) AS vector_rank,
        COALESCE(f.fts_rank, 1000) AS fts_rank,
        (1.0 / (60 + COALESCE(v.vector_rank, 1000))) + (1.0 / (60 + COALESCE(f.fts_rank, 1000))) AS rrf_score
    FROM vector_results v
    FULL OUTER JOIN fts_results f ON v.id = f.id
)
SELECT 
    id,
    title,
    preview,
    vector_rank,
    fts_rank,
    ROUND(rrf_score::numeric, 6) AS hybrid_score
FROM rrf_scores
ORDER BY rrf_score DESC
LIMIT 10;

-- ============================================================================
-- Recipe 3: Hybrid Search with Query Text
-- ============================================================================
-- Use case: Search using both vector embedding and keyword matching
-- Complexity: ⭐⭐
-- Note: In production, use embed_text() to generate query vector from text

WITH query_text AS (
    SELECT 'neural networks deep learning' AS query
),
query_vector AS (
    -- In production, use: embed_text((SELECT query FROM query_text))
    SELECT embedding AS q_vec FROM quickstart_documents WHERE title ILIKE '%Neural%' LIMIT 1
),
vector_results AS (
    SELECT 
        id,
        title,
        LEFT(content, 100) AS preview,
        1 - (embedding <=> (SELECT q_vec FROM query_vector)) AS vector_score,
        ROW_NUMBER() OVER (ORDER BY embedding <=> (SELECT q_vec FROM query_vector)) AS vector_rank
    FROM quickstart_documents
    ORDER BY embedding <=> (SELECT q_vec FROM query_vector)
    LIMIT 20
),
fts_results AS (
    SELECT 
        id,
        title,
        LEFT(content, 100) AS preview,
        ts_rank(fts_vector, plainto_tsquery('english', (SELECT query FROM query_text))) AS fts_score,
        ROW_NUMBER() OVER (ORDER BY ts_rank(fts_vector, plainto_tsquery('english', (SELECT query FROM query_text)) DESC)) AS fts_rank
    FROM quickstart_documents
    WHERE fts_vector @@ plainto_tsquery('english', (SELECT query FROM query_text))
    LIMIT 20
)
SELECT 
    COALESCE(v.id, f.id) AS id,
    COALESCE(v.title, f.title) AS title,
    COALESCE(v.preview, f.preview) AS preview,
    COALESCE(v.vector_score, 0) AS vector_score,
    COALESCE(f.fts_score, 0) AS fts_score,
    (1.0 / (60 + COALESCE(v.vector_rank, 1000))) + (1.0 / (60 + COALESCE(f.fts_rank, 1000))) AS rrf_score
FROM vector_results v
FULL OUTER JOIN fts_results f ON v.id = f.id
ORDER BY rrf_score DESC
LIMIT 10;

-- ============================================================================
-- Recipe 4: Hybrid Search with Minimum Thresholds
-- ============================================================================
-- Use case: Only include results that meet minimum scores from both methods
-- Complexity: ⭐⭐⭐

-- Only include results that score above threshold in both vector and full-text
WITH query_doc_id AS (
    SELECT 1 AS qid
),
vector_results AS (
    SELECT 
        id,
        title,
        embedding <=> (SELECT embedding FROM quickstart_documents WHERE id = (SELECT qid FROM query_doc_id)) AS distance,
        1 - (embedding <=> (SELECT embedding FROM quickstart_documents WHERE id = (SELECT qid FROM query_doc_id))) AS vector_score
    FROM quickstart_documents
    WHERE id != (SELECT qid FROM query_doc_id)
      AND embedding <=> (SELECT embedding FROM quickstart_documents WHERE id = (SELECT qid FROM query_doc_id)) < 0.8  -- Distance threshold
),
fts_results AS (
    SELECT 
        id,
        title,
        ts_rank(fts_vector, (SELECT to_tsvector('english', content) FROM quickstart_documents WHERE id = (SELECT qid FROM query_doc_id))) AS fts_score
    FROM quickstart_documents
    WHERE id != (SELECT qid FROM query_doc_id)
      AND fts_vector @@ (SELECT to_tsvector('english', content) FROM quickstart_documents WHERE id = (SELECT qid FROM query_doc_id))
      AND ts_rank(fts_vector, (SELECT to_tsvector('english', content) FROM quickstart_documents WHERE id = (SELECT qid FROM query_doc_id))) > 0.1  -- FTS threshold
)
SELECT 
    v.id,
    v.title,
    v.vector_score,
    COALESCE(f.fts_score, 0) AS fts_score,
    (v.vector_score * 0.7 + COALESCE(f.fts_score, 0) * 0.3) AS hybrid_score
FROM vector_results v
INNER JOIN fts_results f ON v.id = f.id  -- INNER JOIN ensures both thresholds met
ORDER BY hybrid_score DESC
LIMIT 10;

-- ============================================================================
-- Recipe 5: Hybrid Search with Boosted Fields
-- ============================================================================
-- Use case: Give more weight to matches in title vs content
-- Complexity: ⭐⭐⭐

-- Boost title matches in full-text search
WITH query_text AS (
    SELECT 'machine learning' AS query
),
query_vector AS (
    SELECT embedding AS q_vec FROM quickstart_documents WHERE title ILIKE '%Machine Learning%' LIMIT 1
),
vector_results AS (
    SELECT 
        id,
        title,
        content,
        1 - (embedding <=> (SELECT q_vec FROM query_vector)) AS vector_score
    FROM quickstart_documents
    ORDER BY embedding <=> (SELECT q_vec FROM query_vector)
    LIMIT 20
),
fts_results AS (
    SELECT 
        id,
        title,
        content,
        -- Boost title matches with higher weight
        ts_rank(
            to_tsvector('english', COALESCE(title, '')),
            plainto_tsquery('english', (SELECT query FROM query_text))
        ) * 2.0 +  -- 2x weight for title
        ts_rank(
            to_tsvector('english', COALESCE(content, '')),
            plainto_tsquery('english', (SELECT query FROM query_text))
        ) AS fts_score
    FROM quickstart_documents
    WHERE to_tsvector('english', COALESCE(title, '') || ' ' || COALESCE(content, '')) @@ plainto_tsquery('english', (SELECT query FROM query_text))
    LIMIT 20
)
SELECT 
    COALESCE(v.id, f.id) AS id,
    COALESCE(v.title, f.title) AS title,
    COALESCE(v.vector_score, 0) AS vector_score,
    COALESCE(f.fts_score, 0) AS fts_score,
    (COALESCE(v.vector_score, 0) * 0.7 + COALESCE(f.fts_score, 0) * 0.3) AS hybrid_score
FROM vector_results v
FULL OUTER JOIN fts_results f ON v.id = f.id
ORDER BY hybrid_score DESC
LIMIT 10;

-- ============================================================================
-- Recipe 6: Hybrid Search Performance Comparison
-- ============================================================================
-- Use case: Compare performance of vector-only, FTS-only, and hybrid search
-- Complexity: ⭐⭐⭐

-- Compare search methods
\timing on

-- Vector-only search
SELECT COUNT(*) AS vector_results
FROM (
    SELECT id
    FROM quickstart_documents
    WHERE id != 1
    ORDER BY embedding <=> (SELECT embedding FROM quickstart_documents WHERE id = 1)
    LIMIT 10
) v;

-- FTS-only search
SELECT COUNT(*) AS fts_results
FROM (
    SELECT id
    FROM quickstart_documents
    WHERE id != 1
      AND fts_vector @@ (SELECT to_tsvector('english', content) FROM quickstart_documents WHERE id = 1)
    ORDER BY ts_rank(fts_vector, (SELECT to_tsvector('english', content) FROM quickstart_documents WHERE id = 1)) DESC
    LIMIT 10
) f;

-- Hybrid search (RRF)
SELECT COUNT(*) AS hybrid_results
FROM (
    WITH vector_results AS (
        SELECT id, ROW_NUMBER() OVER (ORDER BY embedding <=> (SELECT embedding FROM quickstart_documents WHERE id = 1)) AS r
        FROM quickstart_documents WHERE id != 1 LIMIT 20
    ),
    fts_results AS (
        SELECT id, ROW_NUMBER() OVER (ORDER BY ts_rank(fts_vector, (SELECT to_tsvector('english', content) FROM quickstart_documents WHERE id = 1)) DESC) AS r
        FROM quickstart_documents WHERE id != 1 AND fts_vector @@ (SELECT to_tsvector('english', content) FROM quickstart_documents WHERE id = 1) LIMIT 20
    )
    SELECT COALESCE(v.id, f.id) AS id, (1.0 / (60 + COALESCE(v.r, 1000))) + (1.0 / (60 + COALESCE(f.r, 1000))) AS score
    FROM vector_results v FULL OUTER JOIN fts_results f ON v.id = f.id
    ORDER BY score DESC LIMIT 10
) h;

\timing off

-- ============================================================================
-- Notes:
-- ============================================================================
-- 
-- 1. Hybrid search combines the strengths of semantic (vector) and keyword 
--    (full-text) search for better results.
--
-- 2. Weighted combination is simpler but RRF (Reciprocal Rank Fusion) often
--    performs better, especially when result sets don't overlap much.
--
-- 3. Tune the weights (e.g., 0.7 vector, 0.3 FTS) based on your use case:
--    - More vector weight: Better for semantic/synonym matching
--    - More FTS weight: Better for exact keyword matching
--
-- 4. In production, use embed_text() to generate query vectors from text:
--    WITH q AS (SELECT embed_text('your query') AS query_vec)
--    ...
--
-- 5. Ensure you have both HNSW index on embeddings and GIN index on tsvector
--    for optimal performance.
--
-- ============================================================================



