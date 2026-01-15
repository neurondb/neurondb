-- ============================================================================
-- Recipe: Reciprocal Rank Fusion (RRF) for Hybrid Search
-- ============================================================================
-- Purpose: Combine vector and text search results using RRF algorithm
-- 
-- Prerequisites:
--   - quickstart_documents table
--   - Full-text search index
--   - Embeddings column populated
--
-- RRF Algorithm:
--   - Combines results from multiple ranking methods
--   - No need to normalize scores between methods
--   - Simple and effective for hybrid search
--   - Formula: RRF_score = sum(1 / (k + rank)) for each method
--
-- Performance Notes:
--   - k parameter (typically 60) controls how much rank matters
--   - Lower k = rank matters more, higher k = rank matters less
-- ============================================================================

-- Example 1: Basic RRF
-- Standard k=60 for RRF
WITH query_vector AS (
    SELECT embed_text('vector similarity search algorithms') AS query_emb
),
-- Vector search results with ranks
vector_results AS (
    SELECT 
        id,
        ROW_NUMBER() OVER (ORDER BY embedding <=> query_vector.query_emb) AS vector_rank
    FROM quickstart_documents, query_vector
    WHERE embedding IS NOT NULL
    ORDER BY embedding <=> query_vector.query_emb
    LIMIT 50
),
-- Text search results with ranks
text_results AS (
    SELECT 
        id,
        ROW_NUMBER() OVER (ORDER BY ts_rank(fts_vector, plainto_tsquery('english', 'vector similarity search algorithms')) DESC) AS text_rank
    FROM quickstart_documents
    WHERE fts_vector @@ plainto_tsquery('english', 'vector similarity search algorithms')
    ORDER BY ts_rank(fts_vector, plainto_tsquery('english', 'vector similarity search algorithms')) DESC
    LIMIT 50
),
-- Calculate RRF scores
rrf_scores AS (
    SELECT 
        COALESCE(vr.id, tr.id) AS id,
        COALESCE(vr.vector_rank, 1000) AS vector_rank,  -- High rank if not in results
        COALESCE(tr.text_rank, 1000) AS text_rank,
        -- RRF formula: 1/(k+rank) for each method, summed
        (1.0 / (60.0 + COALESCE(vr.vector_rank, 1000))) + 
        (1.0 / (60.0 + COALESCE(tr.text_rank, 1000))) AS rrf_score
    FROM vector_results vr
    FULL OUTER JOIN text_results tr ON vr.id = tr.id
)
SELECT 
    d.id,
    d.title,
    rs.vector_rank,
    rs.text_rank,
    ROUND(rs.rrf_score::numeric, 6) AS rrf_score
FROM quickstart_documents d
JOIN rrf_scores rs ON d.id = rs.id
ORDER BY rs.rrf_score DESC
LIMIT 10;

-- Example 2: RRF with configurable k parameter
-- Adjust k to control rank importance
WITH query_vector AS (
    SELECT embed_text('machine learning embeddings') AS query_emb
),
rrf_k AS (
    SELECT 60 AS k  -- Adjust this: lower = rank matters more
),
vector_results AS (
    SELECT 
        id,
        ROW_NUMBER() OVER (ORDER BY embedding <=> query_vector.query_emb) AS vector_rank
    FROM quickstart_documents, query_vector
    WHERE embedding IS NOT NULL
    ORDER BY embedding <=> query_vector.query_emb
    LIMIT 50
),
text_results AS (
    SELECT 
        id,
        ROW_NUMBER() OVER (ORDER BY ts_rank(fts_vector, plainto_tsquery('english', 'machine learning embeddings')) DESC) AS text_rank
    FROM quickstart_documents
    WHERE fts_vector @@ plainto_tsquery('english', 'machine learning embeddings')
    ORDER BY ts_rank(fts_vector, plainto_tsquery('english', 'machine learning embeddings')) DESC
    LIMIT 50
),
rrf_scores AS (
    SELECT 
        COALESCE(vr.id, tr.id) AS id,
        (1.0 / (rk.k + COALESCE(vr.vector_rank, 1000))) + 
        (1.0 / (rk.k + COALESCE(tr.text_rank, 1000))) AS rrf_score
    FROM vector_results vr
    FULL OUTER JOIN text_results tr ON vr.id = tr.id
    CROSS JOIN rrf_k rk
)
SELECT 
    d.id,
    d.title,
    ROUND(rs.rrf_score::numeric, 6) AS rrf_score
FROM quickstart_documents d
JOIN rrf_scores rs ON d.id = rs.id
ORDER BY rs.rrf_score DESC
LIMIT 10;

-- Example 3: RRF with multiple ranking methods
-- Can extend to 3+ methods (e.g., category-based ranking)
WITH query_vector AS (
    SELECT embed_text('database optimization') AS query_emb
),
vector_results AS (
    SELECT 
        id,
        ROW_NUMBER() OVER (ORDER BY embedding <=> query_vector.query_emb) AS vector_rank
    FROM quickstart_documents, query_vector
    WHERE embedding IS NOT NULL
    ORDER BY embedding <=> query_vector.query_emb
    LIMIT 50
),
text_results AS (
    SELECT 
        id,
        ROW_NUMBER() OVER (ORDER BY ts_rank(fts_vector, plainto_tsquery('english', 'database optimization')) DESC) AS text_rank
    FROM quickstart_documents
    WHERE fts_vector @@ plainto_tsquery('english', 'database optimization')
    ORDER BY ts_rank(fts_vector, plainto_tsquery('english', 'database optimization')) DESC
    LIMIT 50
),
category_results AS (
    SELECT 
        id,
        ROW_NUMBER() OVER (ORDER BY CASE WHEN category = 'database' THEN 1 ELSE 2 END) AS category_rank
    FROM quickstart_documents
    WHERE category IS NOT NULL
    LIMIT 50
),
rrf_scores AS (
    SELECT 
        COALESCE(vr.id, COALESCE(tr.id, cr.id)) AS id,
        (1.0 / (60.0 + COALESCE(vr.vector_rank, 1000))) + 
        (1.0 / (60.0 + COALESCE(tr.text_rank, 1000))) +
        (1.0 / (60.0 + COALESCE(cr.category_rank, 1000))) AS rrf_score
    FROM vector_results vr
    FULL OUTER JOIN text_results tr ON vr.id = tr.id
    FULL OUTER JOIN category_results cr ON COALESCE(vr.id, tr.id) = cr.id
)
SELECT 
    d.id,
    d.title,
    d.category,
    ROUND(rs.rrf_score::numeric, 6) AS rrf_score
FROM quickstart_documents d
JOIN rrf_scores rs ON d.id = rs.id
ORDER BY rs.rrf_score DESC
LIMIT 10;




