-- ============================================================================
-- Recipe: Weighted Hybrid Search
-- ============================================================================
-- Purpose: Fine-tune the balance between vector and text search results
-- 
-- Prerequisites:
--   - quickstart_documents table
--   - Full-text search index (from previous recipe)
--   - Embeddings column populated
--
-- Tuning Tips:
--   - Higher vector weight: Better for semantic understanding
--   - Higher text weight: Better for exact keyword matching
--   - Adjust weights based on your data and use case
--
-- Performance Notes:
--   - Weighting doesn't significantly impact query time
--   - Consider caching weighted scores for frequently used weights
-- ============================================================================

-- Example 1: Configurable weights via CTE
-- Easy to adjust vector_weight and text_weight
WITH query_vector AS (
    SELECT embed_text('database performance optimization') AS query_emb
),
weights AS (
    SELECT 0.7 AS vector_weight, 0.3 AS text_weight  -- Adjust these values
),
vector_scores AS (
    SELECT 
        id,
        1 - (embedding <=> query_vector.query_emb) AS vector_score
    FROM quickstart_documents, query_vector
    WHERE embedding IS NOT NULL
),
text_scores AS (
    SELECT 
        id,
        ts_rank(fts_vector, plainto_tsquery('english', 'database performance optimization')) AS text_score
    FROM quickstart_documents
    WHERE fts_vector @@ plainto_tsquery('english', 'database performance optimization')
)
SELECT 
    d.id,
    d.title,
    ROUND(COALESCE(vs.vector_score, 0)::numeric, 4) AS vector_score,
    ROUND(COALESCE(ts.text_score, 0)::numeric, 4) AS text_score,
    ROUND((
        COALESCE(vs.vector_score, 0) * w.vector_weight + 
        COALESCE(ts.text_score, 0) * w.text_weight
    )::numeric, 4) AS hybrid_score
FROM quickstart_documents d
CROSS JOIN weights w
LEFT JOIN vector_scores vs ON d.id = vs.id
LEFT JOIN text_scores ts ON d.id = ts.id
WHERE vs.vector_score IS NOT NULL OR ts.text_score IS NOT NULL
ORDER BY hybrid_score DESC
LIMIT 10;

-- Example 2: Vector-biased search (80% vector, 20% text)
-- Good for semantic similarity focus
WITH query_vector AS (
    SELECT embed_text('neural network architectures') AS query_emb
),
vector_scores AS (
    SELECT 
        id,
        1 - (embedding <=> query_vector.query_emb) AS vector_score
    FROM quickstart_documents, query_vector
    WHERE embedding IS NOT NULL
),
text_scores AS (
    SELECT 
        id,
        ts_rank(fts_vector, plainto_tsquery('english', 'neural network architectures')) AS text_score
    FROM quickstart_documents
    WHERE fts_vector @@ plainto_tsquery('english', 'neural network architectures')
)
SELECT 
    d.id,
    d.title,
    ROUND((COALESCE(vs.vector_score, 0) * 0.8 + COALESCE(ts.text_score, 0) * 0.2)::numeric, 4) AS hybrid_score
FROM quickstart_documents d
LEFT JOIN vector_scores vs ON d.id = vs.id
LEFT JOIN text_scores ts ON d.id = ts.id
WHERE vs.vector_score IS NOT NULL OR ts.text_score IS NOT NULL
ORDER BY hybrid_score DESC
LIMIT 10;

-- Example 3: Text-biased search (30% vector, 70% text)
-- Good for exact keyword matching focus
WITH query_vector AS (
    SELECT embed_text('PostgreSQL indexes') AS query_emb
),
vector_scores AS (
    SELECT 
        id,
        1 - (embedding <=> query_vector.query_emb) AS vector_score
    FROM quickstart_documents, query_vector
    WHERE embedding IS NOT NULL
),
text_scores AS (
    SELECT 
        id,
        ts_rank(fts_vector, plainto_tsquery('english', 'PostgreSQL indexes')) AS text_score
    FROM quickstart_documents
    WHERE fts_vector @@ plainto_tsquery('english', 'PostgreSQL indexes')
)
SELECT 
    d.id,
    d.title,
    ROUND((COALESCE(vs.vector_score, 0) * 0.3 + COALESCE(ts.text_score, 0) * 0.7)::numeric, 4) AS hybrid_score
FROM quickstart_documents d
LEFT JOIN vector_scores vs ON d.id = vs.id
LEFT JOIN text_scores ts ON d.id = ts.id
WHERE vs.vector_score IS NOT NULL OR ts.text_score IS NOT NULL
ORDER BY hybrid_score DESC
LIMIT 10;

-- Example 4: Adaptive weighting based on query type
-- Use different weights based on query characteristics
WITH query_vector AS (
    SELECT 
        embed_text('vector databases') AS query_emb,
        'vector databases' AS query_text
),
query_type AS (
    SELECT 
        CASE 
            WHEN query_text ~ '\m(postgresql|sql|database|index)\M' THEN 'technical'
            ELSE 'semantic'
        END AS type
    FROM query_vector
),
vector_scores AS (
    SELECT 
        id,
        1 - (embedding <=> query_vector.query_emb) AS vector_score
    FROM quickstart_documents, query_vector
    WHERE embedding IS NOT NULL
),
text_scores AS (
    SELECT 
        id,
        ts_rank(fts_vector, plainto_tsquery('english', query_vector.query_text)) AS text_score
    FROM quickstart_documents, query_vector
    WHERE fts_vector @@ plainto_tsquery('english', query_vector.query_text)
)
SELECT 
    d.id,
    d.title,
    qt.type AS query_type,
    CASE 
        WHEN qt.type = 'technical' THEN 
            ROUND((COALESCE(vs.vector_score, 0) * 0.4 + COALESCE(ts.text_score, 0) * 0.6)::numeric, 4)
        ELSE 
            ROUND((COALESCE(vs.vector_score, 0) * 0.8 + COALESCE(ts.text_score, 0) * 0.2)::numeric, 4)
    END AS hybrid_score
FROM quickstart_documents d
CROSS JOIN query_vector qv
CROSS JOIN query_type qt
LEFT JOIN vector_scores vs ON d.id = vs.id
LEFT JOIN text_scores ts ON d.id = ts.id
WHERE vs.vector_score IS NOT NULL OR ts.text_score IS NOT NULL
ORDER BY hybrid_score DESC
LIMIT 10;



