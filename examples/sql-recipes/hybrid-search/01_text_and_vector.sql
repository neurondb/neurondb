-- ============================================================================
-- Recipe: Basic Hybrid Search (Text + Vector)
-- ============================================================================
-- Purpose: Combine PostgreSQL full-text search with vector similarity search
-- 
-- Prerequisites:
--   - quickstart_documents table
--   - Embeddings column populated
--   - Full-text search index (GIN) on text column (created in this recipe)
--
-- Use Cases:
--   - Combine keyword matching with semantic understanding
--   - Get best of both worlds: exact matches + semantic similarity
--   - Better results than either method alone
--
-- Performance Notes:
--   - Requires both vector and full-text indexes
--   - Can be slower than single-method search
-- ============================================================================

-- Step 1: Create full-text search index
-- Add tsvector column for full-text search (if not exists)
ALTER TABLE quickstart_documents 
ADD COLUMN IF NOT EXISTS fts_vector tsvector;

-- Populate full-text search vectors
UPDATE quickstart_documents
SET fts_vector = to_tsvector('english', title || ' ' || content)
WHERE fts_vector IS NULL;

-- Create GIN index for fast full-text search
CREATE INDEX IF NOT EXISTS quickstart_documents_fts_idx 
ON quickstart_documents USING gin(fts_vector);

-- Example 1: Separate vector and text results, then combine
-- Get top results from both methods and union them
WITH query_vector AS (
    SELECT embed_text('database indexing strategies') AS query_emb
),
vector_results AS (
    SELECT 
        id,
        title,
        'vector' AS source,
        1 - (embedding <=> query_vector.query_emb) AS score
    FROM quickstart_documents, query_vector
    WHERE embedding IS NOT NULL
    ORDER BY embedding <=> query_vector.query_emb
    LIMIT 10
),
text_results AS (
    SELECT 
        id,
        title,
        'text' AS source,
        ts_rank(fts_vector, plainto_tsquery('english', 'database indexing strategies')) AS score
    FROM quickstart_documents
    WHERE fts_vector @@ plainto_tsquery('english', 'database indexing strategies')
    ORDER BY score DESC
    LIMIT 10
)
SELECT DISTINCT ON (id)
    id,
    title,
    source,
    ROUND(score::numeric, 4) AS score
FROM (
    SELECT * FROM vector_results
    UNION ALL
    SELECT * FROM text_results
) combined
ORDER BY id, score DESC;

-- Example 2: Weighted combination
-- Combine scores from both methods with weights
WITH query_vector AS (
    SELECT embed_text('machine learning embeddings') AS query_emb
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
        ts_rank(fts_vector, plainto_tsquery('english', 'machine learning embeddings')) AS text_score
    FROM quickstart_documents
    WHERE fts_vector @@ plainto_tsquery('english', 'machine learning embeddings')
)
SELECT 
    d.id,
    d.title,
    COALESCE(vs.vector_score, 0) AS vector_score,
    COALESCE(ts.text_score, 0) AS text_score,
    -- Weighted combination: 70% vector, 30% text
    ROUND((COALESCE(vs.vector_score, 0) * 0.7 + COALESCE(ts.text_score, 0) * 0.3)::numeric, 4) AS hybrid_score
FROM quickstart_documents d
LEFT JOIN vector_scores vs ON d.id = vs.id
LEFT JOIN text_scores ts ON d.id = ts.id
WHERE vs.vector_score IS NOT NULL OR ts.text_score IS NOT NULL
ORDER BY hybrid_score DESC
LIMIT 10;

-- Example 3: Intersection approach
-- Only return documents that appear in both result sets
WITH query_vector AS (
    SELECT embed_text('vector similarity search') AS query_emb
),
vector_results AS (
    SELECT 
        id,
        1 - (embedding <=> query_vector.query_emb) AS vector_score
    FROM quickstart_documents, query_vector
    WHERE embedding IS NOT NULL
    ORDER BY embedding <=> query_vector.query_emb
    LIMIT 20
),
text_results AS (
    SELECT 
        id,
        ts_rank(fts_vector, plainto_tsquery('english', 'vector similarity search')) AS text_score
    FROM quickstart_documents
    WHERE fts_vector @@ plainto_tsquery('english', 'vector similarity search')
    ORDER BY text_score DESC
    LIMIT 20
)
SELECT 
    d.id,
    d.title,
    vr.vector_score,
    tr.text_score,
    (vr.vector_score + tr.text_score) / 2.0 AS avg_score
FROM quickstart_documents d
INNER JOIN vector_results vr ON d.id = vr.id
INNER JOIN text_results tr ON d.id = tr.id
ORDER BY avg_score DESC;




