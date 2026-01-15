-- ============================================================================
-- Recipe: Performance Tuning for Vector Search
-- ============================================================================
-- Purpose: Optimize vector search queries for speed and accuracy
-- 
-- Prerequisites:
--   - quickstart_documents table
--   - HNSW index on embedding column (recommended)
--
-- Optimization Techniques:
--   1. Use appropriate index (HNSW for high-dimensional, fast queries)
--   2. Set ef_search parameter for recall/speed tradeoff
--   3. Limit result sets appropriately
--   4. Use covering indexes
--   5. Parallel query execution
--
-- Performance Notes:
--   - ef_search: Higher = better recall, slower (default: 40, range: 40-200)
--   - Index creation takes time but enables fast queries
-- ============================================================================

-- Example 1: Check index usage
-- Verify that HNSW index is being used for queries
EXPLAIN ANALYZE
WITH query_vector AS (
    SELECT embed_text('database optimization') AS query_emb
)
SELECT id, title
FROM quickstart_documents, query_vector
WHERE embedding IS NOT NULL
ORDER BY embedding <=> query_vector.query_emb
LIMIT 10;

-- Example 2: Adjust ef_search for better recall
-- Higher ef_search = better accuracy, slower queries
-- Use this session setting for the query
SET hnsw.ef_search = 100;  -- Default is 40, max is typically 200

WITH query_vector AS (
    SELECT embed_text('vector databases') AS query_emb
)
SELECT 
    id,
    title,
    1 - (embedding <=> query_vector.query_emb) AS similarity
FROM quickstart_documents, query_vector
WHERE embedding IS NOT NULL
ORDER BY embedding <=> query_vector.query_emb
LIMIT 10;

-- Reset to default
RESET hnsw.ef_search;

-- Example 3: Performance comparison with different ef_search values
-- Compare query time vs recall
\timing on

-- Fast search (lower recall)
SET hnsw.ef_search = 40;
WITH query_vector AS (
    SELECT embed_text('machine learning') AS query_emb
)
SELECT COUNT(*) FROM (
    SELECT id FROM quickstart_documents, query_vector
    ORDER BY embedding <=> query_vector.query_emb
    LIMIT 10
) sub;

-- Balanced search
SET hnsw.ef_search = 100;
WITH query_vector AS (
    SELECT embed_text('machine learning') AS query_emb
)
SELECT COUNT(*) FROM (
    SELECT id FROM quickstart_documents, query_vector
    ORDER BY embedding <=> query_vector.query_emb
    LIMIT 10
) sub;

-- High recall search
SET hnsw.ef_search = 200;
WITH query_vector AS (
    SELECT embed_text('machine learning') AS query_emb
)
SELECT COUNT(*) FROM (
    SELECT id FROM quickstart_documents, query_vector
    ORDER BY embedding <=> query_vector.query_emb
    LIMIT 10
) sub;

RESET hnsw.ef_search;
\timing off

-- Example 4: Create optimized HNSW index
-- Index should already exist from quickstart, but here's how to create optimally
-- Note: This will take time if index doesn't exist
-- Uncomment to run:
/*
CREATE INDEX IF NOT EXISTS quickstart_documents_embedding_hnsw_idx
ON quickstart_documents USING hnsw (embedding vector_cosine_ops)
WITH (
    m = 16,              -- Number of connections (higher = better recall, slower build)
    ef_construction = 64  -- Index build quality (higher = better quality, slower build)
);
*/

-- Example 5: Monitor index usage and performance
-- Check index statistics
SELECT 
    schemaname,
    tablename,
    indexname,
    idx_scan AS index_scans,
    idx_tup_read AS tuples_read,
    idx_tup_fetch AS tuples_fetched
FROM pg_stat_user_indexes
WHERE indexname LIKE '%embedding%'
ORDER BY idx_scan DESC;

-- Example 6: Parallel query execution
-- Enable parallel workers for large result sets
SET max_parallel_workers_per_gather = 4;

WITH query_vector AS (
    SELECT embed_text('vector similarity') AS query_emb
)
SELECT 
    id,
    title,
    1 - (embedding <=> query_vector.query_emb) AS similarity
FROM quickstart_documents, query_vector
WHERE embedding IS NOT NULL
ORDER BY embedding <=> query_vector.query_emb
LIMIT 100;

RESET max_parallel_workers_per_gather;

-- Example 7: Covering index for better performance
-- Include frequently accessed columns in index (PostgreSQL 11+)
-- This reduces table lookups
/*
CREATE INDEX IF NOT EXISTS quickstart_documents_covering_idx
ON quickstart_documents USING hnsw (embedding vector_cosine_ops)
INCLUDE (id, title);  -- Covering columns
*/




