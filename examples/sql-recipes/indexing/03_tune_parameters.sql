-- ============================================================================
-- Recipe: Tune Index Parameters
-- ============================================================================
-- Purpose: Optimize index parameters for your specific use case
-- 
-- Prerequisites:
--   - Existing index or data to create index on
--   - Understanding of recall vs speed tradeoffs
--
-- Tuning Strategy:
--   1. Start with defaults
--   2. Measure query performance
--   3. Adjust parameters based on needs
--   4. Balance recall, speed, and index size
--
-- Performance Notes:
--   - Higher parameters = better recall, slower queries, larger index
--   - Lower parameters = faster queries, may miss some results
--   - Test with representative query workload
-- ============================================================================

-- Example 1: Tune HNSW m parameter (number of connections)
-- Higher m = better recall, slower queries, larger index
-- Typical range: 8-64, default: 16

-- Low m (fast queries, may miss results)
DROP INDEX IF EXISTS quickstart_documents_embedding_tune_low;
CREATE INDEX quickstart_documents_embedding_tune_low
ON quickstart_documents USING hnsw (embedding vector_cosine_ops)
WITH (m = 8, ef_construction = 32);

-- Medium m (balanced)
DROP INDEX IF EXISTS quickstart_documents_embedding_tune_med;
CREATE INDEX quickstart_documents_embedding_tune_med
ON quickstart_documents USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- High m (best recall, slower)
DROP INDEX IF EXISTS quickstart_documents_embedding_tune_high;
CREATE INDEX quickstart_documents_embedding_tune_high
ON quickstart_documents USING hnsw (embedding vector_cosine_ops)
WITH (m = 32, ef_construction = 128);

-- Example 2: Tune ef_construction (index build quality)
-- Higher ef_construction = better index quality, slower build
-- Typical range: 32-200, default: 64

-- Fast build (lower quality)
-- CREATE INDEX ... WITH (m = 16, ef_construction = 32);

-- Balanced build
-- CREATE INDEX ... WITH (m = 16, ef_construction = 64);

-- High quality build (slower)
-- CREATE INDEX ... WITH (m = 16, ef_construction = 128);

-- Example 3: Tune ef_search (query-time parameter)
-- This is a session/query setting, not an index parameter
-- Higher ef_search = better recall, slower queries
-- Typical range: 40-200, default: 40

-- Fast queries (lower recall)
SET hnsw.ef_search = 40;
-- Run your queries here
RESET hnsw.ef_search;

-- Balanced queries
SET hnsw.ef_search = 100;
-- Run your queries here
RESET hnsw.ef_search;

-- High recall queries (slower)
SET hnsw.ef_search = 200;
-- Run your queries here
RESET hnsw.ef_search;

-- Example 4: Benchmark different parameter combinations
-- Test query performance with different settings
\timing on

-- Test 1: Low ef_search
SET hnsw.ef_search = 40;
WITH query_vector AS (
    SELECT embed_text('test query') AS query_emb
)
SELECT COUNT(*) FROM (
    SELECT id FROM quickstart_documents, query_vector
    ORDER BY embedding <=> query_vector.query_emb
    LIMIT 10
) sub;

-- Test 2: Medium ef_search
SET hnsw.ef_search = 100;
WITH query_vector AS (
    SELECT embed_text('test query') AS query_emb
)
SELECT COUNT(*) FROM (
    SELECT id FROM quickstart_documents, query_vector
    ORDER BY embedding <=> query_vector.query_emb
    LIMIT 10
) sub;

-- Test 3: High ef_search
SET hnsw.ef_search = 200;
WITH query_vector AS (
    SELECT embed_text('test query') AS query_emb
)
SELECT COUNT(*) FROM (
    SELECT id FROM quickstart_documents, query_vector
    ORDER BY embedding <=> query_vector.query_emb
    LIMIT 10
) sub;

RESET hnsw.ef_search;
\timing off

-- Example 5: Tune IVF lists parameter
-- For IVF indexes, lists should be approximately sqrt(number_of_rows)
-- Too few lists = poor accuracy
-- Too many lists = slower queries

-- Small dataset (~10K rows)
-- CREATE INDEX ... USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- Medium dataset (~100K rows)
-- CREATE INDEX ... USING ivfflat (embedding vector_cosine_ops) WITH (lists = 316);

-- Large dataset (~1M rows)
-- CREATE INDEX ... USING ivfflat (embedding vector_cosine_ops) WITH (lists = 1000);

-- Example 6: Check index size
-- Monitor index size to understand space tradeoffs
SELECT 
    schemaname,
    tablename,
    indexname,
    pg_size_pretty(pg_relation_size(indexname::regclass)) AS index_size
FROM pg_indexes
WHERE indexname LIKE '%embedding%'
ORDER BY pg_relation_size(indexname::regclass) DESC;

-- Example 7: Compare index performance
-- Create multiple indexes and compare
-- Note: Only keep the best one in production

-- Index A: Conservative parameters
-- CREATE INDEX idx_a ... WITH (m = 16, ef_construction = 64);

-- Index B: Aggressive parameters
-- CREATE INDEX idx_b ... WITH (m = 32, ef_construction = 128);

-- Test both with your workload and compare:
-- - Query time
-- - Recall (do you get expected results?)
-- - Index size
-- - Build time



