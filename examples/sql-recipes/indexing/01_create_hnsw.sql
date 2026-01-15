-- ============================================================================
-- Recipe: Create HNSW Index
-- ============================================================================
-- Purpose: Create HNSW (Hierarchical Navigable Small World) index for fast vector search
-- 
-- Prerequisites:
--   - Table with vector column
--   - Some data in the table (indexes work better with data present)
--
-- HNSW Index:
--   - Best for: High-dimensional vectors, fast queries, good recall
--   - Parameters:
--     * m: Number of connections (default: 16, range: 4-64)
--     * ef_construction: Build quality (default: 64, range: 4-1000)
--   - Operator classes:
--     * vector_cosine_ops: For cosine distance (<=>)
--     * vector_l2_ops: For L2/Euclidean distance (<->)
--     * vector_ip_ops: For inner product (<#>)
--
-- Performance Notes:
--   - Index creation takes time proportional to data size
--   - Higher m/ef_construction = slower build, better query performance
--   - Use IF NOT EXISTS to avoid errors on re-runs
-- ============================================================================

-- Example 1: Basic HNSW index for cosine similarity
-- Most common use case for text embeddings
CREATE INDEX IF NOT EXISTS quickstart_documents_embedding_cosine_idx
ON quickstart_documents USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- Example 2: HNSW index for L2 distance
-- Use for unnormalized vectors or when L2 is preferred
CREATE INDEX IF NOT EXISTS quickstart_documents_embedding_l2_idx
ON quickstart_documents USING hnsw (embedding vector_l2_ops)
WITH (m = 16, ef_construction = 64);

-- Example 3: HNSW index for inner product
-- Use for recommendation systems or normalized vectors
CREATE INDEX IF NOT EXISTS quickstart_documents_embedding_ip_idx
ON quickstart_documents USING hnsw (embedding vector_ip_ops)
WITH (m = 16, ef_construction = 64);

-- Example 4: High-performance HNSW index
-- Higher parameters for better recall (slower build, faster queries)
CREATE INDEX IF NOT EXISTS quickstart_documents_embedding_perf_idx
ON quickstart_documents USING hnsw (embedding vector_cosine_ops)
WITH (
    m = 32,              -- More connections = better recall
    ef_construction = 200  -- Higher build quality = better accuracy
);

-- Example 5: Fast-build HNSW index
-- Lower parameters for quick index creation (faster build, may have lower recall)
CREATE INDEX IF NOT EXISTS quickstart_documents_embedding_fast_idx
ON quickstart_documents USING hnsw (embedding vector_cosine_ops)
WITH (
    m = 8,               -- Fewer connections = faster build
    ef_construction = 32   -- Lower quality = faster build
);

-- Example 6: Check index creation progress
-- Monitor index build progress (PostgreSQL 12+)
SELECT 
    schemaname,
    tablename,
    indexname,
    indexdef
FROM pg_indexes
WHERE indexname LIKE '%embedding%'
ORDER BY tablename, indexname;

-- Example 7: Verify index is being used
-- Check that queries use the index
EXPLAIN (ANALYZE, BUFFERS)
WITH query_vector AS (
    SELECT embed_text('test query') AS query_emb
)
SELECT id, title
FROM quickstart_documents, query_vector
WHERE embedding IS NOT NULL
ORDER BY embedding <=> query_vector.query_emb
LIMIT 10;

-- Example 8: Concurrent index creation
-- Create index without blocking writes (use with caution)
-- Note: This is slower but doesn't lock the table
-- CREATE INDEX CONCURRENTLY quickstart_documents_embedding_concurrent_idx
-- ON quickstart_documents USING hnsw (embedding vector_cosine_ops)
-- WITH (m = 16, ef_construction = 64);




