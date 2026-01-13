-- ============================================================================
-- Recipe: Create IVF Index
-- ============================================================================
-- Purpose: Create IVF (Inverted File) index for vector search
-- 
-- Prerequisites:
--   - Table with vector column
--   - Enough data for training (typically > 1000 vectors)
--
-- IVF Index:
--   - Best for: Large datasets, memory-efficient, approximate search
--   - Parameters:
--     * lists: Number of clusters (default: 100, should be sqrt(data_size))
--   - Requires training: Must have data to train clusters
--
-- Performance Notes:
--   - Training requires scanning data
--   - More lists = better accuracy but slower queries
--   - Good for datasets > 1M vectors
-- ============================================================================

-- Example 1: Basic IVF index
-- Standard configuration for medium datasets
CREATE INDEX IF NOT EXISTS quickstart_documents_embedding_ivf_idx
ON quickstart_documents USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

-- Example 2: IVF index with custom lists
-- Adjust lists based on data size (typically sqrt(number_of_rows))
-- For ~10K rows: lists = 100
-- For ~100K rows: lists = 316
-- For ~1M rows: lists = 1000
CREATE INDEX IF NOT EXISTS quickstart_documents_embedding_ivf_custom_idx
ON quickstart_documents USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 50);  -- Fewer lists for smaller datasets

-- Example 3: IVF index for L2 distance
CREATE INDEX IF NOT EXISTS quickstart_documents_embedding_ivf_l2_idx
ON quickstart_documents USING ivfflat (embedding vector_l2_ops)
WITH (lists = 100);

-- Example 4: IVF index for inner product
CREATE INDEX IF NOT EXISTS quickstart_documents_embedding_ivf_ip_idx
ON quickstart_documents USING ivfflat (embedding vector_ip_ops)
WITH (lists = 100);

-- Example 5: Check IVF index status
-- Verify index exists and check parameters
SELECT 
    schemaname,
    tablename,
    indexname,
    indexdef
FROM pg_indexes
WHERE indexname LIKE '%ivf%'
ORDER BY tablename, indexname;

-- Example 6: Compare IVF vs HNSW
-- For small datasets (<100K), HNSW is usually faster
-- For large datasets (>1M), IVF may be more memory efficient
-- Create both and test which works better for your use case

-- HNSW (better for queries < 100K vectors)
-- CREATE INDEX IF NOT EXISTS idx_hnsw ON table USING hnsw (embedding vector_cosine_ops);

-- IVF (better for queries > 1M vectors, memory constrained)
-- CREATE INDEX IF NOT EXISTS idx_ivf ON table USING ivfflat (embedding vector_cosine_ops);

-- Example 7: Rebuilding IVF index after significant data changes
-- IVF indexes may need rebuilding if data distribution changes significantly
DROP INDEX IF EXISTS quickstart_documents_embedding_ivf_idx;
CREATE INDEX quickstart_documents_embedding_ivf_idx
ON quickstart_documents USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);



