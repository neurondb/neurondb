-- ============================================================================
-- SQL Recipe Library: Indexing
-- ============================================================================
-- Ready-to-run queries for creating vector indexes
-- 
-- Prerequisites:
--   - NeuronDB extension installed
--   - Table with vector column (or use quickstart_documents)
--
-- Usage:
--   psql -f 04_indexing.sql
--   Or copy individual recipes to run them
--
-- ============================================================================

-- ============================================================================
-- Recipe 1: Create Basic HNSW Index (L2 Distance)
-- ============================================================================
-- Use case: Fast approximate nearest neighbor search using L2/Euclidean distance
-- Complexity: ⭐
-- Index type: HNSW (Hierarchical Navigable Small World)

-- Basic HNSW index with default parameters
CREATE INDEX IF NOT EXISTS idx_quickstart_hnsw_l2
ON quickstart_documents
USING hnsw (embedding vector_l2_ops);

-- ============================================================================
-- Recipe 2: Create HNSW Index with Custom Parameters
-- ============================================================================
-- Use case: Tune index for your specific workload (speed vs. recall)
-- Complexity: ⭐⭐
-- Parameters:
--   - m: Number of connections per layer (default: 16, range: 4-64)
--   - ef_construction: Search width during index build (default: 200, range: 4-1000)

-- HNSW index with custom parameters (higher quality, slower build)
CREATE INDEX IF NOT EXISTS idx_quickstart_hnsw_l2_tuned
ON quickstart_documents
USING hnsw (embedding vector_l2_ops)
WITH (m = 16, ef_construction = 200);

-- ============================================================================
-- Recipe 3: Create HNSW Index for Cosine Distance
-- ============================================================================
-- Use case: Similarity search using cosine distance (common for normalized embeddings)
-- Complexity: ⭐

-- HNSW index for cosine similarity search
CREATE INDEX IF NOT EXISTS idx_quickstart_hnsw_cosine
ON quickstart_documents
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- ============================================================================
-- Recipe 4: Create HNSW Index for Inner Product
-- ============================================================================
-- Use case: Maximum inner product search (for normalized vectors)
-- Complexity: ⭐

-- HNSW index for inner product (negative dot product)
CREATE INDEX IF NOT EXISTS idx_quickstart_hnsw_ip
ON quickstart_documents
USING hnsw (embedding vector_ip_ops)
WITH (m = 16, ef_construction = 64);

-- ============================================================================
-- Recipe 5: Create IVF Index
-- ============================================================================
-- Use case: Alternative to HNSW, better for memory-constrained environments
-- Complexity: ⭐⭐
-- Index type: IVF (Inverted File Index)

-- IVF index with default parameters
CREATE INDEX IF NOT EXISTS idx_quickstart_ivf_l2
ON quickstart_documents
USING ivf (embedding vector_l2_ops);

-- IVF index with custom number of lists
-- Rule of thumb: lists = sqrt(rows) for good performance
CREATE INDEX IF NOT EXISTS idx_quickstart_ivf_l2_tuned
ON quickstart_documents
USING ivf (embedding vector_l2_ops)
WITH (lists = 100);

-- ============================================================================
-- Recipe 6: Drop and Recreate Index
-- ============================================================================
-- Use case: Rebuild index with different parameters or after data changes
-- Complexity: ⭐

-- Drop existing index
DROP INDEX IF EXISTS idx_quickstart_hnsw_l2;

-- Recreate with new parameters
CREATE INDEX idx_quickstart_hnsw_l2_new
ON quickstart_documents
USING hnsw (embedding vector_l2_ops)
WITH (m = 32, ef_construction = 400);

-- ============================================================================
-- Recipe 7: Create Multiple Indexes (Different Distance Metrics)
-- ============================================================================
-- Use case: Support different distance metrics on the same column
-- Complexity: ⭐⭐
-- Note: Multiple indexes can exist on the same column for different operators

-- Create indexes for different distance metrics
CREATE INDEX IF NOT EXISTS idx_quickstart_hnsw_l2
ON quickstart_documents
USING hnsw (embedding vector_l2_ops);

CREATE INDEX IF NOT EXISTS idx_quickstart_hnsw_cosine
ON quickstart_documents
USING hnsw (embedding vector_cosine_ops);

-- Now you can use either <-> (L2) or <=> (cosine) operators efficiently
-- PostgreSQL will choose the appropriate index based on the operator used

-- ============================================================================
-- Recipe 8: Create Index on New Table
-- ============================================================================
-- Use case: Create index when setting up a new table
-- Complexity: ⭐

-- Example: Create a new table with index
CREATE TABLE IF NOT EXISTS my_documents (
    id SERIAL PRIMARY KEY,
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    embedding vector(384)
);

-- Insert some data first (indexes are built on existing data)
-- INSERT INTO my_documents (title, content, embedding) VALUES (...);

-- Create index after inserting data
CREATE INDEX idx_my_documents_hnsw
ON my_documents
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- ============================================================================
-- Recipe 9: List All Vector Indexes
-- ============================================================================
-- Use case: See what indexes exist on your tables
-- Complexity: ⭐

-- List all HNSW indexes
SELECT 
    schemaname,
    tablename,
    indexname,
    indexdef
FROM pg_indexes
WHERE indexdef LIKE '%hnsw%'
ORDER BY schemaname, tablename, indexname;

-- List all vector indexes (HNSW and IVF)
SELECT 
    schemaname,
    tablename,
    indexname,
    indexdef
FROM pg_indexes
WHERE indexdef LIKE '%hnsw%' OR indexdef LIKE '%ivf%'
ORDER BY schemaname, tablename, indexname;

-- List indexes on a specific table
SELECT 
    indexname,
    indexdef
FROM pg_indexes
WHERE tablename = 'quickstart_documents'
ORDER BY indexname;

-- ============================================================================
-- Recipe 10: Check Index Size and Statistics
-- ============================================================================
-- Use case: Monitor index size and usage
-- Complexity: ⭐⭐

-- Get index sizes
SELECT 
    schemaname,
    tablename,
    indexname,
    pg_size_pretty(pg_relation_size(indexrelid)) AS index_size
FROM pg_stat_user_indexes
WHERE indexrelname LIKE '%hnsw%' OR indexrelname LIKE '%ivf%'
ORDER BY pg_relation_size(indexrelid) DESC;

-- Get index statistics for quickstart_documents
SELECT 
    indexrelname AS index_name,
    idx_scan AS scans,
    idx_tup_read AS tuples_read,
    idx_tup_fetch AS tuples_fetched,
    pg_size_pretty(pg_relation_size(indexrelid)) AS size
FROM pg_stat_user_indexes
WHERE relname = 'quickstart_documents';

-- ============================================================================
-- Recipe 11: Tune Index Build Performance
-- ============================================================================
-- Use case: Optimize index creation time and memory usage
-- Complexity: ⭐⭐⭐

-- Increase memory for index build (for large datasets)
SET maintenance_work_mem = '2GB';  -- Adjust based on available RAM

-- Create index with more memory available
CREATE INDEX IF NOT EXISTS idx_quickstart_hnsw_large
ON quickstart_documents
USING hnsw (embedding vector_l2_ops)
WITH (m = 16, ef_construction = 200);

-- Reset memory setting after index creation
RESET maintenance_work_mem;

-- ============================================================================
-- Recipe 12: Verify Index Usage
-- ============================================================================
-- Use case: Confirm PostgreSQL is using your index for queries
-- Complexity: ⭐⭐

-- Use EXPLAIN ANALYZE to verify index usage
EXPLAIN ANALYZE
SELECT id, title
FROM quickstart_documents
ORDER BY embedding <=> (SELECT embedding FROM quickstart_documents WHERE id = 1)
LIMIT 10;

-- Expected output should show "Index Scan using idx_quickstart_hnsw_cosine"
-- If it shows "Seq Scan", the index may not be used (check enable_seqscan setting)

-- ============================================================================
-- Recipe 13: Tune Search Performance (Runtime Parameters)
-- ============================================================================
-- Use case: Adjust search quality vs. speed at query time
-- Complexity: ⭐⭐

-- Increase search quality for HNSW (higher ef_search = better recall, slower)
SET hnsw.ef_search = 100;  -- Default is typically 40

-- Run query with higher quality
SELECT id, title
FROM quickstart_documents
ORDER BY embedding <=> (SELECT embedding FROM quickstart_documents WHERE id = 1)
LIMIT 10;

-- Reset to default
RESET hnsw.ef_search;

-- For IVF indexes, use probes parameter
SET ivfflat.probes = 20;  -- Default is typically 1
-- Run query...
RESET ivfflat.probes;

-- ============================================================================
-- Recipe 14: Index Maintenance (REINDEX)
-- ============================================================================
-- Use case: Rebuild index after heavy updates or to reclaim space
-- Complexity: ⭐

-- Rebuild a specific index
REINDEX INDEX idx_quickstart_hnsw_cosine;

-- Rebuild all indexes on a table
REINDEX TABLE quickstart_documents;

-- ============================================================================
-- Recipe 15: Index Recommendations by Use Case
-- ============================================================================
-- Use case: Choose the right index type and parameters
-- Complexity: ⭐⭐⭐

-- Small dataset (< 100K vectors): HNSW with default parameters
-- CREATE INDEX ON documents USING hnsw (embedding vector_cosine_ops);

-- Medium dataset (100K - 1M vectors): HNSW with tuned parameters
-- CREATE INDEX ON documents USING hnsw (embedding vector_cosine_ops)
-- WITH (m = 16, ef_construction = 200);

-- Large dataset (> 1M vectors): Consider IVF or higher m/ef_construction
-- CREATE INDEX ON documents USING ivf (embedding vector_cosine_ops)
-- WITH (lists = sqrt(count(*)));

-- Memory-constrained: Use IVF or lower HNSW m parameter
-- CREATE INDEX ON documents USING ivf (embedding vector_cosine_ops)
-- WITH (lists = 100);

-- High recall requirement: Increase ef_construction and ef_search
-- CREATE INDEX ON documents USING hnsw (embedding vector_cosine_ops)
-- WITH (m = 32, ef_construction = 400);
-- SET hnsw.ef_search = 200;

-- Fast queries (lower recall acceptable): Lower ef_search
-- SET hnsw.ef_search = 20;  -- Faster, lower recall

-- ============================================================================
-- Notes:
-- ============================================================================
-- 
-- 1. HNSW vs IVF:
--    - HNSW: Better query performance, higher memory usage, slower build
--    - IVF: Lower memory usage, faster build, good for very large datasets
--
-- 2. Distance metrics:
--    - vector_l2_ops: For L2/Euclidean distance (<-> operator)
--    - vector_cosine_ops: For cosine distance (<=> operator, common for embeddings)
--    - vector_ip_ops: For inner product (<#> operator, for normalized vectors)
--
-- 3. HNSW parameters:
--    - m: Higher = better recall, more memory, slower build (4-64, default 16)
--    - ef_construction: Higher = better index quality, slower build (4-1000, default 200)
--
-- 4. Search parameters:
--    - hnsw.ef_search: Higher = better recall, slower queries (default ~40)
--    - ivfflat.probes: Higher = better recall, slower queries (default 1)
--
-- 5. Build performance:
--    - Increase maintenance_work_mem for faster builds
--    - Indexes are built on existing data (insert data first)
--
-- 6. Best practices:
--    - Use cosine_ops for most embedding models (they're usually normalized)
--    - Start with defaults, tune based on your query performance
--    - Monitor index size and query performance
--    - Reindex periodically after heavy updates
--
-- ============================================================================




