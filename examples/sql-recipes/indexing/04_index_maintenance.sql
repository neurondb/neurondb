-- ============================================================================
-- Recipe: Index Maintenance
-- ============================================================================
-- Purpose: Monitor, maintain, and optimize vector indexes
-- 
-- Prerequisites:
--   - Existing indexes on vector columns
--
-- Maintenance Tasks:
--   1. Monitor index usage
--   2. Check index size
--   3. Rebuild indexes when needed
--   4. Analyze tables for query planning
--
-- Performance Notes:
--   - Indexes may need rebuilding after large data changes
--   - Regular VACUUM helps maintain index efficiency
--   - Monitor index bloat
-- ============================================================================

-- Example 1: Check index usage statistics
-- See how often indexes are used
SELECT 
    schemaname,
    tablename,
    indexname,
    idx_scan AS index_scans,
    idx_tup_read AS tuples_read,
    idx_tup_fetch AS tuples_fetched,
    CASE 
        WHEN idx_scan = 0 THEN 'Unused'
        WHEN idx_scan < 100 THEN 'Rarely used'
        WHEN idx_scan < 1000 THEN 'Moderately used'
        ELSE 'Heavily used'
    END AS usage_status
FROM pg_stat_user_indexes
WHERE indexname LIKE '%embedding%' OR indexname LIKE '%vector%'
ORDER BY idx_scan DESC;

-- Example 2: Check index sizes
-- Monitor index storage usage
SELECT 
    schemaname,
    tablename,
    indexname,
    pg_size_pretty(pg_relation_size(indexname::regclass)) AS index_size,
    pg_size_pretty(pg_relation_size((schemaname||'.'||tablename)::regclass)) AS table_size,
    ROUND(
        100.0 * pg_relation_size(indexname::regclass) / 
        NULLIF(pg_relation_size((schemaname||'.'||tablename)::regclass), 0),
        2
    ) AS index_to_table_ratio
FROM pg_indexes
WHERE indexname LIKE '%embedding%' OR indexname LIKE '%vector%'
ORDER BY pg_relation_size(indexname::regclass) DESC;

-- Example 3: Analyze tables for better query planning
-- Update statistics for query planner
ANALYZE quickstart_documents;

-- Check table statistics
SELECT 
    schemaname,
    tablename,
    n_live_tup AS row_count,
    last_vacuum,
    last_autovacuum,
    last_analyze,
    last_autoanalyze
FROM pg_stat_user_tables
WHERE tablename = 'quickstart_documents';

-- Example 4: Rebuild index after significant data changes
-- Drop and recreate to optimize index structure
-- Note: This locks the table during creation

-- Check current index
SELECT indexname, indexdef 
FROM pg_indexes 
WHERE tablename = 'quickstart_documents' 
  AND indexname LIKE '%embedding%';

-- Rebuild index
DROP INDEX IF EXISTS quickstart_documents_embedding_idx;
CREATE INDEX quickstart_documents_embedding_idx
ON quickstart_documents USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- Example 5: Concurrent index rebuild (non-blocking)
-- Rebuild without locking table (PostgreSQL 12+)
-- Note: Slower but doesn't block writes

-- CREATE INDEX CONCURRENTLY quickstart_documents_embedding_new_idx
-- ON quickstart_documents USING hnsw (embedding vector_cosine_ops)
-- WITH (m = 16, ef_construction = 64);
-- 
-- -- After new index is ready:
-- DROP INDEX CONCURRENTLY quickstart_documents_embedding_old_idx;
-- ALTER INDEX quickstart_documents_embedding_new_idx 
-- RENAME TO quickstart_documents_embedding_idx;

-- Example 6: Check for index bloat
-- Unused space in indexes affects performance
SELECT 
    schemaname,
    tablename,
    indexname,
    pg_size_pretty(pg_relation_size(indexname::regclass)) AS index_size,
    pg_size_pretty(
        pg_relation_size(indexname::regclass) - 
        pg_stat_get_index_size(pg_indexes.indexrelid)
    ) AS estimated_bloat
FROM pg_indexes
WHERE indexname LIKE '%embedding%'
ORDER BY pg_relation_size(indexname::regclass) DESC;

-- Example 7: Vacuum to reclaim space
-- Helps maintain index efficiency
VACUUM ANALYZE quickstart_documents;

-- Example 8: Monitor index build progress (PostgreSQL 12+)
-- Check if index is currently being built
SELECT 
    pid,
    datname,
    usename,
    query,
    state,
    wait_event_type,
    wait_event
FROM pg_stat_activity
WHERE query LIKE '%CREATE INDEX%' 
   OR query LIKE '%CREATE UNIQUE INDEX%';

-- Example 9: List all vector-related indexes
-- Get comprehensive view of vector indexes
SELECT 
    i.indexname,
    i.tablename,
    i.indexdef,
    pg_size_pretty(pg_relation_size(i.indexname::regclass)) AS size,
    s.idx_scan AS usage_count
FROM pg_indexes i
LEFT JOIN pg_stat_user_indexes s 
    ON i.indexname = s.indexname
WHERE i.indexdef LIKE '%hnsw%' 
   OR i.indexdef LIKE '%ivfflat%'
   OR i.indexdef LIKE '%vector%'
ORDER BY i.tablename, i.indexname;

-- Example 10: Drop unused indexes
-- Remove indexes that aren't being used
-- WARNING: Verify indexes are truly unused before dropping

-- Find unused indexes
SELECT 
    schemaname,
    tablename,
    indexname,
    idx_scan
FROM pg_stat_user_indexes
WHERE (indexname LIKE '%embedding%' OR indexname LIKE '%vector%')
  AND idx_scan = 0
  AND schemaname = 'public';

-- Drop if confirmed unused:
-- DROP INDEX IF EXISTS unused_index_name;



