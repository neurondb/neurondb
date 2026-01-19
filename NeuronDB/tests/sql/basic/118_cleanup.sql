-- ============================================================================
-- NeurondB Regression Test Suite - Cleanup
-- This file removes all test data and structures created during testing
-- ============================================================================

-- ============================================================================
-- Drop Test Tables
-- ============================================================================

DROP TABLE IF EXISTS test_vectors CASCADE;
DROP TABLE IF EXISTS test_high_dim CASCADE;
DROP TABLE IF EXISTS test_index_vectors CASCADE;
DROP TABLE IF EXISTS test_aggregates CASCADE;
DROP TABLE IF EXISTS test_distances CASCADE;
DROP TABLE IF EXISTS test_worker_data CASCADE;

-- ============================================================================
-- Clean Worker Tables (Remove Test Data Only)
-- ============================================================================

-- Remove test jobs from job queue
DELETE FROM neurondb.job_queue 
WHERE job_type IN ('vector_search', 'index_build', 'embedding_generation', 'test_job', 'vector_index')
   OR payload::text LIKE '%test%'
   OR payload::text LIKE '%Test%';

-- Remove test query metrics
DELETE FROM neurondb.query_metrics 
WHERE latency_ms < 100 OR recall_at_k > 0.8;

-- Remove test embedding cache entries
DELETE FROM neurondb.embedding_cache 
WHERE cache_key LIKE 'test_%' 
   OR cache_key LIKE '%_123'
   OR model_name = 'test_model';

-- Remove test index maintenance entries
DELETE FROM neurondb.index_maintenance 
WHERE index_name LIKE 'idx_test%'
   OR index_name LIKE 'test_%';

-- Remove test LLM jobs
DELETE FROM neurondb.llm_jobs 
WHERE operation IN ('completion', 'embedding', 'rerank')
   AND (model_name LIKE '%test%' OR input_text LIKE '%Test%')
   OR job_id < 100; -- Assuming test jobs have low IDs

-- Remove test Prometheus metrics
DELETE FROM neurondb.prometheus_metrics 
WHERE metric_name IN ('test_metric', 'queries_per_second', 'avg_latency_ms', 'total_queries', 'index_size_mb')
   OR metric_name LIKE '%test%';

-- ============================================================================
-- Clean Histograms (if any test data)
-- ============================================================================

DELETE FROM neurondb.histograms 
WHERE metric_name LIKE '%test%';

-- ============================================================================
-- Verify Cleanup
-- ============================================================================

-- Verify no test tables remain
SELECT 
    COUNT(*) AS remaining_test_tables
FROM pg_tables 
WHERE schemaname = 'public' 
    AND tablename LIKE 'test_%';

-- Verify worker tables are clean (or empty)
SELECT 
    'job_queue' AS table_name, 
    COUNT(*) AS remaining_count,
    CASE WHEN COUNT(*) = 0 THEN 'CLEAN' ELSE 'HAS_DATA' END AS status
FROM neurondb.job_queue
WHERE job_type NOT IN ('vector_search', 'index_build', 'embedding_generation', 'test_job', 'vector_index')
UNION ALL
SELECT 
    'query_metrics',
    COUNT(*),
    CASE WHEN COUNT(*) = 0 THEN 'CLEAN' ELSE 'HAS_DATA' END
FROM neurondb.query_metrics
UNION ALL
SELECT 
    'embedding_cache',
    COUNT(*),
    CASE WHEN COUNT(*) = 0 THEN 'CLEAN' ELSE 'HAS_DATA' END
FROM neurondb.embedding_cache
UNION ALL
SELECT 
    'index_maintenance',
    COUNT(*),
    CASE WHEN COUNT(*) = 0 THEN 'CLEAN' ELSE 'HAS_DATA' END
FROM neurondb.index_maintenance
UNION ALL
SELECT 
    'llm_jobs',
    COUNT(*),
    CASE WHEN COUNT(*) = 0 THEN 'CLEAN' ELSE 'HAS_DATA' END
FROM neurondb.llm_jobs
WHERE job_id >= 100
UNION ALL
SELECT 
    'prometheus_metrics',
    COUNT(*),
    CASE WHEN COUNT(*) = 0 THEN 'CLEAN' ELSE 'HAS_DATA' END
FROM neurondb.prometheus_metrics
WHERE metric_name NOT IN ('test_metric', 'queries_per_second', 'avg_latency_ms', 'total_queries', 'index_size_mb')
ORDER BY table_name;

-- ============================================================================
-- Optional: Drop Extension (Commented out - keep extension for other tests)
-- ============================================================================

-- Uncomment below if you want to completely remove the extension
-- DROP EXTENSION IF EXISTS neurondb CASCADE;

-- Final cleanup status
SELECT 'Cleanup completed successfully' AS status;

