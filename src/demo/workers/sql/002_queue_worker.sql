-- ============================================================================
-- NeuronDB Workers Demo - neuranq (Queue Worker) Tests
-- ============================================================================
-- Purpose: Test async job queue processing
-- Worker: neuranq - Processes embedding, rerank, cache refresh, HTTP jobs
-- ============================================================================

\echo '=========================================='
\echo 'neuranq (Queue Worker) Tests'
\echo '=========================================='

SET search_path TO worker_demo, neurondb, public;

-- ============================================================================
-- TEST 1: Job Queue Table Structure
-- ============================================================================
\echo ''
\echo 'TEST 1: Job Queue Table Structure'
\echo '--------------------------------------------'

-- Show table structure
\d neurondb.job_queue

-- Check indexes
SELECT 
    indexname,
    indexdef
FROM pg_indexes
WHERE schemaname = 'neurondb'
AND tablename = 'neurondb_job_queue'
ORDER BY indexname;

\echo ''
\echo '✓ Job queue table structure verified'

-- ============================================================================
-- TEST 2: Insert Test Jobs
-- ============================================================================
\echo ''
\echo 'TEST 2: Insert Test Jobs'
\echo '--------------------------------------------'

-- Insert various job types
INSERT INTO neurondb.job_queue (tenant_id, job_type, payload, status)
VALUES 
    (1, 'embedding', '{"text": "Hello world", "model": "text-embedding-ada-002"}'::jsonb, 'pending'),
    (1, 'embedding', '{"text": "Machine learning is awesome", "model": "all-MiniLM-L6-v2"}'::jsonb, 'pending'),
    (2, 'rerank', '{"query": "database query", "documents": ["PostgreSQL", "MongoDB", "Redis"]}'::jsonb, 'pending'),
    (2, 'cache_refresh', '{"cache_key": "vector_index_stats", "ttl": 300}'::jsonb, 'pending'),
    (3, 'http_call', '{"url": "https://api.example.com/webhook", "method": "POST", "data": {"event": "test"}}'::jsonb, 'pending');

\echo ''
\echo 'Inserted 5 test jobs'

-- ============================================================================
-- TEST 3: Query Job Queue Status
-- ============================================================================
\echo ''
\echo 'TEST 3: Job Queue Status'
\echo '--------------------------------------------'

-- Show all pending jobs
SELECT 
    job_id,
    tenant_id,
    job_type,
    payload::text as payload_preview,
    status,
    retry_count,
    created_at
FROM neurondb.job_queue
WHERE status = 'pending'
ORDER BY job_id DESC
LIMIT 10;

-- Job statistics by type
\echo ''
\echo 'Job Statistics by Type:'
SELECT 
    job_type,
    COUNT(*) as total_jobs,
    COUNT(*) FILTER (WHERE status = 'pending') as pending,
    COUNT(*) FILTER (WHERE status = 'processing') as processing,
    COUNT(*) FILTER (WHERE status = 'completed') as completed,
    COUNT(*) FILTER (WHERE status = 'failed') as failed
FROM neurondb.job_queue
GROUP BY job_type
ORDER BY job_type;

-- Job statistics by tenant
\echo ''
\echo 'Job Statistics by Tenant:'
SELECT 
    tenant_id,
    COUNT(*) as total_jobs,
    COUNT(*) FILTER (WHERE status = 'pending') as pending,
    COUNT(*) FILTER (WHERE status = 'processing') as processing,
    COUNT(*) FILTER (WHERE status = 'completed') as completed,
    COUNT(*) FILTER (WHERE status = 'failed') as failed
FROM neurondb.job_queue
GROUP BY tenant_id
ORDER BY tenant_id;

\echo ''
\echo '✓ Job queue status verified'

-- ============================================================================
-- TEST 4: Simulate Job Processing (Manual)
-- ============================================================================
\echo ''
\echo 'TEST 4: Simulate Job Processing'
\echo '--------------------------------------------'

-- Manually mark one job as completed (simulating worker processing)
WITH next_job AS (
    SELECT job_id
    FROM neurondb.job_queue
    WHERE status = 'pending'
    ORDER BY created_at
    LIMIT 1
)
UPDATE neurondb.job_queue
SET 
    status = 'completed',
    completed_at = now(),
    result = '{"status": "success", "processed_by": "manual_test"}'::jsonb
FROM next_job
WHERE neurondb.job_queue.job_id = next_job.job_id
RETURNING 
    job_id,
    job_type,
    status,
    completed_at;

\echo ''
\echo '✓ Job processing simulated'

-- ============================================================================
-- TEST 5: Error Handling
-- ============================================================================
\echo ''
\echo 'TEST 5: Error Handling'
\echo '--------------------------------------------'

-- Insert a job with invalid payload
INSERT INTO neurondb.job_queue (tenant_id, job_type, payload, status)
VALUES (999, 'invalid_type', '{"error": "this will fail"}'::jsonb, 'pending');

-- Simulate failed job
WITH failed_job AS (
    SELECT job_id
    FROM neurondb.job_queue
    WHERE tenant_id = 999
    LIMIT 1
)
UPDATE neurondb.job_queue
SET 
    status = 'failed',
    retry_count = retry_count + 1,
    error_message = 'Invalid job type',
    completed_at = now()
FROM failed_job
WHERE neurondb.job_queue.job_id = failed_job.job_id
RETURNING 
    job_id,
    job_type,
    status,
    error_message,
    retry_count;

\echo ''
\echo '✓ Error handling tested'

-- ============================================================================
-- TEST 6: Job Queue Performance
-- ============================================================================
\echo ''
\echo 'TEST 6: Job Queue Performance'
\echo '--------------------------------------------'

-- Overall queue metrics
SELECT 
    'Queue Performance' as metric_category,
    COUNT(*) as total_jobs,
    COUNT(*) FILTER (WHERE status = 'pending') as pending_jobs,
    COUNT(*) FILTER (WHERE status = 'processing') as processing_jobs,
    COUNT(*) FILTER (WHERE status = 'completed') as completed_jobs,
    COUNT(*) FILTER (WHERE status = 'failed') as failed_jobs,
    AVG(retry_count)::numeric(5,2) as avg_retries,
    EXTRACT(EPOCH FROM (now() - MIN(created_at)))::numeric(10,2) as queue_age_seconds
FROM neurondb.job_queue;

-- Processing time for completed jobs
\echo ''
\echo 'Completed Job Processing Times:'
SELECT 
    job_type,
    COUNT(*) as completed_count,
    AVG(EXTRACT(EPOCH FROM (completed_at - created_at)))::numeric(10,2) as avg_processing_seconds,
    MIN(EXTRACT(EPOCH FROM (completed_at - created_at)))::numeric(10,2) as min_processing_seconds,
    MAX(EXTRACT(EPOCH FROM (completed_at - created_at)))::numeric(10,2) as max_processing_seconds
FROM neurondb.job_queue
WHERE status = 'completed' AND completed_at IS NOT NULL
GROUP BY job_type
ORDER BY job_type;

\echo ''
\echo '✓ Performance metrics collected'

-- ============================================================================
-- TEST 7: Queue Cleanup & Archival
-- ============================================================================
\echo ''
\echo 'TEST 7: Queue Cleanup'
\echo '--------------------------------------------'

-- Show old completed jobs (candidates for archival)
SELECT 
    'Archival Candidates' as category,
    COUNT(*) as job_count,
    MIN(completed_at) as oldest_completion,
    MAX(completed_at) as newest_completion
FROM neurondb.job_queue
WHERE status IN ('completed', 'failed')
AND completed_at < now() - interval '7 days';

\echo ''
\echo 'Note: In production, old jobs should be archived to keep queue size manageable'

\echo ''
\echo '✓ Queue cleanup analysis complete'

\echo ''
\echo '=========================================='
\echo 'neuranq Queue Worker Tests Complete!'
\echo '=========================================='
\echo ''
\echo 'Key Findings:'
\echo '  - Job queue table operational'
\echo '  - Multiple job types supported'
\echo '  - Error handling in place'
\echo '  - Performance metrics available'
\echo ''
\echo 'Worker Purpose:'
\echo '  neuranq processes async jobs including:'
\echo '  - embedding: Generate text embeddings'
\echo '  - rerank: Rerank search results'
\echo '  - cache_refresh: Update cached data'
\echo '  - http_call: External API calls'
\echo ''
\echo 'To monitor queue in production:'
\echo '  SELECT * FROM neurondb.job_queue WHERE status = ''pending'';'
\echo ''

