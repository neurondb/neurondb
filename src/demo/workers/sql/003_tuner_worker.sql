-- ============================================================================
-- NeuronDB Workers Demo - neuranmon (Auto-Tuner Worker) Tests
-- ============================================================================
-- Purpose: Test automatic query performance tuning
-- Worker: neuranmon - Adjusts ef_search, hybrid weights based on SLOs
-- ============================================================================

\echo '=========================================='
\echo 'neuranmon (Auto-Tuner Worker) Tests'
\echo '=========================================='

SET search_path TO worker_demo, neurondb, public;

-- ============================================================================
-- TEST 1: Query Metrics Table Structure
-- ============================================================================
\echo ''
\echo 'TEST 1: Query Metrics Table Structure'
\echo '--------------------------------------------'

-- Show table structure
\d neurondb.query_metrics

\echo ''
\echo '✓ Query metrics table verified'

-- ============================================================================
-- TEST 2: Insert Baseline Query Metrics
-- ============================================================================
\echo ''
\echo 'TEST 2: Insert Baseline Query Metrics'
\echo '--------------------------------------------'

-- Insert metrics for ef_search = 100
INSERT INTO neurondb.query_metrics (
    query_id, table_oid, index_oid, ef_search, k, 
    latency_ms, recall, recorded_at
)
SELECT 
    generate_series(1, 50),
    16384,  -- dummy OID
    16385,  -- dummy OID
    100,
    10,
    40.0 + (random() * 20.0),  -- 40-60ms
    0.92 + (random() * 0.05),  -- 0.92-0.97 recall
    now() - (random() * interval '10 minutes');

\echo 'Inserted 50 metrics for ef_search=100'

-- Insert metrics for ef_search = 150
INSERT INTO neurondb.query_metrics (
    query_id, table_oid, index_oid, ef_search, k, 
    latency_ms, recall, recorded_at
)
SELECT 
    generate_series(51, 100),
    16384,
    16385,
    150,
    10,
    70.0 + (random() * 20.0),  -- 70-90ms
    0.96 + (random() * 0.03),  -- 0.96-0.99 recall
    now() - (random() * interval '10 minutes');

\echo 'Inserted 50 metrics for ef_search=150'

-- Insert metrics for ef_search = 200
INSERT INTO neurondb.query_metrics (
    query_id, table_oid, index_oid, ef_search, k, 
    latency_ms, recall, recorded_at
)
SELECT 
    generate_series(101, 150),
    16384,
    16385,
    200,
    10,
    110.0 + (random() * 30.0),  -- 110-140ms
    0.98 + (random() * 0.02),  -- 0.98-1.00 recall
    now() - (random() * interval '10 minutes');

\echo 'Inserted 50 metrics for ef_search=200'

\echo ''
\echo '✓ 150 baseline metrics inserted'

-- ============================================================================
-- TEST 3: Analyze Performance by ef_search
-- ============================================================================
\echo ''
\echo 'TEST 3: Performance Analysis by ef_search'
\echo '--------------------------------------------'

SELECT 
    ef_search,
    COUNT(*) as query_count,
    AVG(latency_ms)::numeric(10,2) as avg_latency_ms,
    STDDEV(latency_ms)::numeric(10,2) as stddev_latency,
    MIN(latency_ms)::numeric(10,2) as min_latency_ms,
    MAX(latency_ms)::numeric(10,2) as max_latency_ms,
    AVG(recall)::numeric(5,4) as avg_recall,
    MIN(recall)::numeric(5,4) as min_recall,
    MAX(recall)::numeric(5,4) as max_recall
FROM neurondb.query_metrics
WHERE recorded_at > now() - interval '1 hour'
GROUP BY ef_search
ORDER BY ef_search;

\echo ''
\echo '✓ Performance analysis complete'

-- ============================================================================
-- TEST 4: Tuning Recommendations
-- ============================================================================
\echo ''
\echo 'TEST 4: Auto-Tuning Recommendations'
\echo '--------------------------------------------'

-- SLO targets: latency < 100ms, recall > 0.95
WITH performance_data AS (
    SELECT 
        ef_search,
        AVG(latency_ms) as avg_latency,
        AVG(recall) as avg_recall,
        COUNT(*) as sample_size
    FROM neurondb.query_metrics
    WHERE recorded_at > now() - interval '10 minutes'
    GROUP BY ef_search
)
SELECT 
    ef_search,
    sample_size,
    avg_latency::numeric(10,2),
    avg_recall::numeric(5,4),
    CASE 
        WHEN avg_latency > 100.0 AND avg_recall >= 0.95 THEN 'TUNE DOWN: Latency too high, recall acceptable'
        WHEN avg_latency <= 100.0 AND avg_recall < 0.95 THEN 'TUNE UP: Recall too low, latency acceptable'
        WHEN avg_latency > 100.0 AND avg_recall < 0.95 THEN 'CRITICAL: Both metrics out of SLO'
        ELSE 'OPTIMAL: Within SLO targets'
    END as recommendation,
    CASE 
        WHEN avg_latency > 100.0 THEN ef_search - 50
        WHEN avg_recall < 0.95 THEN ef_search + 50
        ELSE ef_search
    END as suggested_ef_search
FROM performance_data
ORDER BY ef_search;

\echo ''
\echo '✓ Tuning recommendations generated'

-- ============================================================================
-- TEST 5: Latency Distribution Analysis
-- ============================================================================
\echo ''
\echo 'TEST 5: Latency Distribution Analysis'
\echo '--------------------------------------------'

-- P50, P90, P95, P99 percentiles
SELECT 
    ef_search,
    COUNT(*) as sample_size,
    percentile_cont(0.5) WITHIN GROUP (ORDER BY latency_ms)::numeric(10,2) as p50_latency,
    percentile_cont(0.90) WITHIN GROUP (ORDER BY latency_ms)::numeric(10,2) as p90_latency,
    percentile_cont(0.95) WITHIN GROUP (ORDER BY latency_ms)::numeric(10,2) as p95_latency,
    percentile_cont(0.99) WITHIN GROUP (ORDER BY latency_ms)::numeric(10,2) as p99_latency
FROM neurondb.query_metrics
WHERE recorded_at > now() - interval '1 hour'
GROUP BY ef_search
ORDER BY ef_search;

\echo ''
\echo '✓ Latency distribution analyzed'

-- ============================================================================
-- TEST 6: Recall vs Latency Tradeoff
-- ============================================================================
\echo ''
\echo 'TEST 6: Recall vs Latency Tradeoff'
\echo '--------------------------------------------'

-- Show the tradeoff curve
SELECT 
    ef_search,
    AVG(latency_ms)::numeric(10,2) as avg_latency_ms,
    AVG(recall)::numeric(5,4) as avg_recall,
    (AVG(recall) * 1000 / AVG(latency_ms))::numeric(10,4) as recall_per_ms
FROM neurondb.query_metrics
WHERE recorded_at > now() - interval '1 hour'
GROUP BY ef_search
ORDER BY ef_search;

\echo ''
\echo 'Interpretation:'
\echo '  - Higher ef_search = Better recall, Higher latency'
\echo '  - recall_per_ms shows efficiency (higher is better)'
\echo ''
\echo '✓ Tradeoff analysis complete'

-- ============================================================================
-- TEST 7: Time-Series Trends
-- ============================================================================
\echo ''
\echo 'TEST 7: Time-Series Performance Trends'
\echo '--------------------------------------------'

-- Performance over time
SELECT 
    date_trunc('minute', recorded_at) as time_bucket,
    ef_search,
    COUNT(*) as queries,
    AVG(latency_ms)::numeric(10,2) as avg_latency,
    AVG(recall)::numeric(5,4) as avg_recall
FROM neurondb.query_metrics
WHERE recorded_at > now() - interval '15 minutes'
GROUP BY date_trunc('minute', recorded_at), ef_search
ORDER BY time_bucket DESC, ef_search
LIMIT 10;

\echo ''
\echo '✓ Time-series trends analyzed'

-- ============================================================================
-- TEST 8: Tuner Configuration Check
-- ============================================================================
\echo ''
\echo 'TEST 8: Tuner Worker Configuration'
\echo '--------------------------------------------'

SELECT 
    'neuranmon Configuration' as section,
    current_setting('neurondb.neuranmon_naptime') as naptime_ms,
    current_setting('neurondb.neuranmon_sample_size') as sample_size,
    current_setting('neurondb.neuranmon_target_latency') as target_latency_ms,
    current_setting('neurondb.neuranmon_target_recall') as target_recall,
    current_setting('neurondb.neuranmon_enabled') as enabled;

\echo ''
\echo '✓ Configuration verified'

\echo ''
\echo '=========================================='
\echo 'neuranmon Auto-Tuner Tests Complete!'
\echo '=========================================='
\echo ''
\echo 'Key Findings:'
\echo '  - Query metrics collection operational'
\echo '  - Performance analysis working'
\echo '  - Tuning recommendations generated'
\echo '  - SLO tracking functional'
\echo ''
\echo 'Worker Purpose:'
\echo '  neuranmon automatically tunes query parameters:'
\echo '  - Adjusts ef_search based on latency/recall SLOs'
\echo '  - Optimizes hybrid search weights'
\echo '  - Rotates caches for freshness'
\echo '  - Records recall@k metrics'
\echo '  - Exports Prometheus metrics'
\echo ''
\echo 'Tuning Strategy:'
\echo '  - Target: latency < 100ms, recall > 0.95'
\echo '  - If latency high: Decrease ef_search'
\echo '  - If recall low: Increase ef_search'
\echo '  - Continuous adaptation to workload'
\echo ''

