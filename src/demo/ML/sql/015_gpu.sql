\set ON_ERROR_STOP on
\set QUIET on

-- ============================================================================
-- EXTREME GPU Test - Large Scale Vector Operations
-- ============================================================================

CREATE EXTENSION IF NOT EXISTS neurondb;

\echo '══════════════════════════════════════════════════════════════════'
\echo '  EXTREME GPU Performance Test - Apple M4 Metal'
\echo '══════════════════════════════════════════════════════════════════'
\echo ''

-- Create VERY LARGE dataset for true GPU performance
\echo 'Step 1: Creating EXTREME dataset...'
\echo '  • 50,000 vectors'
\echo '  • 2048 dimensions each'
\echo '  • ~400 MB total data'
\echo ''

DROP TABLE IF EXISTS extreme_vectors CASCADE;
CREATE TABLE extreme_vectors AS
SELECT 
    i as id,
    array_agg(random()::real ORDER BY j)::real[]::vector(2048) as vec
FROM generate_series(1, 50000) i,
     generate_series(1, 2048) j
GROUP BY i;

\echo '  ✓ Dataset ready: 50,000 vectors x 2048 dimensions'
\echo ''

-- ============================================================================
-- TEST 1: CPU - K-Nearest Neighbors Search
-- ============================================================================

\echo '══════════════════════════════════════════════════════════════════'
\echo ' TEST 1: CPU - KNN Search (100 queries x 50K vectors)'
\echo '══════════════════════════════════════════════════════════════════'
\echo ''

SET neurondb.compute_mode = false;

\echo 'CPU: Finding top-10 nearest neighbors for 100 queries'
\echo '  (100 queries × 50,000 vectors × 2048 dims = 5M distances)'
\echo ''

\timing on
WITH query_vectors AS (
    SELECT id, vec FROM extreme_vectors WHERE id <= 100
)
SELECT 
    COUNT(*) as total_comparisons,
    AVG(distance)::numeric(10,4) as avg_distance
FROM (
    SELECT 
        q.id as query_id,
        e.id as result_id,
        q.vec <-> e.vec as distance,
        ROW_NUMBER() OVER (PARTITION BY q.id ORDER BY q.vec <-> e.vec) as rank
    FROM query_vectors q
    CROSS JOIN extreme_vectors e
    WHERE q.id != e.id
) ranked
WHERE rank <= 10;
\timing off

\echo ''

-- ============================================================================
-- TEST 2: GPU - Same KNN Search
-- ============================================================================

\echo '══════════════════════════════════════════════════════════════════'
\echo ' TEST 2: GPU - KNN Search (SAME workload)'
\echo '══════════════════════════════════════════════════════════════════'
\echo ''

SET neurondb.compute_mode = true;
SET neurondb.gpu_backend = 'metal';

\echo 'GPU: Finding top-10 nearest neighbors for 100 queries'
\echo '  Using vector_l2_distance_gpu() with Metal backend'
\echo ''

\timing on
WITH query_vectors AS (
    SELECT id, vec FROM extreme_vectors WHERE id <= 100
)
SELECT 
    COUNT(*) as total_comparisons,
    AVG(distance)::numeric(10,4) as avg_distance
FROM (
    SELECT 
        q.id as query_id,
        e.id as result_id,
        vector_l2_distance_gpu(q.vec, e.vec) as distance,
        ROW_NUMBER() OVER (PARTITION BY q.id ORDER BY vector_l2_distance_gpu(q.vec, e.vec)) as rank
    FROM query_vectors q
    CROSS JOIN extreme_vectors e
    WHERE q.id != e.id
) ranked
WHERE rank <= 10;
\timing off

\echo ''

-- ============================================================================
-- TEST 3: CPU - Batch Distance Matrix
-- ============================================================================

\echo '══════════════════════════════════════════════════════════════════'
\echo ' TEST 3: CPU - Distance Matrix (500 x 1000 vectors)'
\echo '══════════════════════════════════════════════════════════════════'
\echo ''

SET neurondb.compute_mode = false;

\echo 'CPU: Computing 500,000 distance calculations'
\echo '  (500 queries × 1,000 targets × 2048 dimensions)'
\echo ''

\timing on
SELECT 
    COUNT(*) as total_distances,
    MIN(distance)::numeric(10,4) as min_dist,
    MAX(distance)::numeric(10,4) as max_dist,
    AVG(distance)::numeric(10,4) as avg_dist
FROM (
    SELECT 
        v1.id as id1,
        v2.id as id2,
        v1.vec <-> v2.vec as distance
    FROM extreme_vectors v1
    CROSS JOIN extreme_vectors v2
    WHERE v1.id <= 500 AND v2.id BETWEEN 1001 AND 2000
) distances;
\timing off

\echo ''

-- ============================================================================
-- TEST 4: GPU - Batch Distance Matrix
-- ============================================================================

\echo '══════════════════════════════════════════════════════════════════'
\echo ' TEST 4: GPU - Distance Matrix (SAME workload)'
\echo '══════════════════════════════════════════════════════════════════'
\echo ''

SET neurondb.compute_mode = true;
SET neurondb.gpu_backend = 'metal';

\echo 'GPU: Computing 500,000 distance calculations'
\echo '  Using GPU-accelerated distance functions'
\echo ''

\timing on
SELECT 
    COUNT(*) as total_distances,
    MIN(distance)::numeric(10,4) as min_dist,
    MAX(distance)::numeric(10,4) as max_dist,
    AVG(distance)::numeric(10,4) as avg_dist
FROM (
    SELECT 
        v1.id as id1,
        v2.id as id2,
        vector_l2_distance_gpu(v1.vec, v2.vec) as distance
    FROM extreme_vectors v1
    CROSS JOIN extreme_vectors v2
    WHERE v1.id <= 500 AND v2.id BETWEEN 1001 AND 2000
) distances;
\timing off

\echo ''

-- ============================================================================
-- TEST 5: CPU - Clustering Distance Matrix
-- ============================================================================

\echo '══════════════════════════════════════════════════════════════════'
\echo ' TEST 5: CPU - All-to-All Distance (1000 x 1000 vectors)'
\echo '══════════════════════════════════════════════════════════════════'
\echo ''

SET neurondb.compute_mode = false;

\echo 'CPU: Computing 1 MILLION distances for clustering'
\echo '  (1,000 × 1,000 vectors × 2048 dimensions)'
\echo ''

\timing on
SELECT 
    COUNT(*) as total_distances,
    AVG(distance)::numeric(10,4) as avg_dist,
    STDDEV(distance)::numeric(10,4) as stddev_dist
FROM (
    SELECT 
        v1.vec <-> v2.vec as distance
    FROM extreme_vectors v1
    CROSS JOIN extreme_vectors v2
    WHERE v1.id <= 1000 AND v2.id <= 1000 AND v1.id < v2.id
) distances;
\timing off

\echo ''

-- ============================================================================
-- TEST 6: GPU - Clustering Distance Matrix
-- ============================================================================

\echo '══════════════════════════════════════════════════════════════════'
\echo ' TEST 6: GPU - All-to-All Distance (SAME workload)'
\echo '══════════════════════════════════════════════════════════════════'
\echo ''

SET neurondb.compute_mode = true;
SET neurondb.gpu_backend = 'metal';

\echo 'GPU: Computing 1 MILLION distances for clustering'
\echo '  Using GPU batch operations'
\echo ''

\timing on
SELECT 
    COUNT(*) as total_distances,
    AVG(distance)::numeric(10,4) as avg_dist,
    STDDEV(distance)::numeric(10,4) as stddev_dist
FROM (
    SELECT 
        vector_l2_distance_gpu(v1.vec, v2.vec) as distance
    FROM extreme_vectors v1
    CROSS JOIN extreme_vectors v2
    WHERE v1.id <= 1000 AND v2.id <= 1000 AND v1.id < v2.id
) distances;
\timing off

\echo ''

-- ============================================================================
-- SUMMARY
-- ============================================================================

\echo '══════════════════════════════════════════════════════════════════'
\echo '                 EXTREME Performance Summary'
\echo '══════════════════════════════════════════════════════════════════'
\echo ''
\echo 'Dataset: 50,000 vectors x 2048 dimensions (~400 MB)'
\echo ''
\echo 'Compare the times above for:'
\echo '  • TEST 1 vs TEST 2: KNN Search (5M distances)'
\echo '  • TEST 3 vs TEST 4: Distance Matrix (500K distances)'
\echo '  • TEST 5 vs TEST 6: Clustering (1M distances)'
\echo ''
\echo 'GPU Configuration:'
SELECT name, setting FROM pg_settings 
WHERE name LIKE 'neurondb.gpu%' AND name IN ('neurondb.compute_mode', 'neurondb.gpu_backend');
\echo ''
\echo '══════════════════════════════════════════════════════════════════'

