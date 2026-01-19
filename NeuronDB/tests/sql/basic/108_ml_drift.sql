-- ====================================================================
-- NeurondB Regression Tests: Drift Detection
-- ====================================================================
-- Tests for Centroid Drift and Distribution Divergence
-- Uses real data from: sift1m.vectors (simulating baseline vs current)
-- ====================================================================

\echo '=== Using SIFT1M Dataset for Drift Detection Tests ==='

-- Create baseline data (vectors 1-500)
CREATE TEMP TABLE test_drift_baseline AS
SELECT 
    id,
    array_to_vector(embedding[1:8])::vector(8) as vec,
    CASE WHEN id % 2 = 0 THEN 'A' ELSE 'B' END as category
FROM sift1m.vectors
WHERE id BETWEEN 1 AND 500
LIMIT 500;

-- Create current data (vectors 501-1000 for category A, 1-500 for category B)
-- Category A will have different distribution (drift), B will be similar (no drift)
CREATE TEMP TABLE test_drift_current AS
SELECT 
    id,
    vec,
    category
FROM (
    -- Category A: use different vectors (simulating drift)
    SELECT 
        id,
        array_to_vector(embedding[1:8])::vector(8) as vec,
        'A' as category
    FROM sift1m.vectors
    WHERE id BETWEEN 501 AND 750 AND id % 2 = 0
    LIMIT 250
    UNION ALL
    -- Category B: use same range as baseline (no drift)
    SELECT 
        id,
        array_to_vector(embedding[1:8])::vector(8) as vec,
        'B' as category
    FROM sift1m.vectors
    WHERE id BETWEEN 1 AND 500 AND id % 2 = 1
    LIMIT 250
) sub;

-- Show sample
SELECT category, COUNT(*) as count
FROM test_drift_baseline
GROUP BY category
UNION ALL
SELECT category, COUNT(*) as count
FROM test_drift_current
GROUP BY category
ORDER BY category;

\echo '=== Testing Centroid Drift Detection ==='

-- Detect drift between baseline and current for category A
SELECT 
    baseline_centroid,
    current_centroid,
    ROUND(drift_distance::numeric, 4) as drift,
    has_drifted
FROM neurondb.detect_centroid_drift(
    'test_drift_baseline', 'vec',
    'test_drift_current', 'vec',
    'category', 'A',
    0.3  -- threshold
);

-- Detect drift for category B (should show no drift)
SELECT 
    baseline_centroid,
    current_centroid,
    ROUND(drift_distance::numeric, 4) as drift,
    has_drifted
FROM neurondb.detect_centroid_drift(
    'test_drift_baseline', 'vec',
    'test_drift_current', 'vec',
    'category', 'B',
    0.3
);

-- Test with different thresholds
SELECT 
    threshold,
    (SELECT has_drifted 
     FROM neurondb.detect_centroid_drift(
         'test_drift_baseline', 'vec',
         'test_drift_current', 'vec',
         'category', 'A', threshold
     )) as has_drifted
FROM (VALUES (0.1), (0.3), (0.5), (1.0)) t(threshold)
ORDER BY threshold;

\echo '=== Testing Distribution Divergence ==='

-- Test distribution divergence (KL-like divergence)
SELECT 
    ROUND(divergence::numeric, 4) as kl_divergence,
    is_divergent
FROM neurondb.compute_distribution_divergence(
    'test_drift_baseline', 'vec',
    'test_drift_current', 'vec',
    'category', 'A',
    0.5  -- threshold
);

-- Test for stable category B
SELECT 
    ROUND(divergence::numeric, 4) as kl_divergence,
    is_divergent
FROM neurondb.compute_distribution_divergence(
    'test_drift_baseline', 'vec',
    'test_drift_current', 'vec',
    'category', 'B',
    0.5
);

\echo '=== Testing Temporal Drift Monitoring ==='

-- Create time-series data for drift monitoring
CREATE TABLE test_drift_timeseries (
    id SERIAL PRIMARY KEY,
    vec vector(3),
    timestamp TIMESTAMPTZ
);

-- Insert data with gradual drift over time
INSERT INTO test_drift_timeseries (vec, timestamp) VALUES
    -- Time 0: baseline
    ('[1.0, 1.0, 1.0]'::vector, NOW() - INTERVAL '10 days'),
    ('[1.1, 0.9, 1.0]'::vector, NOW() - INTERVAL '10 days'),
    ('[0.9, 1.1, 1.0]'::vector, NOW() - INTERVAL '10 days'),
    -- Time 1: slight drift
    ('[1.2, 1.2, 1.2]'::vector, NOW() - INTERVAL '8 days'),
    ('[1.3, 1.1, 1.2]'::vector, NOW() - INTERVAL '8 days'),
    ('[1.1, 1.3, 1.2]'::vector, NOW() - INTERVAL '8 days'),
    -- Time 2: more drift
    ('[1.5, 1.5, 1.5]'::vector, NOW() - INTERVAL '6 days'),
    ('[1.6, 1.4, 1.5]'::vector, NOW() - INTERVAL '6 days'),
    ('[1.4, 1.6, 1.5]'::vector, NOW() - INTERVAL '6 days'),
    -- Time 3: significant drift
    ('[2.0, 2.0, 2.0]'::vector, NOW() - INTERVAL '4 days'),
    ('[2.1, 1.9, 2.0]'::vector, NOW() - INTERVAL '4 days'),
    ('[1.9, 2.1, 2.0]'::vector, NOW() - INTERVAL '4 days'),
    -- Time 4: current (drifted)
    ('[2.5, 2.5, 2.5]'::vector, NOW() - INTERVAL '2 days'),
    ('[2.6, 2.4, 2.5]'::vector, NOW() - INTERVAL '2 days'),
    ('[2.4, 2.6, 2.5]'::vector, NOW() - INTERVAL '2 days');

-- Monitor drift with 3-day window
SELECT 
    window_start,
    window_end,
    centroid,
    ROUND(drift_from_baseline::numeric, 4) as drift
FROM neurondb.monitor_drift_timeseries(
    'test_drift_timeseries', 
    'vec',
    'timestamp',
    INTERVAL '3 days'
)
ORDER BY window_start;

\echo '=== Testing Drift Sensitivity ==='

-- Test how threshold affects drift detection
WITH drift_tests AS (
    SELECT 
        t.threshold,
        (SELECT has_drifted 
         FROM neurondb.detect_centroid_drift(
             'test_drift_baseline', 'vec',
             'test_drift_current', 'vec',
             'category', 'A', t.threshold
         )) as detected_drift,
        (SELECT drift_distance 
         FROM neurondb.detect_centroid_drift(
             'test_drift_baseline', 'vec',
             'test_drift_current', 'vec',
             'category', 'A', t.threshold
         )) as actual_drift
    FROM (VALUES (0.1), (0.2), (0.3), (0.5), (0.7), (1.0)) t(threshold)
)
SELECT 
    threshold,
    ROUND(actual_drift::numeric, 4) as drift_distance,
    detected_drift,
    CASE 
        WHEN detected_drift THEN 'Alarm'
        ELSE 'OK'
    END as status
FROM drift_tests
ORDER BY threshold;

\echo '=== Edge Cases and Error Handling ==='

-- Test with no data in current
CREATE TABLE test_drift_empty (
    id SERIAL PRIMARY KEY,
    vec vector(2),
    category TEXT
);

-- Should handle gracefully
SELECT 
    has_drifted
FROM neurondb.detect_centroid_drift(
    'test_drift_baseline', 'vec',
    'test_drift_empty', 'vec',
    'category', 'A',
    0.3
);

-- Test with identical distributions (no drift)
CREATE TABLE test_drift_identical AS 
SELECT * FROM test_drift_baseline;

SELECT 
    ROUND(drift_distance::numeric, 4) as drift,
    has_drifted
FROM neurondb.detect_centroid_drift(
    'test_drift_baseline', 'vec',
    'test_drift_identical', 'vec',
    'category', 'A',
    0.1  -- Even with tight threshold
);

\echo '=== Testing Multi-Category Drift ==='

-- Monitor drift across all categories
WITH all_categories AS (
    SELECT DISTINCT category FROM test_drift_baseline
)
SELECT 
    ac.category,
    ROUND(d.drift_distance::numeric, 4) as drift,
    d.has_drifted,
    CASE 
        WHEN d.has_drifted THEN 'ALERT: Drift detected'
        ELSE 'OK'
    END as status
FROM all_categories ac
CROSS JOIN LATERAL neurondb.detect_centroid_drift(
    'test_drift_baseline', 'vec',
    'test_drift_current', 'vec',
    'category', ac.category,
    0.3
) d
ORDER BY d.drift_distance DESC;

-- Cleanup
DROP TABLE test_drift_baseline CASCADE;
DROP TABLE test_drift_current CASCADE;
DROP TABLE test_drift_timeseries CASCADE;
DROP TABLE test_drift_empty CASCADE;
DROP TABLE test_drift_identical CASCADE;

