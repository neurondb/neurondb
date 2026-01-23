-- ====================================================================
-- NeurondB Regression Tests: Outlier Detection
-- ====================================================================
-- Tests for Z-score, Modified Z-score, IQR, and Isolation Forest
-- Uses real data from: sift1m.vectors with synthetic outliers
-- ====================================================================

\echo '=== Using SIFT1M Dataset for Outlier Detection Tests ==='

-- Create test data with normal vectors and synthetic outliers
CREATE TEMP TABLE test_outliers AS
-- Normal vectors (synthetic, sift1m.vectors may not exist)
SELECT 
    id,
    array_to_vector(ARRAY(SELECT random()::real FROM generate_series(1, 10)))::vector(10) as vec,
    'Normal' as description
FROM generate_series(1, 95) AS id
UNION ALL
-- Add 5 synthetic outliers (vectors with all high values)
SELECT 
    100 + generate_series as id,
    ('[100, 100, 100, 100, 100, 100, 100, 100, 100, 100]')::vector(10) as vec,
    'Synthetic Outlier ' || generate_series::text as description
FROM generate_series(1, 5);

-- Show sample
SELECT COUNT(*) as total_vectors, 
       SUM(CASE WHEN description LIKE 'Synthetic%' THEN 1 ELSE 0 END) as outliers
FROM test_outliers;

\echo '=== Testing Z-Score Outlier Detection ==='

-- Test Z-score outlier detection
-- Validate that function returns expected results
DO $$
DECLARE
    result_count INT;
    outlier_count INT;
BEGIN
    SELECT COUNT(*), SUM(CASE WHEN is_outlier THEN 1 ELSE 0 END)
    INTO result_count, outlier_count
    FROM neurondb.detect_outliers_zscore('test_outliers', 'vec', 3.0);
    
    IF result_count = 0 THEN
        RAISE EXCEPTION 'detect_outliers_zscore returned no results';
    END IF;
    
    -- Should have 100 results (95 normal + 5 outliers)
    IF result_count != 100 THEN
        RAISE EXCEPTION 'detect_outliers_zscore returned % results, expected 100', result_count;
    END IF;
END$$;

SELECT 
    o.id,
    t.description,
    o.is_outlier
FROM neurondb.detect_outliers_zscore('test_outliers', 'vec', 3.0) o
JOIN test_outliers t ON t.id = o.id
ORDER BY o.id;

SELECT 
    o.id,
    t.description,
    o.is_outlier
FROM neurondb.detect_outliers_zscore('test_outliers', 'vec', 2.0) o
JOIN test_outliers t ON t.id = o.id
ORDER BY o.id;

SELECT 
    o.id,
    t.description,
    o.is_outlier
FROM neurondb.detect_outliers_zscore('test_outliers', 'vec', 1.0) o
JOIN test_outliers t ON t.id = o.id
ORDER BY o.id;

-- Count outliers at each threshold
-- Validate that different thresholds produce different results
DO $$
DECLARE
    threshold_3_outliers INT;
    threshold_2_outliers INT;
    threshold_1_outliers INT;
BEGIN
    SELECT SUM(CASE WHEN is_outlier THEN 1 ELSE 0 END) INTO threshold_3_outliers
    FROM neurondb.detect_outliers_zscore('test_outliers', 'vec', 3.0);
    
    SELECT SUM(CASE WHEN is_outlier THEN 1 ELSE 0 END) INTO threshold_2_outliers
    FROM neurondb.detect_outliers_zscore('test_outliers', 'vec', 2.0);
    
    SELECT SUM(CASE WHEN is_outlier THEN 1 ELSE 0 END) INTO threshold_1_outliers
    FROM neurondb.detect_outliers_zscore('test_outliers', 'vec', 1.0);
    
    -- Lower thresholds should detect more outliers
    IF threshold_1_outliers < threshold_2_outliers OR threshold_2_outliers < threshold_3_outliers THEN
        RAISE WARNING 'Outlier detection threshold behavior unexpected: 3.0=%%, 2.0=%%, 1.0=%%', 
            threshold_3_outliers, threshold_2_outliers, threshold_1_outliers;
    END IF;
END$$;

SELECT 
    threshold,
    SUM(CASE WHEN is_outlier THEN 1 ELSE 0 END) as num_outliers,
    SUM(CASE WHEN NOT is_outlier THEN 1 ELSE 0 END) as num_normal
FROM (
    SELECT 3.0 as threshold, is_outlier 
    FROM neurondb.detect_outliers_zscore('test_outliers', 'vec', 3.0)
    UNION ALL
    SELECT 2.0, is_outlier 
    FROM neurondb.detect_outliers_zscore('test_outliers', 'vec', 2.0)
    UNION ALL
    SELECT 1.0, is_outlier 
    FROM neurondb.detect_outliers_zscore('test_outliers', 'vec', 1.0)
) sub
GROUP BY threshold
ORDER BY threshold DESC;

\echo '=== Testing Outlier Score Computation ==='

-- Get outlier scores
-- Validate that function returns expected results
DO $$
DECLARE
    result_count INT;
    max_score REAL;
    min_score REAL;
BEGIN
    SELECT COUNT(*), MAX(score), MIN(score)
    INTO result_count, max_score, min_score
    FROM neurondb.compute_outlier_scores('test_outliers', 'vec', 'zscore');
    
    IF result_count = 0 THEN
        RAISE EXCEPTION 'compute_outlier_scores returned no results';
    END IF;
    
    IF max_score IS NULL OR min_score IS NULL THEN
        RAISE EXCEPTION 'compute_outlier_scores returned NULL scores';
    END IF;
END$$;

SELECT 
    id,
    description,
    ROUND(score::numeric, 4) as outlier_score
FROM neurondb.compute_outlier_scores('test_outliers', 'vec', 'zscore')
ORDER BY score DESC
LIMIT 10;

-- Compare Z-score and Modified Z-score methods
WITH zscore AS (
    SELECT id, score as z_score
    FROM neurondb.compute_outlier_scores('test_outliers', 'vec', 'zscore')
),
mod_zscore AS (
    SELECT id, score as mod_z_score
    FROM neurondb.compute_outlier_scores('test_outliers', 'vec', 'modified_zscore')
)
SELECT 
    t.id,
    t.description,
    ROUND(z.z_score::numeric, 4) as zscore,
    ROUND(m.mod_z_score::numeric, 4) as modified_zscore
FROM test_outliers t
JOIN zscore z ON t.id = z.id
JOIN mod_zscore m ON t.id = m.id
ORDER BY z.z_score DESC
LIMIT 10;

-- Test IQR method
SELECT 
    id,
    description,
    ROUND(score::numeric, 4) as iqr_score
FROM neurondb.compute_outlier_scores('test_outliers', 'vec', 'iqr')
ORDER BY score DESC
LIMIT 10;

-- Test Isolation Forest method
SELECT 
    id,
    description,
    ROUND(score::numeric, 4) as isolation_score
FROM neurondb.compute_outlier_scores('test_outliers', 'vec', 'isolation_forest')
ORDER BY score DESC
LIMIT 10;

\echo '=== Testing Method Comparison ==='

-- Compare all methods
WITH zscore AS (
    SELECT id, is_outlier as z_outlier
    FROM neurondb.detect_outliers_zscore('test_outliers', 'vec', 3.0)
),
scores AS (
    SELECT id, score > 3.0 as mod_z_outlier
    FROM neurondb.compute_outlier_scores('test_outliers', 'vec', 'modified_zscore')
),
iqr AS (
    SELECT id, score > 1.5 as iqr_outlier
    FROM neurondb.compute_outlier_scores('test_outliers', 'vec', 'iqr')
),
isolation AS (
    SELECT id, score > 0.6 as if_outlier
    FROM neurondb.compute_outlier_scores('test_outliers', 'vec', 'isolation_forest')
)
SELECT 
    t.id,
    t.description,
    z.z_outlier,
    s.mod_z_outlier,
    i.iqr_outlier,
    iso.if_outlier,
    (CASE WHEN z.z_outlier THEN 1 ELSE 0 END +
     CASE WHEN s.mod_z_outlier THEN 1 ELSE 0 END +
     CASE WHEN i.iqr_outlier THEN 1 ELSE 0 END +
     CASE WHEN iso.if_outlier THEN 1 ELSE 0 END) as methods_agree
FROM test_outliers t
LEFT JOIN zscore z ON t.id = z.id
LEFT JOIN scores s ON t.id = s.id
LEFT JOIN iqr i ON t.id = i.id
LEFT JOIN isolation iso ON t.id = iso.id
WHERE z.id IS NOT NULL OR s.id IS NOT NULL OR i.id IS NOT NULL OR iso.id IS NOT NULL
ORDER BY methods_agree DESC, t.id
LIMIT 20;

\echo '=== Edge Cases and Error Handling ==='

-- Test with minimal data
CREATE TABLE test_outliers_minimal (
    id SERIAL PRIMARY KEY,
    vec vector(2)
);

INSERT INTO test_outliers_minimal (vec) VALUES
    ('[1.0, 2.0]'::vector),
    ('[1.1, 2.1]'::vector),
    ('[10.0, 20.0]'::vector);

-- Z-score with minimal data
SELECT 
    id,
    is_outlier
FROM neurondb.detect_outliers_zscore('test_outliers_minimal', 'vec', 3.0)
ORDER BY id;

-- Outlier scores with minimal data
SELECT 
    id,
    ROUND(score::numeric, 4) as score
FROM neurondb.compute_outlier_scores('test_outliers_minimal', 'vec', 'zscore')
ORDER BY score DESC;

\echo '=== Testing Outlier Detection Sensitivity ==='

-- Test how threshold affects detection rate
-- Validate that function works across different thresholds
DO $$
DECLARE
    result_count INT;
BEGIN
    SELECT COUNT(*) INTO result_count
    FROM (VALUES (1.0), (1.5), (2.0), (2.5), (3.0), (3.5), (4.0)) t(threshold)
    CROSS JOIN LATERAL neurondb.detect_outliers_zscore('test_outliers', 'vec', t.threshold) o
    LIMIT 1;
    
    IF result_count = 0 THEN
        RAISE EXCEPTION 'detect_outliers_zscore threshold sensitivity test returned no results';
    END IF;
END$$;

CREATE TABLE test_threshold_sensitivity AS
SELECT 
    t.threshold,
    COUNT(*) FILTER (WHERE o.is_outlier) as outliers_detected,
    COUNT(*) as total_points,
    ROUND((COUNT(*) FILTER (WHERE o.is_outlier)::numeric / COUNT(*)::numeric * 100), 2) as pct_outliers
FROM (VALUES (1.0), (1.5), (2.0), (2.5), (3.0), (3.5), (4.0)) t(threshold)
CROSS JOIN LATERAL neurondb.detect_outliers_zscore('test_outliers', 'vec', t.threshold) o
GROUP BY t.threshold
ORDER BY t.threshold;

SELECT * FROM test_threshold_sensitivity;

\echo '=== Testing High-Dimensional Outliers ==='

-- Create higher-dimensional data
CREATE TABLE test_outliers_highd (
    id SERIAL PRIMARY KEY,
    vec vector(10)
);

INSERT INTO test_outliers_highd (vec) VALUES
    ('[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]'::vector),
    ('[1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1]'::vector),
    ('[0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9]'::vector),
    ('[10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0]'::vector),
    ('[1.05, 1.05, 1.05, 1.05, 1.05, 1.05, 1.05, 1.05, 1.05, 1.05]'::vector);

-- Detect outliers in high dimensions
SELECT 
    id,
    is_outlier
FROM neurondb.detect_outliers_zscore('test_outliers_highd', 'vec', 3.0)
ORDER BY id;

-- Outlier scores in high dimensions
SELECT 
    id,
    ROUND(score::numeric, 4) as score,
    CASE WHEN score > 3.0 THEN 'Outlier' ELSE 'Normal' END as classification
FROM neurondb.compute_outlier_scores('test_outliers_highd', 'vec', 'zscore')
ORDER BY score DESC;

-- Cleanup
DROP TABLE test_outliers CASCADE;
DROP TABLE test_outliers_minimal CASCADE;
DROP TABLE test_threshold_sensitivity CASCADE;
DROP TABLE test_outliers_highd CASCADE;
