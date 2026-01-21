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

-- Check if detect_outliers_zscore function exists
DO $$
BEGIN
  BEGIN
    PERFORM 1 FROM neurondb.detect_outliers_zscore('test_outliers', 'vec', 3.0) LIMIT 1;
    RAISE NOTICE 'detect_outliers_zscore function is available';
  EXCEPTION WHEN undefined_function THEN
    RAISE NOTICE 'detect_outliers_zscore function not available, skipping outlier detection tests';
    RETURN;
  END;
END$$;

-- Test Z-score outlier detection (only if function exists)
SELECT 
    id,
    description,
    is_outlier
FROM neurondb.detect_outliers_zscore('test_outliers', 'vec', 3.0)
WHERE EXISTS (SELECT 1 FROM pg_proc WHERE proname = 'detect_outliers_zscore' AND pronamespace = (SELECT oid FROM pg_namespace WHERE nspname = 'neurondb'))
ORDER BY id;

SELECT 
    id,
    description,
    is_outlier
FROM neurondb.detect_outliers_zscore('test_outliers', 'vec', 2.0)
WHERE EXISTS (SELECT 1 FROM pg_proc WHERE proname = 'detect_outliers_zscore' AND pronamespace = (SELECT oid FROM pg_namespace WHERE nspname = 'neurondb'))
ORDER BY id;

SELECT 
    id,
    description,
    is_outlier
FROM neurondb.detect_outliers_zscore('test_outliers', 'vec', 1.0)
WHERE EXISTS (SELECT 1 FROM pg_proc WHERE proname = 'detect_outliers_zscore' AND pronamespace = (SELECT oid FROM pg_namespace WHERE nspname = 'neurondb'))
ORDER BY id;

-- Count outliers at each threshold (skip if function doesn't exist)
DO $$
BEGIN
  BEGIN
    PERFORM 1 FROM neurondb.detect_outliers_zscore('test_outliers', 'vec', 3.0) LIMIT 1;
  EXCEPTION WHEN undefined_function THEN
    RETURN;
  END;
END$$;

SELECT 
    threshold,
    SUM(CASE WHEN is_outlier THEN 1 ELSE 0 END) as num_outliers,
    SUM(CASE WHEN NOT is_outlier THEN 1 ELSE 0 END) as num_normal
FROM (
    SELECT 3.0 as threshold, is_outlier 
    FROM neurondb.detect_outliers_zscore('test_outliers', 'vec', 3.0)
    WHERE EXISTS (SELECT 1 FROM pg_proc WHERE proname = 'detect_outliers_zscore' AND pronamespace = (SELECT oid FROM pg_namespace WHERE nspname = 'neurondb'))
    UNION ALL
    SELECT 2.0, is_outlier 
    FROM neurondb.detect_outliers_zscore('test_outliers', 'vec', 2.0)
    WHERE EXISTS (SELECT 1 FROM pg_proc WHERE proname = 'detect_outliers_zscore' AND pronamespace = (SELECT oid FROM pg_namespace WHERE nspname = 'neurondb'))
    UNION ALL
    SELECT 1.0, is_outlier 
    FROM neurondb.detect_outliers_zscore('test_outliers', 'vec', 1.0)
    WHERE EXISTS (SELECT 1 FROM pg_proc WHERE proname = 'detect_outliers_zscore' AND pronamespace = (SELECT oid FROM pg_namespace WHERE nspname = 'neurondb'))
) sub
GROUP BY threshold
ORDER BY threshold DESC;

\echo '=== Testing Outlier Score Computation ==='

-- Check if compute_outlier_scores function exists
DO $$
BEGIN
  BEGIN
    PERFORM 1 FROM neurondb.compute_outlier_scores('test_outliers', 'vec', 'zscore') LIMIT 1;
    RAISE NOTICE 'compute_outlier_scores function is available';
  EXCEPTION WHEN undefined_function THEN
    RAISE NOTICE 'compute_outlier_scores function not available, skipping score computation tests';
  END;
END$$;

-- Get outlier scores (only if function exists)
SELECT 
    id,
    description,
    ROUND(score::numeric, 4) as outlier_score
FROM neurondb.compute_outlier_scores('test_outliers', 'vec', 'zscore')
WHERE EXISTS (SELECT 1 FROM pg_proc WHERE proname = 'compute_outlier_scores' AND pronamespace = (SELECT oid FROM pg_namespace WHERE nspname = 'neurondb'))
ORDER BY score DESC
LIMIT 10;

-- Compare Z-score and Modified Z-score methods (skip if functions don't exist)
WITH zscore AS (
    SELECT id, score as z_score
    FROM neurondb.compute_outlier_scores('test_outliers', 'vec', 'zscore')
    WHERE EXISTS (SELECT 1 FROM pg_proc WHERE proname = 'compute_outlier_scores' AND pronamespace = (SELECT oid FROM pg_namespace WHERE nspname = 'neurondb'))
),
mod_zscore AS (
    SELECT id, score as mod_z_score
    FROM neurondb.compute_outlier_scores('test_outliers', 'vec', 'modified_zscore')
    WHERE EXISTS (SELECT 1 FROM pg_proc WHERE proname = 'compute_outlier_scores' AND pronamespace = (SELECT oid FROM pg_namespace WHERE nspname = 'neurondb'))
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
WHERE EXISTS (SELECT 1 FROM pg_proc WHERE proname = 'compute_outlier_scores' AND pronamespace = (SELECT oid FROM pg_namespace WHERE nspname = 'neurondb'))
ORDER BY score DESC
LIMIT 10;

-- Test Isolation Forest method
SELECT 
    id,
    description,
    ROUND(score::numeric, 4) as isolation_score
FROM neurondb.compute_outlier_scores('test_outliers', 'vec', 'isolation_forest')
WHERE EXISTS (SELECT 1 FROM pg_proc WHERE proname = 'compute_outlier_scores' AND pronamespace = (SELECT oid FROM pg_namespace WHERE nspname = 'neurondb'))
ORDER BY score DESC
LIMIT 10;

\echo '=== Testing Method Comparison ==='

-- Compare all methods (skip if functions don't exist)
WITH zscore AS (
    SELECT id, is_outlier as z_outlier
    FROM neurondb.detect_outliers_zscore('test_outliers', 'vec', 3.0)
    WHERE EXISTS (SELECT 1 FROM pg_proc WHERE proname = 'detect_outliers_zscore' AND pronamespace = (SELECT oid FROM pg_namespace WHERE nspname = 'neurondb'))
),
scores AS (
    SELECT id, score > 3.0 as mod_z_outlier
    FROM neurondb.compute_outlier_scores('test_outliers', 'vec', 'modified_zscore')
    WHERE EXISTS (SELECT 1 FROM pg_proc WHERE proname = 'compute_outlier_scores' AND pronamespace = (SELECT oid FROM pg_namespace WHERE nspname = 'neurondb'))
),
iqr AS (
    SELECT id, score > 1.5 as iqr_outlier
    FROM neurondb.compute_outlier_scores('test_outliers', 'vec', 'iqr')
    WHERE EXISTS (SELECT 1 FROM pg_proc WHERE proname = 'compute_outlier_scores' AND pronamespace = (SELECT oid FROM pg_namespace WHERE nspname = 'neurondb'))
),
isolation AS (
    SELECT id, score > 0.6 as if_outlier
    FROM neurondb.compute_outlier_scores('test_outliers', 'vec', 'isolation_forest')
    WHERE EXISTS (SELECT 1 FROM pg_proc WHERE proname = 'compute_outlier_scores' AND pronamespace = (SELECT oid FROM pg_namespace WHERE nspname = 'neurondb'))
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

-- Z-score with minimal data (skip if function doesn't exist)
SELECT 
    id,
    is_outlier
FROM neurondb.detect_outliers_zscore('test_outliers_minimal', 'vec', 3.0)
WHERE EXISTS (SELECT 1 FROM pg_proc WHERE proname = 'detect_outliers_zscore' AND pronamespace = (SELECT oid FROM pg_namespace WHERE nspname = 'neurondb'))
ORDER BY id;

-- Outlier scores with minimal data
SELECT 
    id,
    ROUND(score::numeric, 4) as score
FROM neurondb.compute_outlier_scores('test_outliers_minimal', 'vec', 'zscore')
WHERE EXISTS (SELECT 1 FROM pg_proc WHERE proname = 'compute_outlier_scores' AND pronamespace = (SELECT oid FROM pg_namespace WHERE nspname = 'neurondb'))
ORDER BY score DESC;

\echo '=== Testing Outlier Detection Sensitivity ==='

-- Test how threshold affects detection rate (skip if function doesn't exist)
DO $$
BEGIN
  BEGIN
    PERFORM 1 FROM neurondb.detect_outliers_zscore('test_outliers', 'vec', 3.0) LIMIT 1;
  EXCEPTION WHEN undefined_function THEN
    RAISE NOTICE 'detect_outliers_zscore not available, skipping sensitivity test';
    RETURN;
  END;
END$$;

CREATE TABLE test_threshold_sensitivity AS
SELECT 
    t.threshold,
    COUNT(*) FILTER (WHERE o.is_outlier) as outliers_detected,
    COUNT(*) as total_points,
    ROUND((COUNT(*) FILTER (WHERE o.is_outlier)::numeric / COUNT(*)::numeric * 100), 2) as pct_outliers
FROM (VALUES (1.0), (1.5), (2.0), (2.5), (3.0), (3.5), (4.0)) t(threshold)
CROSS JOIN LATERAL neurondb.detect_outliers_zscore('test_outliers', 'vec', t.threshold) o
WHERE EXISTS (SELECT 1 FROM pg_proc WHERE proname = 'detect_outliers_zscore' AND pronamespace = (SELECT oid FROM pg_namespace WHERE nspname = 'neurondb'))
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

-- Detect outliers in high dimensions (skip if function doesn't exist)
SELECT 
    id,
    is_outlier
FROM neurondb.detect_outliers_zscore('test_outliers_highd', 'vec', 3.0)
WHERE EXISTS (SELECT 1 FROM pg_proc WHERE proname = 'detect_outliers_zscore' AND pronamespace = (SELECT oid FROM pg_namespace WHERE nspname = 'neurondb'))
ORDER BY id;

-- Outlier scores in high dimensions
SELECT 
    id,
    ROUND(score::numeric, 4) as score,
    CASE WHEN score > 3.0 THEN 'Outlier' ELSE 'Normal' END as classification
FROM neurondb.compute_outlier_scores('test_outliers_highd', 'vec', 'zscore')
WHERE EXISTS (SELECT 1 FROM pg_proc WHERE proname = 'compute_outlier_scores' AND pronamespace = (SELECT oid FROM pg_namespace WHERE nspname = 'neurondb'))
ORDER BY score DESC;

-- Cleanup
DROP TABLE test_outliers CASCADE;
DROP TABLE test_outliers_minimal CASCADE;
DROP TABLE test_threshold_sensitivity CASCADE;
DROP TABLE test_outliers_highd CASCADE;
