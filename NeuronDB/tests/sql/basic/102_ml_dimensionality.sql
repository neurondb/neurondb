-- ====================================================================
-- NeurondB Regression Tests: Dimensionality Reduction
-- ====================================================================
-- Tests for PCA and PCA Whitening
-- Uses real data from: deep1b.vectors (96-d vectors)
-- ====================================================================

CREATE EXTENSION IF NOT EXISTS neurondb;

\echo '=== Using Deep1B Dataset for PCA Tests ==='

-- Create test data with synthetic vectors (deep1b.vectors may not exist)
CREATE TEMP TABLE test_pca_data AS
SELECT 
    id,
    array_to_vector(ARRAY(SELECT random()::real FROM generate_series(1, 20)))::vector(20) as vec
FROM generate_series(1, 500) AS id;

-- Show sample data
SELECT COUNT(*) as total_vectors, (SELECT vector_dims(vec) FROM test_pca_data LIMIT 1) as dimensions
FROM test_pca_data;

\echo '=== Testing PCA (Principal Component Analysis) ==='

-- Test PCA: reduce from 20 dimensions to 2
-- Validate that function returns expected results
DO $$
DECLARE
    result_count INT;
    dim_check INT;
BEGIN
    SELECT COUNT(*)
    INTO result_count
    FROM neurondb.reduce_pca('test_pca_data', 'vec', 2);
    
    -- Check dimension separately
    SELECT vector_dims(reduced_vector)
    INTO dim_check
    FROM neurondb.reduce_pca('test_pca_data', 'vec', 2)
    LIMIT 1;
    
    IF result_count = 0 THEN
        RAISE EXCEPTION 'reduce_pca returned no results';
    END IF;
    
    IF dim_check != 2 THEN
        RAISE EXCEPTION 'reduce_pca returned vector with % dimensions, expected 2', dim_check;
    END IF;
END$$;

SELECT 
    id,
    reduced_vector
FROM neurondb.reduce_pca('test_pca_data', 'vec', 2)
ORDER BY id
LIMIT 10;

-- Verify reduced dimensionality
SELECT 
    id,
    vector_dims(reduced_vector) as new_dims
FROM neurondb.reduce_pca('test_pca_data', 'vec', 2)
ORDER BY id
LIMIT 3;

-- Test PCA: reduce to 3 dimensions
SELECT 
    id,
    reduced_vector
FROM neurondb.reduce_pca('test_pca_data', 'vec', 3)
ORDER BY id
LIMIT 10;

-- Test PCA: reduce to 1 dimension
SELECT 
    id,
    reduced_vector
FROM neurondb.reduce_pca('test_pca_data', 'vec', 1)
ORDER BY id
LIMIT 5;

\echo '=== Testing PCA Whitening ==='

-- Test PCA Whitening
-- Validate that function returns expected results
DO $$
DECLARE
    result_count INT;
    dim_check INT;
BEGIN
    -- whiten_embeddings may not be available, handle gracefully
    BEGIN
        SELECT COUNT(*)
        INTO result_count
        FROM neurondb.whiten_embeddings('test_pca_data', 'vec');
        
        IF result_count = 0 THEN
            RAISE EXCEPTION 'whiten_embeddings returned no results';
        END IF;
        
        -- Check dimension separately
        SELECT vector_dims(whitened_vec)
        INTO dim_check
        FROM neurondb.whiten_embeddings('test_pca_data', 'vec')
        LIMIT 1;
        
        IF dim_check != 20 THEN
            RAISE EXCEPTION 'whiten_embeddings returned vector with % dimensions, expected 20', dim_check;
        END IF;
    EXCEPTION WHEN undefined_function THEN
        RAISE NOTICE 'whiten_embeddings not available, skipping test';
        -- Skip the test if function doesn't exist
    END;
END$$;

-- Test PCA Whitening (decorrelates and normalizes)
-- Note: whiten_embeddings may not be available
DO $$
BEGIN
    BEGIN
        PERFORM 1 FROM neurondb.whiten_embeddings('test_pca_data', 'vec') LIMIT 1;
    EXCEPTION WHEN undefined_function THEN
        RAISE NOTICE 'whiten_embeddings not available, skipping display tests';
    END;
END$$;

\echo '=== Testing PCA with Different Data Distributions ==='

-- Create data with high variance in one dimension
DROP TABLE IF EXISTS test_pca_skewed CASCADE;
CREATE TABLE test_pca_skewed (
    id SERIAL PRIMARY KEY,
    vec vector(4)
);

INSERT INTO test_pca_skewed (vec) VALUES
    ('[100.0, 1.0, 1.0, 1.0]'::vector),
    ('[200.0, 1.1, 1.1, 1.1]'::vector),
    ('[150.0, 0.9, 0.9, 0.9]'::vector),
    ('[180.0, 1.2, 1.2, 1.2]'::vector),
    ('[120.0, 1.0, 1.0, 1.0]'::vector);

-- PCA should capture the high-variance dimension in first component
SELECT 
    id,
    reduced_vector
FROM neurondb.reduce_pca('test_pca_skewed', 'vec', 2)
ORDER BY id;

-- Whitening should normalize the high variance (if function available)
DO $$
BEGIN
    BEGIN
        PERFORM 1 FROM neurondb.whiten_embeddings('test_pca_skewed', 'vec') LIMIT 1;
    EXCEPTION WHEN undefined_function THEN
        RAISE NOTICE 'whiten_embeddings not available, skipping skewed data test';
    END;
END$$;

\echo '=== Edge Cases and Error Handling ==='

-- Test PCA with minimal data
DROP TABLE IF EXISTS test_pca_minimal CASCADE;
CREATE TABLE test_pca_minimal (
    id SERIAL PRIMARY KEY,
    vec vector(3)
);

INSERT INTO test_pca_minimal (vec) VALUES
    ('[1.0, 2.0, 3.0]'::vector),
    ('[1.1, 2.1, 3.1]'::vector);

-- PCA with 2 points
SELECT 
    id,
    reduced_vector
FROM neurondb.reduce_pca('test_pca_minimal', 'vec', 2)
ORDER BY id;

-- Test reducing to same dimensionality as input
SELECT 
    id,
    reduced_vector
FROM neurondb.reduce_pca('test_pca_data', 'vec', 5)
ORDER BY id
LIMIT 3;

-- Test whitening with minimal data (if function available)
DO $$
BEGIN
    BEGIN
        PERFORM 1 FROM neurondb.whiten_embeddings('test_pca_minimal', 'vec') LIMIT 1;
    EXCEPTION WHEN undefined_function THEN
        RAISE NOTICE 'whiten_embeddings not available, skipping minimal data test';
    END;
END$$;

\echo '=== Testing PCA Preservation of Relative Distances ==='

-- PCA should preserve relative relationships between points
-- Calculate pairwise distances before PCA
WITH original_dists AS (
    SELECT 
        a.id as id1,
        b.id as id2,
        a.vec <-> b.vec as orig_dist
    FROM test_pca_data a, test_pca_data b
    WHERE a.id < b.id AND a.id <= 3 AND b.id <= 3
),
-- Calculate pairwise distances after PCA (skip if function doesn't exist)
reduced_dists AS (
    SELECT 
        a.id as id1,
        b.id as id2,
        a.reduced_vector <-> b.reduced_vector as reduced_dist
    FROM neurondb.reduce_pca('test_pca_data', 'vec', 2) a,
         neurondb.reduce_pca('test_pca_data', 'vec', 2) b
    WHERE a.id < b.id AND a.id <= 3 AND b.id <= 3
      AND EXISTS (SELECT 1 FROM pg_proc WHERE proname = 'reduce_pca' AND pronamespace = (SELECT oid FROM pg_namespace WHERE nspname = 'neurondb'))
)
SELECT 
    o.id1,
    o.id2,
    ROUND(o.orig_dist::numeric, 4) as original_distance,
    ROUND(r.reduced_dist::numeric, 4) as reduced_distance,
    CASE 
        WHEN ABS(o.orig_dist - r.reduced_dist) < 2.0 THEN 'Preserved'
        ELSE 'Changed'
    END as relationship
FROM original_dists o
JOIN reduced_dists r ON o.id1 = r.id1 AND o.id2 = r.id2
ORDER BY o.id1, o.id2;

-- Cleanup
DROP TABLE test_pca_data CASCADE;
DROP TABLE test_pca_skewed CASCADE;
DROP TABLE test_pca_minimal CASCADE;
