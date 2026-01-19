-- ====================================================================
-- NeurondB Regression Tests: Dimensionality Reduction
-- ====================================================================
-- Tests for PCA and PCA Whitening
-- Uses real data from: deep1b.vectors (96-d vectors)
-- ====================================================================

\echo '=== Using Deep1B Dataset for PCA Tests ==='

-- Create test data from Deep1B vectors (take first 500 for speed)
CREATE TEMP TABLE test_pca_data AS
SELECT 
    id,
    array_to_vector(embedding[1:20])::vector(20) as vec
FROM deep1b.vectors
WHERE id <= 500
LIMIT 500;

-- Show sample data
SELECT COUNT(*) as total_vectors, vector_dims(vec) as dimensions
FROM test_pca_data
LIMIT 1;

\echo '=== Testing PCA (Principal Component Analysis) ==='

-- Test PCA: reduce from 5 dimensions to 2
SELECT 
    id,
    reduced_vec
FROM neurondb.reduce_pca('test_pca_data', 'vec', 2)
ORDER BY id;

-- Verify reduced dimensionality
SELECT 
    id,
    vector_dims(reduced_vec) as new_dims
FROM neurondb.reduce_pca('test_pca_data', 'vec', 2)
ORDER BY id
LIMIT 3;

-- Test PCA: reduce to 3 dimensions
SELECT 
    id,
    reduced_vec
FROM neurondb.reduce_pca('test_pca_data', 'vec', 3)
ORDER BY id;

-- Test PCA: reduce to 1 dimension
SELECT 
    id,
    reduced_vec
FROM neurondb.reduce_pca('test_pca_data', 'vec', 1)
ORDER BY id
LIMIT 5;

\echo '=== Testing PCA Whitening ==='

-- Test PCA Whitening (decorrelates and normalizes)
SELECT 
    id,
    whitened_vec
FROM neurondb.whiten_embeddings('test_pca_data', 'vec')
ORDER BY id;

-- Verify whitened vectors have same dimensionality
SELECT 
    id,
    vector_dims(whitened_vec) as dims
FROM neurondb.whiten_embeddings('test_pca_data', 'vec')
ORDER BY id
LIMIT 3;

-- Verify whitening produces different vectors than original
SELECT 
    t.id,
    CASE 
        WHEN t.vec = w.whitened_vec THEN 'Same'
        ELSE 'Different'
    END as vec_changed
FROM test_pca_data t
JOIN neurondb.whiten_embeddings('test_pca_data', 'vec') w ON t.id = w.id
ORDER BY t.id
LIMIT 5;

\echo '=== Testing PCA with Different Data Distributions ==='

-- Create data with high variance in one dimension
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
    reduced_vec
FROM neurondb.reduce_pca('test_pca_skewed', 'vec', 2)
ORDER BY id;

-- Whitening should normalize the high variance
SELECT 
    id,
    whitened_vec
FROM neurondb.whiten_embeddings('test_pca_skewed', 'vec')
ORDER BY id;

\echo '=== Edge Cases and Error Handling ==='

-- Test PCA with minimal data
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
    reduced_vec
FROM neurondb.reduce_pca('test_pca_minimal', 'vec', 2)
ORDER BY id;

-- Test reducing to same dimensionality as input
SELECT 
    id,
    reduced_vec
FROM neurondb.reduce_pca('test_pca_data', 'vec', 5)
ORDER BY id
LIMIT 3;

-- Test whitening with minimal data
SELECT 
    id,
    whitened_vec
FROM neurondb.whiten_embeddings('test_pca_minimal', 'vec')
ORDER BY id;

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
-- Calculate pairwise distances after PCA
reduced_dists AS (
    SELECT 
        a.id as id1,
        b.id as id2,
        a.reduced_vec <-> b.reduced_vec as reduced_dist
    FROM neurondb.reduce_pca('test_pca_data', 'vec', 2) a,
         neurondb.reduce_pca('test_pca_data', 'vec', 2) b
    WHERE a.id < b.id AND a.id <= 3 AND b.id <= 3
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

