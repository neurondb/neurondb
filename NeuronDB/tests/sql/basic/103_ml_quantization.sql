-- ====================================================================
-- NeurondB Regression Tests: Vector Quantization
-- ====================================================================
-- Tests for Product Quantization (PQ) and Optimized PQ (OPQ)
-- Uses real data from: sift1m.vectors (128-d vectors perfect for PQ)
-- ====================================================================

\echo '=== Using SIFT1M Dataset for Quantization Tests ==='

-- Create test data from SIFT1M vectors (128-d vectors, perfect for PQ)
-- Take first 2000 vectors for reasonable training time
CREATE TEMP TABLE test_pq_data AS
SELECT 
    id,
    array_to_vector(embedding)::vector(128) as vec
FROM sift1m.vectors
WHERE id <= 2000
LIMIT 2000;

-- Show sample data
SELECT COUNT(*) as total_vectors, vector_dims(vec) as dimensions
FROM test_pq_data
LIMIT 1;

\echo '=== Testing Product Quantization (PQ) ==='

-- Train PQ codebook: 8-dim vectors, 2 subvectors (4 dims each), 4 centroids per subvector
SELECT 
    subvec_id,
    centroid_id,
    centroid
FROM neurondb.train_pq_codebook('test_pq_data', 'vec', 2, 4, 50)
ORDER BY subvec_id, centroid_id;

-- Verify codebook structure
SELECT 
    subvec_id,
    COUNT(*) as num_centroids,
    vector_dims(centroid) as centroid_dims
FROM neurondb.train_pq_codebook('test_pq_data', 'vec', 2, 4, 50)
GROUP BY subvec_id, vector_dims(centroid)
ORDER BY subvec_id;

\echo '=== Testing PQ Encoding ==='

-- Store codebook in a table for encoding
CREATE TEMP TABLE pq_codebook AS
SELECT * FROM neurondb.train_pq_codebook('test_pq_data', 'vec', 2, 4, 50);

-- Encode vectors using the trained codebook
SELECT 
    id,
    vec,
    neurondb.pq_encode_vector(vec, 2, 4, 
        (SELECT array_agg(centroid ORDER BY subvec_id, centroid_id) 
         FROM pq_codebook)) as pq_codes
FROM test_pq_data
ORDER BY id;

-- Verify encoding produces correct number of codes
SELECT 
    id,
    array_length(neurondb.pq_encode_vector(vec, 2, 4, 
        (SELECT array_agg(centroid ORDER BY subvec_id, centroid_id) 
         FROM pq_codebook)), 1) as num_codes
FROM test_pq_data
ORDER BY id
LIMIT 5;

\echo '=== Testing PQ Asymmetric Distance ==='

-- Test asymmetric distance calculation
-- Compare original vector with PQ-encoded vector
WITH encoded AS (
    SELECT 
        id,
        vec,
        neurondb.pq_encode_vector(vec, 2, 4, 
            (SELECT array_agg(centroid ORDER BY subvec_id, centroid_id) 
             FROM pq_codebook)) as pq_codes
    FROM test_pq_data
)
SELECT 
    e1.id as id1,
    e2.id as id2,
    ROUND(neurondb.pq_asymmetric_distance(
        e1.vec, 
        e2.pq_codes, 
        2, 
        4,
        (SELECT array_agg(centroid ORDER BY subvec_id, centroid_id) FROM pq_codebook)
    )::numeric, 4) as pq_dist,
    ROUND((e1.vec <-> e2.vec)::numeric, 4) as actual_dist
FROM encoded e1, encoded e2
WHERE e1.id < e2.id AND e1.id <= 3 AND e2.id <= 3
ORDER BY e1.id, e2.id;

\echo '=== Testing Optimized Product Quantization (OPQ) ==='

-- Train OPQ rotation matrix
SELECT 
    rotation_matrix
FROM neurondb.train_opq_rotation('test_pq_data', 'vec', 2, 4, 30)
LIMIT 1;

-- Verify rotation matrix dimensions (should be dim x dim)
SELECT 
    vector_dims(rotation_matrix) as matrix_dims
FROM neurondb.train_opq_rotation('test_pq_data', 'vec', 2, 4, 30)
LIMIT 1;

-- Apply OPQ rotation to vectors
WITH rotation AS (
    SELECT rotation_matrix 
    FROM neurondb.train_opq_rotation('test_pq_data', 'vec', 2, 4, 30)
    LIMIT 1
)
SELECT 
    t.id,
    t.vec as original,
    neurondb.apply_opq_rotation(t.vec, r.rotation_matrix) as rotated
FROM test_pq_data t, rotation r
ORDER BY t.id
LIMIT 5;

-- Verify rotated vectors have same dimensionality
WITH rotation AS (
    SELECT rotation_matrix 
    FROM neurondb.train_opq_rotation('test_pq_data', 'vec', 2, 4, 30)
    LIMIT 1
)
SELECT 
    t.id,
    vector_dims(t.vec) as original_dims,
    vector_dims(neurondb.apply_opq_rotation(t.vec, r.rotation_matrix)) as rotated_dims
FROM test_pq_data t, rotation r
ORDER BY t.id
LIMIT 3;

\echo '=== Testing PQ with Different Configurations ==='

-- Test with 4 subvectors (2 dims each)
SELECT 
    subvec_id,
    COUNT(*) as num_centroids
FROM neurondb.train_pq_codebook('test_pq_data', 'vec', 4, 4, 50)
GROUP BY subvec_id
ORDER BY subvec_id;

-- Test with more centroids per subvector
SELECT 
    subvec_id,
    COUNT(*) as num_centroids
FROM neurondb.train_pq_codebook('test_pq_data', 'vec', 2, 8, 50)
GROUP BY subvec_id
ORDER BY subvec_id;

\echo '=== Edge Cases and Error Handling ==='

-- Test PQ with minimal data
CREATE TABLE test_pq_minimal (
    id SERIAL PRIMARY KEY,
    vec vector(4)
);

INSERT INTO test_pq_minimal (vec) VALUES
    ('[1.0, 2.0, 3.0, 4.0]'::vector),
    ('[1.1, 2.1, 3.1, 4.1]'::vector),
    ('[2.0, 3.0, 4.0, 5.0]'::vector);

-- Train codebook with minimal data
SELECT 
    subvec_id,
    centroid_id,
    centroid
FROM neurondb.train_pq_codebook('test_pq_minimal', 'vec', 2, 2, 10)
ORDER BY subvec_id, centroid_id;

-- Test PQ with single subvector (entire vector)
SELECT 
    subvec_id,
    COUNT(*) as num_centroids
FROM neurondb.train_pq_codebook('test_pq_data', 'vec', 1, 4, 30)
GROUP BY subvec_id;

\echo '=== Testing PQ Compression Ratio ==='

-- Calculate storage savings from PQ encoding
WITH encoded AS (
    SELECT 
        id,
        vec,
        neurondb.pq_encode_vector(vec, 2, 4, 
            (SELECT array_agg(centroid ORDER BY subvec_id, centroid_id) 
             FROM pq_codebook)) as pq_codes
    FROM test_pq_data
)
SELECT 
    'Original Vector' as type,
    pg_column_size(vec) as bytes,
    COUNT(*) as num_vectors,
    pg_column_size(vec) * COUNT(*) as total_bytes
FROM test_pq_data
UNION ALL
SELECT 
    'PQ Codes' as type,
    pg_column_size(pq_codes) as bytes,
    COUNT(*) as num_vectors,
    pg_column_size(pq_codes) * COUNT(*) as total_bytes
FROM encoded
LIMIT 1;

-- Cleanup
DROP TABLE test_pq_data CASCADE;
DROP TABLE test_pq_minimal CASCADE;

