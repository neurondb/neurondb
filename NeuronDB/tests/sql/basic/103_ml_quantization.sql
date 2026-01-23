-- ====================================================================
-- NeurondB Regression Tests: Vector Quantization
-- ====================================================================
-- Tests for Product Quantization (PQ) and Optimized PQ (OPQ)
-- Uses real data from: sift1m.vectors (128-d vectors perfect for PQ)
-- ====================================================================

\echo '=== Using SIFT1M Dataset for Quantization Tests ==='

-- Create test data with synthetic vectors (sift1m.vectors may not exist)
-- 128-d vectors, perfect for PQ
CREATE TEMP TABLE test_pq_data AS
SELECT 
    id,
    array_to_vector(ARRAY(SELECT random()::real FROM generate_series(1, 128)))::vector(128) as vec
FROM generate_series(1, 2000) AS id;

-- Show sample data

SELECT COUNT(*) as total_vectors, (SELECT vector_dims(vec) FROM test_pq_data LIMIT 1) as dimensions
FROM test_pq_data;

\echo '=== Testing Product Quantization (PQ) ==='

-- Train PQ codebook: 8-dim vectors, 2 subvectors (4 dims each), 4 centroids per subvector
-- Validate that function returns expected results
DO $$
DECLARE
    result_count INT;
    subvec_count INT;
BEGIN
    SELECT COUNT(*), COUNT(DISTINCT subvec_id)
    INTO result_count, subvec_count
    FROM neurondb.train_pq_codebook('test_pq_data', 'vec', 2, 4, 50);
    
    IF result_count = 0 THEN
        RAISE EXCEPTION 'train_pq_codebook returned no results';
    END IF;
    
    IF subvec_count != 2 THEN
        RAISE EXCEPTION 'train_pq_codebook returned % subvectors, expected 2', subvec_count;
    END IF;
END$$;

SELECT 
    subvec_id,
    centroid_id,
    centroid
FROM neurondb.train_pq_codebook('test_pq_data', 'vec', 2, 4, 50)
ORDER BY subvec_id, centroid_id
LIMIT 20;

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
-- Validate that function returns expected results
DO $$
DECLARE
    code_count INT;
BEGIN
    SELECT array_length(neurondb.pq_encode_vector(
        (SELECT vec FROM test_pq_data LIMIT 1), 
        2, 4, 
        (SELECT array_agg(centroid ORDER BY subvec_id, centroid_id) FROM pq_codebook)
    ), 1) INTO code_count;
    
    IF code_count IS NULL OR code_count = 0 THEN
        RAISE EXCEPTION 'pq_encode_vector returned invalid codes';
    END IF;
    
    IF code_count != 2 THEN
        RAISE EXCEPTION 'pq_encode_vector returned % codes, expected 2', code_count;
    END IF;
END$$;

SELECT 
    id,
    vec,
    neurondb.pq_encode_vector(vec, 2, 4, 
        (SELECT array_agg(centroid ORDER BY subvec_id, centroid_id) 
         FROM pq_codebook)) as pq_codes
FROM test_pq_data
ORDER BY id
LIMIT 10;

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
-- Validate that function returns expected results
DO $$
DECLARE
    dist_result REAL;
BEGIN
    WITH encoded AS (
        SELECT neurondb.pq_encode_vector(
            (SELECT vec FROM test_pq_data LIMIT 1), 
            2, 4, 
            (SELECT array_agg(centroid ORDER BY subvec_id, centroid_id) FROM pq_codebook)
        ) as pq_codes
    )
    SELECT neurondb.pq_asymmetric_distance(
        (SELECT vec FROM test_pq_data LIMIT 1 OFFSET 1),
        (SELECT pq_codes FROM encoded),
        2, 4,
        (SELECT array_agg(centroid ORDER BY subvec_id, centroid_id) FROM pq_codebook)
    ) INTO dist_result;
    
    IF dist_result IS NULL THEN
        RAISE EXCEPTION 'pq_asymmetric_distance returned NULL';
    END IF;
    
    IF dist_result < 0 THEN
        RAISE EXCEPTION 'pq_asymmetric_distance returned negative distance: %', dist_result;
    END IF;
END$$;

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

-- Test OPQ rotation
-- Validate that function returns expected results
DO $$
DECLARE
    result_count INT;
    dim_check INT;
BEGIN
    SELECT COUNT(*), vector_dims(rotation_matrix)
    INTO result_count, dim_check
    FROM neurondb.train_opq_rotation('test_pq_data', 'vec', 2, 4, 30);
    
    IF result_count = 0 THEN
        RAISE EXCEPTION 'train_opq_rotation returned no results';
    END IF;
    
    IF dim_check != 128 THEN
        RAISE EXCEPTION 'train_opq_rotation returned matrix with % dimensions, expected 128', dim_check;
    END IF;
END$$;

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

-- Test OPQ rotation application
-- Validate that function returns expected results
DO $$
DECLARE
    rotated_dims INT;
BEGIN
    WITH rotation AS (
        SELECT rotation_matrix 
        FROM neurondb.train_opq_rotation('test_pq_data', 'vec', 2, 4, 30)
        LIMIT 1
    )
    SELECT vector_dims(neurondb.apply_opq_rotation(
        (SELECT vec FROM test_pq_data LIMIT 1),
        (SELECT rotation_matrix FROM rotation)
    )) INTO rotated_dims;
    
    IF rotated_dims IS NULL THEN
        RAISE EXCEPTION 'apply_opq_rotation returned NULL';
    END IF;
    
    IF rotated_dims != 128 THEN
        RAISE EXCEPTION 'apply_opq_rotation returned vector with % dimensions, expected 128', rotated_dims;
    END IF;
END$$;

-- Apply OPQ rotation to vectors (only if functions exist)
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
  AND EXISTS (SELECT 1 FROM rotation)
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
  AND EXISTS (SELECT 1 FROM rotation)
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

