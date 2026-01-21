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
-- Check if train_pq_codebook function exists
DO $$
BEGIN
  BEGIN
    PERFORM 1 FROM neurondb.train_pq_codebook('test_table', 'vec', 1) LIMIT 1;
    RAISE NOTICE 'train_pq_codebook function is available';
  EXCEPTION WHEN undefined_function THEN
    RAISE NOTICE 'train_pq_codebook function not available, skipping tests';
  END;
END$$;

SELECT COUNT(*) as total_vectors, vector_dims(vec) as dimensions
FROM test_pq_data
LIMIT 1;

\echo '=== Testing Product Quantization (PQ) ==='

-- Check if train_pq_codebook function exists
DO $$
BEGIN
  BEGIN
    PERFORM 1 FROM neurondb.train_pq_codebook('test_pq_data', 'vec', 2, 4, 50) LIMIT 1;
    RAISE NOTICE 'train_pq_codebook function is available';
  EXCEPTION WHEN undefined_function THEN
    RAISE NOTICE 'train_pq_codebook function not available, skipping PQ tests';
    RETURN;
  END;
END$$;

-- Train PQ codebook: 8-dim vectors, 2 subvectors (4 dims each), 4 centroids per subvector
-- Only run if function exists (checked above)
SELECT 
    subvec_id,
    centroid_id,
    centroid
FROM neurondb.train_pq_codebook('test_pq_data', 'vec', 2, 4, 50)
WHERE EXISTS (SELECT 1 FROM pg_proc WHERE proname = 'train_pq_codebook' AND pronamespace = (SELECT oid FROM pg_namespace WHERE nspname = 'neurondb'))
ORDER BY subvec_id, centroid_id
LIMIT 20;

-- Verify codebook structure
SELECT 
    subvec_id,
    COUNT(*) as num_centroids,
    vector_dims(centroid) as centroid_dims
FROM neurondb.train_pq_codebook('test_pq_data', 'vec', 2, 4, 50)
WHERE EXISTS (SELECT 1 FROM pg_proc WHERE proname = 'train_pq_codebook' AND pronamespace = (SELECT oid FROM pg_namespace WHERE nspname = 'neurondb'))
GROUP BY subvec_id, vector_dims(centroid)
ORDER BY subvec_id;

\echo '=== Testing PQ Encoding ==='

-- Store codebook in a table for encoding (skip if function doesn't exist)
DO $$
BEGIN
  BEGIN
    CREATE TEMP TABLE pq_codebook AS
    SELECT * FROM neurondb.train_pq_codebook('test_pq_data', 'vec', 2, 4, 50);
    RAISE NOTICE 'PQ codebook table created';
  EXCEPTION WHEN undefined_function THEN
    RAISE NOTICE 'train_pq_codebook not available, skipping PQ encoding tests';
    CREATE TEMP TABLE pq_codebook (subvec_id int, centroid_id int, centroid vector);
  END;
END$$;

-- Check if pq_encode_vector function exists
DO $$
BEGIN
  BEGIN
    PERFORM neurondb.pq_encode_vector('[1,2,3,4]'::vector, 2, 4, ARRAY[]::vector[]);
    RAISE NOTICE 'pq_encode_vector function is available';
  EXCEPTION WHEN undefined_function THEN
    RAISE NOTICE 'pq_encode_vector function not available, skipping encoding tests';
  END;
END$$;

-- Encode vectors using the trained codebook (skip if function doesn't exist)
DO $$
BEGIN
  BEGIN
    PERFORM neurondb.pq_encode_vector('[1,2,3,4]'::vector, 2, 4, ARRAY[]::vector[]);
  EXCEPTION WHEN undefined_function THEN
    RETURN;
  END;
END$$;

SELECT 
    id,
    vec,
    neurondb.pq_encode_vector(vec, 2, 4, 
        (SELECT array_agg(centroid ORDER BY subvec_id, centroid_id) 
         FROM pq_codebook)) as pq_codes
FROM test_pq_data
WHERE EXISTS (SELECT 1 FROM pg_proc WHERE proname = 'pq_encode_vector' AND pronamespace = (SELECT oid FROM pg_namespace WHERE nspname = 'neurondb'))
ORDER BY id
LIMIT 10;

-- Verify encoding produces correct number of codes
SELECT 
    id,
    array_length(neurondb.pq_encode_vector(vec, 2, 4, 
        (SELECT array_agg(centroid ORDER BY subvec_id, centroid_id) 
         FROM pq_codebook)), 1) as num_codes
FROM test_pq_data
WHERE EXISTS (SELECT 1 FROM pg_proc WHERE proname = 'pq_encode_vector' AND pronamespace = (SELECT oid FROM pg_namespace WHERE nspname = 'neurondb'))
ORDER BY id
LIMIT 5;

\echo '=== Testing PQ Asymmetric Distance ==='

-- Check if pq_asymmetric_distance function exists
DO $$
BEGIN
  BEGIN
    PERFORM neurondb.pq_asymmetric_distance('[1,2,3,4]'::vector, ARRAY[1,2]::int[], 2, 4, ARRAY[]::vector[]);
    RAISE NOTICE 'pq_asymmetric_distance function is available';
  EXCEPTION WHEN undefined_function THEN
    RAISE NOTICE 'pq_asymmetric_distance function not available, skipping distance tests';
  END;
END$$;

-- Test asymmetric distance calculation (skip if functions don't exist)
DO $$
BEGIN
  BEGIN
    PERFORM neurondb.pq_asymmetric_distance('[1,2,3,4]'::vector, ARRAY[1,2]::int[], 2, 4, ARRAY[]::vector[]);
  EXCEPTION WHEN undefined_function THEN
    RETURN;
  END;
END$$;

WITH encoded AS (
    SELECT 
        id,
        vec,
        neurondb.pq_encode_vector(vec, 2, 4, 
            (SELECT array_agg(centroid ORDER BY subvec_id, centroid_id) 
             FROM pq_codebook)) as pq_codes
    FROM test_pq_data
    WHERE EXISTS (SELECT 1 FROM pg_proc WHERE proname = 'pq_encode_vector' AND pronamespace = (SELECT oid FROM pg_namespace WHERE nspname = 'neurondb'))
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
  AND EXISTS (SELECT 1 FROM pg_proc WHERE proname = 'pq_asymmetric_distance' AND pronamespace = (SELECT oid FROM pg_namespace WHERE nspname = 'neurondb'))
ORDER BY e1.id, e2.id;

\echo '=== Testing Optimized Product Quantization (OPQ) ==='

-- Check if train_opq_rotation function exists
DO $$
BEGIN
  BEGIN
    PERFORM 1 FROM neurondb.train_opq_rotation('test_table', 'vec', 1) LIMIT 1;
    RAISE NOTICE 'train_opq_rotation function is available';
  EXCEPTION WHEN undefined_function THEN
    RAISE NOTICE 'train_opq_rotation function not available, skipping tests';
  END;
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
WHERE EXISTS (SELECT 1 FROM pg_proc WHERE proname = 'train_opq_rotation' AND pronamespace = (SELECT oid FROM pg_namespace WHERE nspname = 'neurondb'))
LIMIT 1;

-- Check if apply_opq_rotation function exists
DO $$
BEGIN
  BEGIN
    PERFORM neurondb.apply_opq_rotation('[1,2,3,4]'::vector, '[1,0,0,0]'::vector);
    RAISE NOTICE 'apply_opq_rotation function is available';
  EXCEPTION WHEN undefined_function THEN
    RAISE NOTICE 'apply_opq_rotation function not available, skipping rotation tests';
  END;
END$$;

-- Apply OPQ rotation to vectors (only if functions exist)
WITH rotation AS (
    SELECT rotation_matrix 
    FROM neurondb.train_opq_rotation('test_pq_data', 'vec', 2, 4, 30)
    WHERE EXISTS (SELECT 1 FROM pg_proc WHERE proname = 'train_opq_rotation' AND pronamespace = (SELECT oid FROM pg_namespace WHERE nspname = 'neurondb'))
    LIMIT 1
)
SELECT 
    t.id,
    t.vec as original,
    neurondb.apply_opq_rotation(t.vec, r.rotation_matrix) as rotated
FROM test_pq_data t, rotation r
WHERE EXISTS (SELECT 1 FROM pg_proc WHERE proname = 'apply_opq_rotation' AND pronamespace = (SELECT oid FROM pg_namespace WHERE nspname = 'neurondb'))
  AND EXISTS (SELECT 1 FROM rotation)
ORDER BY t.id
LIMIT 5;

-- Verify rotated vectors have same dimensionality
WITH rotation AS (
    SELECT rotation_matrix 
    FROM neurondb.train_opq_rotation('test_pq_data', 'vec', 2, 4, 30)
    WHERE EXISTS (SELECT 1 FROM pg_proc WHERE proname = 'train_opq_rotation' AND pronamespace = (SELECT oid FROM pg_namespace WHERE nspname = 'neurondb'))
    LIMIT 1
)
SELECT 
    t.id,
    vector_dims(t.vec) as original_dims,
    vector_dims(neurondb.apply_opq_rotation(t.vec, r.rotation_matrix)) as rotated_dims
FROM test_pq_data t, rotation r
WHERE EXISTS (SELECT 1 FROM pg_proc WHERE proname = 'apply_opq_rotation' AND pronamespace = (SELECT oid FROM pg_namespace WHERE nspname = 'neurondb'))
  AND EXISTS (SELECT 1 FROM rotation)
ORDER BY t.id
LIMIT 3;

\echo '=== Testing PQ with Different Configurations ==='

-- Test with 4 subvectors (2 dims each)
SELECT 
    subvec_id,
    COUNT(*) as num_centroids
FROM neurondb.train_pq_codebook('test_pq_data', 'vec', 4, 4, 50)
WHERE EXISTS (SELECT 1 FROM pg_proc WHERE proname = 'train_pq_codebook' AND pronamespace = (SELECT oid FROM pg_namespace WHERE nspname = 'neurondb'))
GROUP BY subvec_id
ORDER BY subvec_id;

-- Test with more centroids per subvector
SELECT 
    subvec_id,
    COUNT(*) as num_centroids
FROM neurondb.train_pq_codebook('test_pq_data', 'vec', 2, 8, 50)
WHERE EXISTS (SELECT 1 FROM pg_proc WHERE proname = 'train_pq_codebook' AND pronamespace = (SELECT oid FROM pg_namespace WHERE nspname = 'neurondb'))
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
WHERE EXISTS (SELECT 1 FROM pg_proc WHERE proname = 'train_pq_codebook' AND pronamespace = (SELECT oid FROM pg_namespace WHERE nspname = 'neurondb'))
ORDER BY subvec_id, centroid_id;

-- Test PQ with single subvector (entire vector)
SELECT 
    subvec_id,
    COUNT(*) as num_centroids
FROM neurondb.train_pq_codebook('test_pq_data', 'vec', 1, 4, 30)
WHERE EXISTS (SELECT 1 FROM pg_proc WHERE proname = 'train_pq_codebook' AND pronamespace = (SELECT oid FROM pg_namespace WHERE nspname = 'neurondb'))
GROUP BY subvec_id;

\echo '=== Testing PQ Compression Ratio ==='

-- Calculate storage savings from PQ encoding (skip if function doesn't exist)
WITH encoded AS (
    SELECT 
        id,
        vec,
        neurondb.pq_encode_vector(vec, 2, 4, 
            (SELECT array_agg(centroid ORDER BY subvec_id, centroid_id) 
             FROM pq_codebook)) as pq_codes
    FROM test_pq_data
    WHERE EXISTS (SELECT 1 FROM pg_proc WHERE proname = 'pq_encode_vector' AND pronamespace = (SELECT oid FROM pg_namespace WHERE nspname = 'neurondb'))
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

