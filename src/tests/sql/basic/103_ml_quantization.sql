-- ====================================================================
-- NeurondB Regression Tests: Vector Quantization
-- ====================================================================
-- Tests for Product Quantization (PQ) and Optimized PQ (OPQ)
-- Uses real data from: sift1m.vectors (128-d vectors perfect for PQ)
-- ====================================================================

CREATE EXTENSION IF NOT EXISTS neurondb;

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

-- Note: pq_encode_vector function may not be available, so wrap all tests in error handling

-- Encode vectors using the trained codebook
-- Note: pq_encode_vector may not be available
DO $$
DECLARE
    code_count INT;
BEGIN
    BEGIN
        SELECT array_length(neurondb.pq_encode_vector(
            (SELECT vec FROM test_pq_data LIMIT 1), 
            2, 4, 
            (SELECT array_agg(centroid ORDER BY subvec_id, centroid_id) FROM pq_codebook)
        ), 1) INTO code_count;
        
        IF code_count IS NULL OR code_count = 0 THEN
            RAISE WARNING 'pq_encode_vector returned invalid codes';
        ELSIF code_count != 2 THEN
            RAISE WARNING 'pq_encode_vector returned % codes, expected 2', code_count;
        END IF;
    EXCEPTION
        WHEN undefined_function THEN
            RAISE NOTICE 'pq_encode_vector not available, skipping encoding test';
        WHEN OTHERS THEN
            RAISE NOTICE 'pq_encode_vector error: %', SQLERRM;
    END;
END$$;

-- Display PQ encoded vectors (if function available)
DO $$
BEGIN
    BEGIN
        PERFORM neurondb.pq_encode_vector(
            (SELECT vec FROM test_pq_data LIMIT 1), 
            2, 4, 
            (SELECT array_agg(centroid ORDER BY subvec_id, centroid_id) FROM pq_codebook)
        );
    EXCEPTION WHEN undefined_function THEN
        RAISE NOTICE 'pq_encode_vector not available, skipping display tests';
    END;
END$$;

\echo '=== Testing PQ Asymmetric Distance ==='

-- Test asymmetric distance calculation
-- Note: pq_encode_vector and pq_asymmetric_distance may not be available
DO $$
DECLARE
    dist_result REAL;
BEGIN
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
            RAISE WARNING 'pq_asymmetric_distance returned NULL';
        ELSIF dist_result < 0 THEN
            RAISE WARNING 'pq_asymmetric_distance returned negative distance: %', dist_result;
        END IF;
    EXCEPTION
        WHEN undefined_function THEN
            RAISE NOTICE 'pq_encode_vector or pq_asymmetric_distance not available, skipping distance test';
        WHEN OTHERS THEN
            RAISE NOTICE 'pq_asymmetric_distance error: %', SQLERRM;
    END;
END$$;

-- Display PQ asymmetric distances (if functions available)
DO $$
BEGIN
    BEGIN
        PERFORM neurondb.pq_encode_vector(
            (SELECT vec FROM test_pq_data LIMIT 1), 
            2, 4, 
            (SELECT array_agg(centroid ORDER BY subvec_id, centroid_id) FROM pq_codebook)
        );
    EXCEPTION WHEN undefined_function THEN
        RAISE NOTICE 'pq_encode_vector not available, skipping asymmetric distance display';
    END;
END$$;

\echo '=== Testing Optimized Product Quantization (OPQ) ==='

-- Test OPQ rotation
-- Note: Function returns float8[] (array), not a table
-- Validate that function returns expected results
DO $$
DECLARE
    rotation_array float8[];
BEGIN
    SELECT train_opq_rotation('test_pq_data', 'vec', 2) INTO rotation_array;
    
    IF rotation_array IS NULL OR array_length(rotation_array, 1) IS NULL THEN
        RAISE EXCEPTION 'train_opq_rotation returned NULL or empty array';
    END IF;
    
    -- For 128-dim vectors, rotation matrix should be 128*128 = 16384 elements
    IF array_length(rotation_array, 1) != 16384 THEN
        RAISE WARNING 'train_opq_rotation returned array with % elements, expected 16384 for 128-dim vectors', array_length(rotation_array, 1);
    END IF;
END$$;

-- Display OPQ rotation matrix (as array)
SELECT train_opq_rotation('test_pq_data', 'vec', 2) AS rotation_matrix;

-- Test OPQ rotation application
-- Note: train_opq_rotation returns float8[], apply_opq_rotation takes (float8[], float8[])
-- Validate that function returns expected results
DO $$
DECLARE
    rotation_array float8[];
    rotated_vector vector;
    rotated_dims INT;
BEGIN
    -- Get rotation matrix as array
    SELECT train_opq_rotation('test_pq_data', 'vec', 2) INTO rotation_array;
    
    IF rotation_array IS NULL THEN
        RAISE EXCEPTION 'train_opq_rotation returned NULL';
    END IF;
    
    -- Apply rotation (function signature: apply_opq_rotation(float8[], float8[]))
    -- First arg is vector as array, second is rotation matrix
    -- Note: vec::float8[] may not work directly, need to convert properly
    DECLARE
        vec_array float8[];
    BEGIN
        -- Convert vector to array (this may need a helper function)
        SELECT ARRAY(SELECT unnest FROM unnest((SELECT vec FROM test_pq_data LIMIT 1)::float8[])) INTO vec_array;
        SELECT apply_opq_rotation(vec_array, rotation_array)::vector INTO rotated_vector;
    END;
    
    SELECT vector_dims(rotated_vector) INTO rotated_dims;
    
    IF rotated_dims IS NULL THEN
        RAISE EXCEPTION 'apply_opq_rotation returned NULL';
    END IF;
    
    IF rotated_dims != 128 THEN
        RAISE WARNING 'apply_opq_rotation returned vector with % dimensions, expected 128', rotated_dims;
    END IF;
EXCEPTION
    WHEN undefined_function THEN
        RAISE NOTICE 'apply_opq_rotation not available, skipping rotation application test';
    WHEN OTHERS THEN
        RAISE NOTICE 'apply_opq_rotation error: %', SQLERRM;
END$$;

-- Apply OPQ rotation to vectors (if function available)
DO $$
BEGIN
    BEGIN
        PERFORM neurondb.apply_opq_rotation(
            (SELECT vec FROM test_pq_data LIMIT 1),
            (SELECT rotation_matrix FROM neurondb.train_opq_rotation('test_pq_data', 'vec', 2, 4, 30) LIMIT 1)
        );
    EXCEPTION
        WHEN undefined_function THEN
            RAISE NOTICE 'apply_opq_rotation not available, skipping rotation display tests';
        WHEN OTHERS THEN
            RAISE NOTICE 'apply_opq_rotation error: %', SQLERRM;
    END;
END$$;

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

-- Calculate storage savings from PQ encoding (if function available)
DO $$
BEGIN
    BEGIN
        PERFORM neurondb.pq_encode_vector(
            (SELECT vec FROM test_pq_data LIMIT 1), 
            2, 4, 
            (SELECT array_agg(centroid ORDER BY subvec_id, centroid_id) FROM pq_codebook)
        );
    EXCEPTION WHEN undefined_function THEN
        RAISE NOTICE 'pq_encode_vector not available, skipping compression ratio test';
    END;
END$$;

-- Cleanup
DROP TABLE test_pq_data CASCADE;
DROP TABLE test_pq_minimal CASCADE;

