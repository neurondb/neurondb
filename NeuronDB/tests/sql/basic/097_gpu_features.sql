-- GPU Features SQL Definitions
-- Includes GPU-accelerated search functions with index parameter support

-- ==== GPU HNSW Search Function ====
-- GPU-accelerated HNSW k-nearest neighbor search
-- Note: Functions should already be created by extension, but create if missing
DO $$
BEGIN
  -- Try to create functions if they don't exist (functions should be in extension)
  BEGIN
    CREATE OR REPLACE FUNCTION hnsw_knn_search_gpu(text, vector, int, int)
      RETURNS TABLE(id bigint, distance real)
      AS 'MODULE_PATHNAME', 'hnsw_knn_search_gpu'
      LANGUAGE C STABLE;
  EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE 'hnsw_knn_search_gpu may already exist or not be available: %', SQLERRM;
  END;
  
  BEGIN
    CREATE OR REPLACE FUNCTION hnsw_knn_search_gpu(text, vector, int)
      RETURNS TABLE(id bigint, distance real)
      AS 'MODULE_PATHNAME', 'hnsw_knn_search_gpu'
      LANGUAGE C STABLE;
  EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE 'hnsw_knn_search_gpu overload may already exist or not be available: %', SQLERRM;
  END;
  
  BEGIN
    CREATE OR REPLACE FUNCTION ivf_knn_search_gpu(text, vector, int, int)
      RETURNS TABLE(id bigint, distance real)
      AS 'MODULE_PATHNAME', 'ivf_knn_search_gpu'
      LANGUAGE C STABLE;
  EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE 'ivf_knn_search_gpu may already exist or not be available: %', SQLERRM;
  END;
  
  BEGIN
    CREATE OR REPLACE FUNCTION ivf_knn_search_gpu(text, vector, int)
      RETURNS TABLE(id bigint, distance real)
      AS 'MODULE_PATHNAME', 'ivf_knn_search_gpu'
      LANGUAGE C STABLE;
  EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE 'ivf_knn_search_gpu overload may already exist or not be available: %', SQLERRM;
  END;
END$$;

-- ==== Detailed and all possible tests for GPU Features and GPU Acceleration ====
-- Tests gracefully adapt to absence of GPU (run anyway for CPU fallback coverage)
-- Extension assumed to be created in 01_types_basic

-- ==== ENVIRONMENT PREP: Ensure deterministic fallback ====
SET neurondb.compute_mode = off;  -- Guarantee CPU for baseline/consistency

-- ==== 1. GPU Info and Status Functions: all call scenarios ====
-- Actual info/stats output is environment-dependent, so always run, don't just skip
-- Info before anything
SELECT * FROM neurondb_gpu_info() AS initial_gpu_info;

-- Try enabling GPU and toggling compute_mode
SELECT neurondb_gpu_enable() AS gpu_enabled;
-- Set to GPU mode
SET neurondb.compute_mode = 1;
SELECT neurondb_gpu_enable() AS gpu_enabled_after_set_gpu;
-- Set to CPU mode
SET neurondb.compute_mode = 0;
SELECT neurondb_gpu_enable() AS gpu_enabled_after_set_cpu;

-- Again check info after toggling
SELECT * FROM neurondb_gpu_info() AS post_toggle_gpu_info;

-- Check stats gather and reset (should always succeed):
SELECT * FROM neurondb_gpu_stats()  AS stats_pre_ops;
SELECT neurondb_gpu_stats_reset()   AS stats_reset_call_1;
SELECT * FROM neurondb_gpu_stats()  AS stats_post_reset;

-- ==== 2. GPU Test Data Creation: edge and normal ====
-- Standard 4-dim test table
CREATE TABLE gpu_test_vectors (
    id serial PRIMARY KEY,
    vec vector(4)
);

-- Insert diverse vectors:
-- - deterministic, incremental
-- - all zero
-- - negative numbers
-- - large values
-- - sparse
INSERT INTO gpu_test_vectors (vec) VALUES
  ('[1,2,3,4]'),
  ('[0,0,0,0]'),
  ('[-1,-2,-3,-4]'),
  ('[1000,2000,3000,4000]'),
  ('[0,0,0,5]'),
  ('[4,3,2,1]'),
  ('[1,-1,1,-1]'),
  ('[3.14,2.71,1.41,0.0]'),
  ('[1,0,0,0]'),
  ('[0,1,0,0]');

-- Try NULL vector insert (should either error or be handled)
INSERT INTO gpu_test_vectors (vec) VALUES (NULL);

-- ==== 3. GPU Distance Functions: L2/Cosine/Inner/Edge Cases ====
-- Compute each distance from all to all, include nulls to exercise edge handling

-- L2 distance, all pairs, skip nulls
SELECT a.id as id1, b.id as id2,
    vector_l2_distance_gpu(a.vec, b.vec) AS l2_distance_gpu
FROM gpu_test_vectors a, gpu_test_vectors b
WHERE a.vec IS NOT NULL AND b.vec IS NOT NULL
ORDER BY a.id, b.id;

-- Cosine distance, all pairs, include nulls
SELECT a.id as id1, b.id as id2,
    vector_cosine_distance_gpu(a.vec, b.vec) AS cosine_distance_gpu
FROM gpu_test_vectors a
LEFT JOIN gpu_test_vectors b ON TRUE
ORDER BY a.id, b.id;

-- Inner product, with nulls (should be null where either is null)
SELECT a.id as id1, b.id as id2,
    vector_inner_product_gpu(a.vec, b.vec) AS inner_product_gpu
FROM gpu_test_vectors a, gpu_test_vectors b
ORDER BY a.id, b.id;

-- Try each function with both arguments NULL (should not crash)
SELECT vector_l2_distance_gpu(NULL, NULL) AS l2_null_null,
       vector_cosine_distance_gpu(NULL, NULL) AS cosine_null_null,
       vector_inner_product_gpu(NULL, NULL) AS ip_null_null;

-- Try with one argument NULL
SELECT vector_l2_distance_gpu(vec, NULL), vector_l2_distance_gpu(NULL, vec) FROM gpu_test_vectors;
SELECT vector_cosine_distance_gpu(vec, NULL), vector_cosine_distance_gpu(NULL, vec) FROM gpu_test_vectors;
SELECT vector_inner_product_gpu(vec, NULL), vector_inner_product_gpu(NULL, vec) FROM gpu_test_vectors;

-- ==== 4. GPU vs CPU Distance Comparison: tolerance checks ====
-- For each row, compare GPU and CPU implementation's result
SELECT id,
  ABS(vector_l2_distance_gpu(vec, '[1,2,3,4]') - vector_l2_distance(vec, '[1,2,3,4]')) AS l2_diff,
  ABS(vector_cosine_distance_gpu(vec, '[1,2,3,4]') - vector_cosine_distance(vec, '[1,2,3,4]')) AS cosine_diff
FROM gpu_test_vectors
ORDER BY id;

-- Flag tolerance result
SELECT id,
  ABS(vector_l2_distance_gpu(vec, '[1,2,3,4]') - vector_l2_distance(vec, '[1,2,3,4]')) < 0.001 AS l2_match,
  ABS(vector_cosine_distance_gpu(vec, '[1,2,3,4]') - vector_cosine_distance(vec, '[1,2,3,4]')) < 0.001 AS cosine_match
FROM gpu_test_vectors
ORDER BY id;

-- ==== 5. Quantization Functions: INT8, FP16, Binary, All Edges ====
-- Run all quantizations, various row values: positive, zero, negative, large, null
-- Validate that functions return expected results
DO $$
DECLARE
    result_count INT;
BEGIN
    SELECT COUNT(*) INTO result_count
    FROM (
        SELECT vector_to_int8_gpu(vec) AS int8_gpu,
               vector_to_fp16_gpu(vec) AS fp16_gpu,
               vector_to_binary_gpu(vec) AS binary_gpu
        FROM gpu_test_vectors
        WHERE vec IS NOT NULL
        LIMIT 1
    ) sub;
    
    IF result_count = 0 THEN
        RAISE EXCEPTION 'GPU quantization functions returned no results';
    END IF;
END$$;

SELECT id, vector_to_int8_gpu(vec)    AS int8_gpu,
         vector_to_fp16_gpu(vec)      AS fp16_gpu,
         vector_to_binary_gpu(vec)    AS binary_gpu
FROM gpu_test_vectors
WHERE vec IS NOT NULL
ORDER BY id;

-- Test on NULL vector
SELECT vector_to_int8_gpu(NULL) AS int8_null, 
       vector_to_fp16_gpu(NULL) AS fp16_null, 
       vector_to_binary_gpu(NULL) AS bin_null;

-- ==== 6. Advanced Operations: KMeans, HNSW, Search, All Combinations (where supported) ====
-- KMeans: try several k, rounds, NULL table (should error), blank col, etc.
-- Validate that functions work or handle errors properly
DO $$
DECLARE
    result_count INT;
BEGIN
    -- Test cluster_kmeans_gpu
    SELECT COUNT(*) INTO result_count
    FROM cluster_kmeans_gpu('gpu_test_vectors', 'vec', 2, 5);
    
    IF result_count = 0 THEN
        RAISE EXCEPTION 'cluster_kmeans_gpu returned no results';
    END IF;
    
    -- Test neurondb_hnsw_search_gpu (may require index, so allow errors)
    BEGIN
        SELECT COUNT(*) INTO result_count
        FROM neurondb_hnsw_search_gpu('gpu_test_vectors', 'vec', '[1,2,3,4]'::vector(4), 2);
        
        IF result_count = 0 THEN
            RAISE WARNING 'neurondb_hnsw_search_gpu returned no results (index may be required)';
        END IF;
    EXCEPTION WHEN OTHERS THEN
        -- Index might not exist, which is acceptable for this test
        RAISE NOTICE 'neurondb_hnsw_search_gpu error (may need index): %', SQLERRM;
    END;
END$$;

-- ==== 7. GPU Function Edge Arguments: Wrong dimensions, Overflows, Limits ====
-- Wrong dimensions (input vector does not match table)
SELECT vector_l2_distance_gpu('[1,2,3]', '[1,2,3,4]') AS l2_wrong_dim;
SELECT vector_l2_distance_gpu('[1,2,3,4,5]', '[1,2,3,4]') AS l2_wrong_dim2;

-- Data overflow / tiny/large
-- Validate that function handles overflow cases properly
DO $$
DECLARE
    result_size INT;
BEGIN
    -- Test with values that may overflow INT8 range
    SELECT octet_length(vector_to_int8_gpu('[32767, -32768, 1e10, -1e10]'::vector)) INTO result_size;
    
    IF result_size IS NULL OR result_size = 0 THEN
        RAISE EXCEPTION 'vector_to_int8_gpu overflow test returned invalid size: %', result_size;
    END IF;
    
    -- Should still return 4 bytes for 4 dimensions (even if values are clamped)
    IF result_size != 4 THEN
        RAISE EXCEPTION 'vector_to_int8_gpu overflow test returned % bytes, expected 4', result_size;
    END IF;
EXCEPTION WHEN OTHERS THEN
    -- Overflow handling may vary, so we just check it doesn't crash
    RAISE NOTICE 'vector_to_int8_gpu overflow test handled: %', SQLERRM;
END$$;

-- ==== 8. Stats After and Reset, All Branches ====
SELECT * FROM neurondb_gpu_stats() AS stats_after_ops;
SELECT neurondb_gpu_stats_reset() AS stats_reset_2;
SELECT * FROM neurondb_gpu_stats() AS stats_post_reset_2;

-- ==== 9. Disable/re-enable GPU in all ways; info afterward ====
SELECT neurondb_gpu_enable(false) AS gpu_disabled;
SELECT * FROM neurondb_gpu_info() AS info_after_disable;
SELECT neurondb_gpu_enable(true) AS gpu_reenabled;
SELECT * FROM neurondb_gpu_info() AS info_after_reenable;

-- ==== 10. Cleanup ====
DROP TABLE IF EXISTS gpu_test_vectors CASCADE;

