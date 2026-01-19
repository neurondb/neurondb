-- GPU Features SQL Definitions
-- Includes GPU-accelerated search functions with index parameter support

-- ==== GPU HNSW Search Function ====
-- GPU-accelerated HNSW k-nearest neighbor search
-- Signature: hnsw_knn_search_gpu(index_name text, query vector, k int, ef_search int)
CREATE FUNCTION hnsw_knn_search_gpu(text, vector, int, int)
	RETURNS TABLE(id bigint, distance real)
	AS 'MODULE_PATHNAME', 'hnsw_knn_search_gpu'
	LANGUAGE C STABLE;

COMMENT ON FUNCTION hnsw_knn_search_gpu(text, vector, int, int) IS
	'GPU-accelerated HNSW k-nearest neighbor search. Returns top-k results with distances.';

-- Overload with default ef_search
CREATE FUNCTION hnsw_knn_search_gpu(text, vector, int)
	RETURNS TABLE(id bigint, distance real)
	AS 'MODULE_PATHNAME', 'hnsw_knn_search_gpu'
	LANGUAGE C STABLE;

COMMENT ON FUNCTION hnsw_knn_search_gpu(text, vector, int) IS
	'GPU-accelerated HNSW k-nearest neighbor search with default ef_search=100.';

-- ==== GPU IVF Search Function ====
-- GPU-accelerated IVF k-nearest neighbor search
-- Signature: ivf_knn_search_gpu(index_name text, query vector, k int, nprobe int)
CREATE FUNCTION ivf_knn_search_gpu(text, vector, int, int)
	RETURNS TABLE(id bigint, distance real)
	AS 'MODULE_PATHNAME', 'ivf_knn_search_gpu'
	LANGUAGE C STABLE;

COMMENT ON FUNCTION ivf_knn_search_gpu(text, vector, int, int) IS
	'GPU-accelerated IVF k-nearest neighbor search. Returns top-k results with distances.';

-- Overload with default nprobe
CREATE FUNCTION ivf_knn_search_gpu(text, vector, int)
	RETURNS TABLE(id bigint, distance real)
	AS 'MODULE_PATHNAME', 'ivf_knn_search_gpu'
	LANGUAGE C STABLE;

COMMENT ON FUNCTION ivf_knn_search_gpu(text, vector, int) IS
	'GPU-accelerated IVF k-nearest neighbor search with default nprobe=10.';

-- ==== Detailed and all possible tests for GPU Features and GPU Acceleration ====
-- Tests gracefully adapt to absence of GPU (run anyway for CPU fallback coverage)
-- Extension assumed to be created in 01_types_basic

-- ==== ENVIRONMENT PREP: Ensure deterministic fallback ====
SET neurondb.compute_mode = off;  -- Guarantee CPU for baseline/consistency

-- ==== 1. GPU Info and Status Functions: all call scenarios ====
-- Actual info/stats output is environment-dependent, so always run, don't just skip
-- Info before anything
SELECT neurondb_gpu_info() AS initial_gpu_info;

-- Try enabling GPU and toggling compute_mode
SELECT neurondb_gpu_enable() AS gpu_enabled;
-- Set to GPU mode
SET neurondb.compute_mode = 1;
SELECT neurondb_gpu_enable() AS gpu_enabled_after_set_gpu;
-- Set to CPU mode
SET neurondb.compute_mode = 0;
SELECT neurondb_gpu_enable() AS gpu_enabled_after_set_cpu;

-- Again check info after toggling
SELECT neurondb_gpu_info() AS post_toggle_gpu_info;

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
SELECT id, vector_to_int8_gpu(vec)    AS int8_gpu,
         vector_to_fp16_gpu(vec)      AS fp16_gpu,
         vector_to_binary_gpu(vec)    AS binary_gpu
FROM gpu_test_vectors
ORDER BY id;

-- Test on NULL vector
SELECT vector_to_int8_gpu(NULL) AS int8_null, vector_to_fp16_gpu(NULL) AS fp16_null, vector_to_binary_gpu(NULL) AS bin_null;

-- ==== 6. Advanced Operations: KMeans, HNSW, Search, All Combinations (where supported) ====
-- KMeans: try several k, rounds, NULL table (should error), blank col, etc.
SELECT cluster_kmeans_gpu('gpu_test_vectors', 'vec', 2, 5) AS kmeans_k2;
SELECT cluster_kmeans_gpu('gpu_test_vectors', 'vec', 3, 10) AS kmeans_k3;
SELECT cluster_kmeans_gpu('gpu_test_vectors', 'vec', 1, 1) AS kmeans_k1;
-- invalid table/column (should error gracefully)
SELECT cluster_kmeans_gpu('nonexistent_table', 'vec', 2, 5) AS kmeans_bad_table;
SELECT cluster_kmeans_gpu('gpu_test_vectors', 'nonexistent_col', 2, 5) AS kmeans_bad_col;

-- HNSW search on valid, missing/NULL table/col/params
SELECT neurondb_hnsw_search_gpu('gpu_test_vectors', 'vec', '[1,2,3,4]', 2) AS hnsw_result;
SELECT neurondb_hnsw_search_gpu('gpu_test_vectors', 'vec', NULL, 2) AS hnsw_null_query;
SELECT neurondb_hnsw_search_gpu('nonexistent_table', 'vec', '[1,2,3,4]', 2) AS hnsw_bad_table;
SELECT neurondb_hnsw_search_gpu('gpu_test_vectors', 'nonexistent_col', '[1,2,3,4]', 2) AS hnsw_bad_col;

-- ==== 7. GPU Function Edge Arguments: Wrong dimensions, Overflows, Limits ====
-- Wrong dimensions (input vector does not match table)
SELECT vector_l2_distance_gpu('[1,2,3]', '[1,2,3,4]') AS l2_wrong_dim;
SELECT vector_l2_distance_gpu('[1,2,3,4,5]', '[1,2,3,4]') AS l2_wrong_dim2;

-- Data overflow / tiny/large
SELECT vector_to_int8_gpu('[32767, -32768, 1e10, -1e10]') AS int8_overflow;

-- ==== 8. Stats After and Reset, All Branches ====
SELECT * FROM neurondb_gpu_stats() AS stats_after_ops;
SELECT neurondb_gpu_stats_reset() AS stats_reset_2;
SELECT * FROM neurondb_gpu_stats() AS stats_post_reset_2;

-- ==== 9. Disable/re-enable GPU in all ways; info afterward ====
SELECT neurondb_gpu_enable(false) AS gpu_disabled;
SELECT neurondb_gpu_info() AS info_after_disable;
SELECT neurondb_gpu_enable(true) AS gpu_reenabled;
SELECT neurondb_gpu_info() AS info_after_reenable;

-- ==== 10. Cleanup ====
DROP TABLE IF EXISTS gpu_test_vectors CASCADE;

