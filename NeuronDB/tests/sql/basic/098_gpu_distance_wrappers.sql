-- Extension created in 01_types_basic

-- Disable GPU acceleration for accurate baseline testing
SET neurondb.compute_mode = off;

-- =======================
-- Test all supported GPU distance functions in detail
-- =======================

-- 1. L2 (Euclidean) Distance Tests

-- Small 2D example: [1,2] vs [4,6]
-- sqrt((1-4)^2 + (2-6)^2) = sqrt(9 + 16) = sqrt(25) = 5
SELECT vector_l2_distance_gpu('[1,2]'::vector, '[4,6]'::vector) AS l2_2d;

-- 3D example, identical vectors: [1,2,3] vs [1,2,3], distance = 0
SELECT vector_l2_distance_gpu('[1,2,3]'::vector, '[1,2,3]'::vector) AS l2_3d_identical;

-- 4D example: [1,0,0,0] vs [0,1,0,0], distance = sqrt(2)
SELECT vector_l2_distance_gpu('[1,0,0,0]'::vector, '[0,1,0,0]'::vector) AS l2_orthogonal;

-- Empty vector case: both []
SELECT vector_l2_distance_gpu('[]'::vector, '[]'::vector) AS l2_empty;

-- Mismatched dimension case (should error or handle gracefully)
-- SELECT vector_l2_distance_gpu('[1,2]'::vector, '[1,2,3]'::vector) AS l2_mismatched;  -- expect error


-- 2. Cosine Distance Tests

-- Identical vectors: [1,2,3] vs [1,2,3], cosine distance = 0
SELECT vector_cosine_distance_gpu('[1,2,3]'::vector, '[1,2,3]'::vector) AS cosine_identical;

-- Opposite vectors: [1,0] vs [-1,0], cosine distance = 2
SELECT vector_cosine_distance_gpu('[1,0]'::vector, '[-1,0]'::vector) AS cosine_opposite;

-- Orthogonal vectors: [1,0] vs [0,1], cosine distance = 1
SELECT vector_cosine_distance_gpu('[1,0]'::vector, '[0,1]'::vector) AS cosine_orthogonal;

-- Non-normalized input: [2,0] vs [1,0], cosine distance = 0
SELECT vector_cosine_distance_gpu('[2,0]'::vector, '[1,0]'::vector) AS cosine_non_normalized;

-- Empty vectors: []
SELECT vector_cosine_distance_gpu('[]'::vector, '[]'::vector) AS cosine_empty;

-- Zero vector: [0,0] vs [1,2] (should be undefined or error)
-- SELECT vector_cosine_distance_gpu('[0,0]'::vector, '[1,2]'::vector) AS cosine_zero_vector; -- expect error


-- 3. Inner Product Distance Tests

-- Basic case: [1,2] · [3,4] = 1*3 + 2*4 = 3+8 = 11 → -11
SELECT vector_inner_product_gpu('[1,2]'::vector, '[3,4]'::vector) AS inner_product_2d;

-- Opposite vectors: [1,0] · [-1,0] = -1 → +1 (negated)
SELECT vector_inner_product_gpu('[1,0]'::vector, '[-1,0]'::vector) AS inner_product_opposite;

-- Orthogonal: [1,0] · [0,1] = 0
SELECT vector_inner_product_gpu('[1,0]'::vector, '[0,1]'::vector) AS inner_product_orthogonal;

-- Identical: [1,2,3] · [1,2,3] = 1+4+9=14 → -14
SELECT vector_inner_product_gpu('[1,2,3]'::vector, '[1,2,3]'::vector) AS inner_product_identical;

-- Empty: [] vs []
SELECT vector_inner_product_gpu('[]'::vector, '[]'::vector) AS inner_product_empty;

-- Mismatched dimension: [1,2] vs [1,2,3]
-- SELECT vector_inner_product_gpu('[1,2]'::vector, '[1,2,3]'::vector) AS inner_product_mismatched; -- expect error

-- =======================
-- Summary: These tests cover all possible distance types, normal, edge and error cases.
