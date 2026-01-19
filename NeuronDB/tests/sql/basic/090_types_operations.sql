-- =========================
-- Detailed Vector Type Operation Tests
-- =========================

-- Arithmetic: add, subtract, multiply, divide, negation
SELECT vector_add('[1.0, 2.0]'::vector, '[3.0, 4.0]'::vector) AS add12_34;
SELECT '[1.0, 2.0]'::vector + '[3.0, 4.0]'::vector AS op_add12_34;

SELECT vector_sub('[5.0, 7.0]'::vector, '[2.0, 3.0]'::vector) AS sub57_23;
SELECT '[5.0, 7.0]'::vector - '[2.0, 3.0]'::vector AS op_sub57_23;

SELECT vector_mul('[2.0, 3.0]'::vector, 2.5) AS mul23_scalar;
SELECT '[2.0, 3.0]'::vector * 2.5 AS op_mul23_scalar;

SELECT vector_div('[6.0, 9.0]'::vector, 3.0) AS div69_scalar;
SELECT '[6.0, 9.0]'::vector / 3.0 AS op_div69_scalar;

SELECT vector_neg('[1.0, -2.0]'::vector) AS neg1m2;
SELECT -('[1.0, -2.0]'::vector) AS op_neg1m2;

-- Elementwise min, max, abs
SELECT vector_min('[1.1, 2.2, 3.3]'::vector, '[0.9, 5.5, -7.7]'::vector) AS min_elem;
SELECT vector_max('[1.1, 2.2, 3.3]'::vector, '[0.9, 5.5, -7.7]'::vector) AS max_elem;
SELECT vector_abs('[-1, 2, -3]'::vector) AS abs_elem;

-- Aggregate: sum, mean, dot product
SELECT vector_sum('[1, 2, 3]'::vector) AS sum123;
SELECT vector_mean('[2, 4, 6]'::vector) AS mean246;
SELECT vector_dot('[1, 2]'::vector, '[3, 4]'::vector) AS dot12_34;

-- Norm, normalization, cosine similarity
SELECT vector_norm('[3, 4]'::vector) AS norm34; -- == 5.0
SELECT vector_normalize('[3, 4]'::vector) AS normalize_34;
SELECT vector_cosine_sim('[1, 0]'::vector, '[0, 1]'::vector) AS cosine_sim_perp; -- == 0

-- Distance metrics
SELECT vector_l2_distance('[0, 0]'::vector, '[3, 4]'::vector) AS l2_00_34;
SELECT vector_cosine_distance('[1, 0]'::vector, '[0, 1]'::vector) AS cosine_dist;
SELECT vector_inner_product('[1, 2]'::vector, '[2, 1]'::vector) AS inner_prod12_21;

-- Comparison
SELECT '[1, 2]'::vector = '[1, 2]'::vector AS cmp_eq;    -- true
SELECT '[1, 2]'::vector <> '[2, 1]'::vector AS cmp_neq;  -- true

-- Concatenation
SELECT vector_concat('[1, 2]'::vector, '[3, 4, 5]'::vector) AS concat_vecs;

-- Slicing and subvector
SELECT vector_slice('[1,2,3,4,5]'::vector, 2, 4) AS slice2to4;  -- zero-based, inclusive start, exclusive end?

-- Check vector dimensions
SELECT vector_dims('[1.0]'::vector) AS dims_1;
SELECT vector_dims('[1.0, 2.0]'::vector) AS dims_2;
SELECT vector_dims('[1.0, 2.0, 3.0, 4.0, 5.0]'::vector) AS dims_5;

-- Array conversion (vector <-> array)
SELECT array_to_vector(ARRAY[1.0, 2.0, 3.0]::real[]) AS arr_to_vec;
SELECT array_to_vector(ARRAY[6,7,8]::float4[]) AS arr_to_vec4;
SELECT vector_to_array('[1.0, 2.0, 3.0]'::vector) AS vec_to_arr;

-- Type casting and text I/O
SELECT '[1.0, 2.0, 3.0]'::vector AS txt_io_vec;
SELECT vector_out('[4.5, 5.5, 6.5]'::vector) AS vec_out;

-- Edge: broadcast/scalar ops
SELECT '[1,2]'::vector * 2 AS op_scalar_mul_broadcast;
SELECT 10 * '[1,2]'::vector AS left_scalar_mul; -- if supported

-- Edge: try invalid dimension operation (should ERROR)
DO $$
BEGIN
  BEGIN
    PERFORM vector_add('[1,2,3]'::vector, '[1,2]'::vector);
    RAISE WARNING 'ERROR: vector_add should require same dimensions!';
  EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE 'Correctly rejected mismatched dimension add: %', SQLERRM;
  END;
END$$;

-- Edge: try empty vector (should ERROR)
DO $$
BEGIN
  BEGIN
    PERFORM '[]'::vector;
    RAISE WARNING 'Empty vector allowed! Bug!';
  EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE 'Correctly rejected empty vector: %', SQLERRM;
  END;
END$$;
