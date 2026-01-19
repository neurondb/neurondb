-- NeurondB: Detailed Basic Type and Operation Tests

-- Extension creation (should succeed, idempotent in tests)
DROP EXTENSION IF EXISTS neurondb CASCADE;
CREATE EXTENSION neurondb CASCADE;

-- ===============================
-- VECTOR TYPE (float32 dense)
-- ===============================

-- Type parsing: valid input
SELECT '[1.0, 2.0, 3.0]'::vector AS v1;
SELECT array_to_string(vector_to_array('[1,2,3,4]'::vector), ',') AS arr_v2;

-- Type parsing: whitespace tolerance, scientific notation
SELECT '[ -3.5,4.01e1 , 0 , 2e-2]'::vector AS v3;
SELECT '[0]'::vector AS v_singleton;

-- Implicit input/output/invariance
SELECT '[1.00,2.00]'::vector = '[1,2]'::vector AS eq_simple;

-- Edge: empty vector (should ERROR)
DO $$
BEGIN
  BEGIN
    PERFORM '[]'::vector;
    RAISE WARNING 'Empty vector allowed! Bug!';
  EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE 'Correctly rejected empty vector: %', SQLERRM;
  END;
END$$;

-- vector_dims
SELECT vector_dims('[1,2,3,4,5]'::vector) AS dims_v4;   -- should be 5
SELECT vector_dims('[5]'::vector) AS dims_v5;           -- should be 1

-- Norm, normalization
SELECT vector_norm('[3,4]'::vector) AS norm_3_4;           -- 5.0 (L2)
SELECT vector_normalize('[0,100]'::vector) AS unit_vec;
SELECT vector_norm(vector_normalize('[0,100]'::vector)) AS unit_norm; -- ~1

-- Vector arithmetic
SELECT '[1,2,3]'::vector + '[4,5,6]'::vector AS v_add;
SELECT '[2,4]'::vector * 2.0 AS v_scale;
SELECT '[6,9,12]'::vector / 3.0 AS v_div;

-- Concatenation
SELECT vector_concat('[1,2]'::vector, '[3,4]'::vector) AS vcat;

-- ===============================
-- PACKED VECTORS (vectorp)
-- ===============================
SELECT vectorp_in('[1.0, 2.0, 3.0]')::text AS vp_parse;
SELECT vectorp_dims(vectorp_in('[1.0, 2.0, 3.0, 4.0]')) AS vp_dims;

-- Edge: invalid (should fail)
DO $$
BEGIN
  BEGIN
    PERFORM vectorp_in('foo');
    RAISE WARNING 'vectorp_in accepted invalid!';
  EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE 'vectorp_in correctly rejected invalid: %', SQLERRM;
  END;
END$$;

-- ===============================
-- SPARSE VECTOR MAP (vecmap)
-- ===============================
SELECT vecmap_in('{dim:10, nnz:3, indices:[0,3,7], values:[1.5,2.3,0.8]}')::text AS vm_parse;

-- Edge: missing field (should fail)
DO $$
BEGIN
  BEGIN
    PERFORM vecmap_in('{dim:10, indices:[1]}');
    RAISE WARNING 'vecmap_in accepted incomplete!';
  EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE 'vecmap_in correctly rejected incomplete: %', SQLERRM;
  END;
END$$;

-- ===============================
-- VECTOR GRAPH (vgraph)
-- ===============================
SELECT vgraph_in('{nodes:5, edges:[[0,1],[1,2],[2,3],[3,4]]}')::text AS vg_parse;

-- Loop edge and multiedge
SELECT vgraph_in('{nodes:3, edges:[[0,0],[1,2],[1,2]]}')::text AS vg_loops;

-- ===============================
-- RETRIEVAL TEXT (rtext)
-- ===============================
SELECT rtext_in('sample text for retrieval')::text AS rtxt_basic;
SELECT rtext_in('こんにちは')::text AS rtxt_unicode;
SELECT rtext_in('')::text AS rtxt_empty;

-- Edge: binary/NULL (should fail)
DO $$
BEGIN
  BEGIN
    PERFORM rtext_in(E'\\xdeadbeef');
    RAISE WARNING 'rtext_in accepted binary!';
  EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE 'rtext_in correctly rejected binary: %', SQLERRM;
  END;
END$$;
