-- ============================================================================
-- Test 001: Vector Type Basics
-- ============================================================================
-- Demonstrates: Vector creation, I/O, dimensions, basic properties
-- ============================================================================

\echo '=========================================================================='
\echo '|              Vector Type Basics - NeuronDB                            |'
\echo '=========================================================================='
\echo ''

-- Test 1: Vector creation from text
\echo 'Test 1: Vector creation from text literals'
SELECT 
    '[1,2,3]'::vector AS vec_3d,
    '[1,2,3,4,5]'::vector AS vec_5d,
    '[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]'::vector AS vec_8d;

\echo ''
\echo 'Test 2: Vector with specified dimensions'
SELECT 
    '[1,2,3]'::vector(3) AS vec_3d,
    '[1,2,3,4,5,6,7,8,9,10]'::vector(10) AS vec_10d,
    '[' || array_to_string((SELECT array_agg(i::float4) FROM generate_series(1, 384) i), ',') || ']'::vector(384) AS vec_384d;

\echo ''
\echo 'Test 3: Vector dimensions and properties'
SELECT 
    vector_dims('[1,2,3]'::vector) AS dims_3,
    vector_dims('[1,2,3,4,5]'::vector) AS dims_5,
    vector_dims('[' || array_to_string((SELECT array_agg(i::float4) FROM generate_series(1, 128) i), ',') || ']'::vector) AS dims_128;

\echo ''
\echo 'Test 4: Vector norm (magnitude)'
SELECT 
    '[3,4]'::vector AS vector,
    vector_norm('[3,4]'::vector) AS norm_should_be_5,
    '[1,0,0]'::vector AS unit_vec,
    vector_norm('[1,0,0]'::vector) AS norm_should_be_1;

\echo ''
\echo 'Test 5: Vector normalization'
SELECT 
    '[3,4]'::vector AS original,
    vector_normalize('[3,4]'::vector) AS normalized,
    vector_norm(vector_normalize('[3,4]'::vector)) AS norm_should_be_1;

\echo ''
\echo 'Test 6: Creating vectors from arrays'
SELECT 
    array_to_vector(ARRAY[1,2,3]::real[]) AS from_array,
    array_to_vector(ARRAY[0.1, 0.2, 0.3, 0.4, 0.5]::real[]) AS from_float_array;

\echo ''
\echo 'Test 7: Converting vectors to arrays'
SELECT 
    '[1,2,3,4,5]'::vector AS original,
    vector_to_array('[1,2,3,4,5]'::vector) AS as_array,
    (vector_to_array('[1,2,3,4,5]'::vector))[1] AS first_element,
    (vector_to_array('[1,2,3,4,5]'::vector))[5] AS last_element;

\echo ''
\echo 'Test 8: Vector storage in tables'
CREATE TEMP TABLE vector_test (
    id SERIAL PRIMARY KEY,
    vec_small VECTOR(3),
    vec_medium VECTOR(128),
    vec_large VECTOR(384),
    description TEXT
);

INSERT INTO vector_test (vec_small, vec_medium, vec_large, description) VALUES
    ('[1,2,3]'::vector, NULL, NULL, '3D vector'),
    (NULL, '[' || array_to_string((SELECT array_agg(random()::float4) FROM generate_series(1, 128) i), ',') || ']'::vector, NULL, '128D random vector'),
    (NULL, NULL, '[' || array_to_string((SELECT array_agg(random()::float4) FROM generate_series(1, 384) i), ',') || ']'::vector, '384D embedding');

SELECT 
    id,
    description,
    vector_dims(vec_small) AS small_dims,
    vector_dims(vec_medium) AS medium_dims,
    vector_dims(vec_large) AS large_dims
FROM vector_test;

\echo ''
\echo '=========================================================================='
\echo 'Vector Basics Test Complete!'
\echo '  ✅ Vector creation from text'
\echo '  ✅ Dimension specification'
\echo '  ✅ Vector properties (dims, norm)'
\echo '  ✅ Normalization'
\echo '  ✅ Array conversion (to/from)'
\echo '  ✅ Table storage'
\echo '=========================================================================='
\echo ''

