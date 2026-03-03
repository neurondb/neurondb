-- ============================================================================
-- Test 003: Vector Operations
-- ============================================================================
-- Demonstrates: Addition, subtraction, multiplication, concatenation
-- ============================================================================

\echo '=========================================================================='
\echo '|              Vector Operations - NeuronDB                             |'
\echo '=========================================================================='
\echo ''

-- Test 1: Vector addition
\echo 'Test 1: Vector addition'
SELECT 
    '[1,2,3]'::vector AS vec1,
    '[4,5,6]'::vector AS vec2,
    vector_add('[1,2,3]'::vector, '[4,5,6]'::vector) AS sum,
    '[1,2,3]'::vector + '[4,5,6]'::vector AS sum_with_operator;

\echo ''
\echo 'Test 2: Vector subtraction'
SELECT 
    '[10,20,30]'::vector AS vec1,
    '[1,2,3]'::vector AS vec2,
    vector_sub('[10,20,30]'::vector, '[1,2,3]'::vector) AS difference,
    '[10,20,30]'::vector - '[1,2,3]'::vector AS diff_with_operator;

\echo ''
\echo 'Test 3: Scalar multiplication'
SELECT 
    '[1,2,3]'::vector AS original,
    vector_mul('[1,2,3]'::vector, 2.0) AS multiplied_by_2,
    '[1,2,3]'::vector * 2.0 AS mul_with_operator,
    vector_mul('[1,2,3]'::vector, 0.5) AS multiplied_by_half;

\echo ''
\echo 'Test 4: Vector concatenation'
SELECT 
    '[1,2,3]'::vector AS vec1,
    '[4,5,6]'::vector AS vec2,
    vector_concat('[1,2,3]'::vector, '[4,5,6]'::vector) AS concatenated,
    vector_dims(vector_concat('[1,2,3]'::vector, '[4,5,6]'::vector)) AS result_dims;

\echo ''
\echo 'Test 5: Complex operations (normalization after multiplication)'
SELECT 
    '[3,4]'::vector AS original,
    vector_mul('[3,4]'::vector, 10.0) AS scaled_up,
    vector_norm(vector_mul('[3,4]'::vector, 10.0)) AS norm_scaled,
    vector_normalize(vector_mul('[3,4]'::vector, 10.0)) AS normalized,
    vector_norm(vector_normalize(vector_mul('[3,4]'::vector, 10.0))) AS norm_should_be_1;

\echo ''
\echo 'Test 6: Vector arithmetic expressions'
SELECT 
    '[1,1,1]'::vector AS a,
    '[2,2,2]'::vector AS b,
    '[3,3,3]'::vector AS c,
    ('[1,1,1]'::vector + '[2,2,2]'::vector) - '[3,3,3]'::vector AS result_should_be_zero,
    vector_norm(('[1,1,1]'::vector + '[2,2,2]'::vector) - '[3,3,3]'::vector) AS norm_should_be_zero;

\echo ''
\echo 'Test 7: Building vectors programmatically'
WITH numbers AS (
    SELECT i, sin(i::float) AS val
    FROM generate_series(1, 10) i
)
SELECT 
    '[' || string_agg(val::text, ',') || ']'::vector AS sine_wave_vector,
    vector_dims('[' || string_agg(val::text, ',') || ']'::vector) AS dimensions
FROM numbers;

\echo ''
\echo '=========================================================================='
\echo 'Vector Operations Test Complete!'
\echo '  ✅ Addition (vector + vector)'
\echo '  ✅ Subtraction (vector - vector)'
\echo '  ✅ Scalar multiplication (vector * scalar)'
\echo '  ✅ Concatenation (combine vectors)'
\echo '  ✅ Complex expressions'
\echo '  ✅ Programmatic vector creation'
\echo '=========================================================================='
\echo ''

