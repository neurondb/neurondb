\timing on
\pset footer off
\pset pager off

\set ON_ERROR_STOP off

\echo '=========================================================================='
\echo '=========================================================================='

\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT vector_dims(NULL::vector);
SELECT vector_norm(NULL::vector);
SELECT vector_add(NULL::vector, '[1,2,3]'::vector);
SELECT vector_add('[1,2,3]'::vector, NULL::vector);
SELECT vector_l2_distance(NULL::vector, '[1,2,3]'::vector);
SELECT vector_l2_distance('[1,2,3]'::vector, NULL::vector);

\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT vector_add('[1,2,3]'::vector, '[4,5]'::vector);
SELECT vector_sub('[1,2,3]'::vector, '[4,5,6,7]'::vector);
SELECT vector_l2_distance('[1,2,3]'::vector, '[4,5]'::vector);
SELECT vector_cosine_distance('[1,2]'::vector, '[4,5,6]'::vector);

\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT vector_dims('[]'::vector);
SELECT vector_norm('[]'::vector);
SELECT vector_normalize('[]'::vector);

\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT '[1,2,abc]'::vector;
SELECT '[1,2,3,4,5'::vector;
SELECT '1,2,3,4,5]'::vector;

\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT vector_normalize('[0,0,0]'::vector);
SELECT vector_cosine_distance('[0,0,0]'::vector, '[1,2,3]'::vector);

\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT vector_mul('[1,2,3]'::vector, NULL::float8);
SELECT vector_mul('[1,2,3]'::vector, 'NaN'::float8);
SELECT vector_mul('[1,2,3]'::vector, 'Infinity'::float8);

\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT vector_minkowski_distance('[1,2,3]'::vector, '[4,5,6]'::vector, 0.0);
SELECT vector_minkowski_distance('[1,2,3]'::vector, '[4,5,6]'::vector, -1.0);
SELECT vector_minkowski_distance('[1,2,3]'::vector, '[4,5,6]'::vector, NULL::float8);

\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT vector_concat(NULL::vector, '[1,2,3]'::vector);
SELECT vector_concat('[1,2,3]'::vector, NULL::vector);
SELECT vector_concat('[]'::vector, '[1,2,3]'::vector);
SELECT vector_concat('[1,2,3]'::vector, '[]'::vector);

\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT vector_to_binary(NULL::vector);
SELECT vector_to_int8(NULL::vector);
SELECT int8_to_vector(NULL::bytea);

\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT int8_to_vector('\x00'::bytea);
SELECT int8_to_vector('invalid_binary_data'::bytea);

\echo ''
\echo '=========================================================================='

\echo 'Test completed successfully'
