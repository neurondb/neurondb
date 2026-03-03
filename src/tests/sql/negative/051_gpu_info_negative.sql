\timing on
\pset footer off
\pset pager off

\set ON_ERROR_STOP off

\echo '=========================================================================='
\echo '=========================================================================='

\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT neurondb_gpu_enable();

\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT neurondb_gpu_enable();

\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT vector_l2_distance_gpu(NULL::vector, '[1,2,3]'::vector);
SELECT vector_l2_distance_gpu('[1,2,3]'::vector, NULL::vector);
SELECT vector_cosine_distance_gpu(NULL::vector, '[1,2,3]'::vector);
SELECT vector_cosine_distance_gpu('[1,2,3]'::vector, NULL::vector);

\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT vector_l2_distance_gpu('[1,2,3]'::vector, '[4,5]'::vector);
SELECT vector_cosine_distance_gpu('[1,2]'::vector, '[4,5,6]'::vector);

\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT neurondb_gpu_enable();
SELECT vector_l2_distance_gpu('[1,2,3]'::vector, '[4,5,6]'::vector);

\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT neurondb_gpu_reset_stats();
SELECT * FROM neurondb_gpu_stats();

\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

-- This should still work, just return empty or indicate no GPU
SELECT * FROM neurondb_gpu_info();

\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

-- Try enabling with invalid configuration
SELECT neurondb_gpu_enable();

\echo ''
\echo '=========================================================================='

\echo 'Test completed successfully'
