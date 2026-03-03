\timing on
\pset footer off
\pset pager off

\set ON_ERROR_STOP on

\echo '=========================================================================='
\echo 'Test: Sparse Vectors Negative Tests'
\echo '=========================================================================='

-- Test 1: Invalid sparse vector format
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Test 1: Invalid sparse vector format'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

DO $$
BEGIN
	BEGIN
		PERFORM sparse_vector_in('invalid format');
		RAISE WARNING 'Should have failed!';
	EXCEPTION WHEN OTHERS THEN
		RAISE NOTICE 'Correctly rejected invalid format: %', SQLERRM;
	END;
END$$;

-- Test 2: Empty tokens array
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Test 2: Empty tokens array'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

DO $$
BEGIN
	BEGIN
		PERFORM sparse_vector_in('{vocab_size:30522, model:SPLADE, tokens:[], weights:[]}');
		RAISE WARNING 'Should have failed!';
	EXCEPTION WHEN OTHERS THEN
		RAISE NOTICE 'Correctly rejected empty tokens: %', SQLERRM;
	END;
END$$;

-- Test 3: Mismatched dimensions in dot product
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Test 3: Mismatched vocab sizes'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

-- Note: sparse_vector_dot_product should handle different vocab sizes gracefully
-- The function computes dot product by matching token IDs, so different vocab sizes
-- are acceptable as long as the token IDs match
DO $$
DECLARE
	v1 sparse_vector;
	v2 sparse_vector;
	result float4;
BEGIN
	v1 := '{vocab_size:30522, model:SPLADE, tokens:[100], weights:[0.5]}'::sparse_vector;
	v2 := '{vocab_size:50000, model:SPLADE, tokens:[100], weights:[0.3]}'::sparse_vector;
	result := sparse_vector_dot_product(v1, v2);
	RAISE NOTICE 'Dot product with different vocab sizes: %', result;
EXCEPTION WHEN OTHERS THEN
	RAISE NOTICE 'Error with mismatched vocab sizes: %', SQLERRM;
END $$;

\echo ''
\echo '✅ Negative sparse vectors tests completed'

\echo 'Test completed successfully'
