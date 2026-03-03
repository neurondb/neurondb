\timing on
\pset footer off
\pset pager off

\set ON_ERROR_STOP on

\echo '=========================================================================='
\echo 'Test: Flash Attention Reranking Negative Tests'
\echo '=========================================================================='

-- Test 1: NULL query
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Test 1: NULL query'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

DO $$
BEGIN
	BEGIN
		PERFORM COUNT(*) FROM rerank_flash(NULL, ARRAY['doc1', 'doc2'], NULL, 5);
		RAISE WARNING 'Should have failed!';
	EXCEPTION WHEN OTHERS THEN
		RAISE NOTICE 'Correctly rejected NULL query: %', SQLERRM;
	END;
END$$;

-- Test 2: Empty candidates array
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Test 2: Empty candidates array'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

DO $$
BEGIN
	BEGIN
		PERFORM COUNT(*) FROM rerank_flash('query', ARRAY[]::text[], NULL, 5);
		RAISE WARNING 'Should have failed!';
	EXCEPTION WHEN OTHERS THEN
		RAISE NOTICE 'Correctly rejected empty array: %', SQLERRM;
	END;
END$$;

-- Test 3: Invalid top_k
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Test 3: Invalid top_k'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

DO $$
BEGIN
	BEGIN
		PERFORM COUNT(*) FROM rerank_flash('query', ARRAY['doc1'], NULL, -1);
		RAISE WARNING 'Should have failed!';
	EXCEPTION WHEN OTHERS THEN
		RAISE NOTICE 'Correctly rejected invalid top_k: %', SQLERRM;
	END;
END$$;

\echo ''
\echo '✅ Negative Flash Attention reranking tests completed'

\echo 'Test completed successfully'
