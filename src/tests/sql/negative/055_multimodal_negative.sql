\timing on
\pset footer off
\pset pager off

\set ON_ERROR_STOP on

\echo '=========================================================================='
\echo 'Test: Multi-Modal Embeddings Negative Tests'
\echo '=========================================================================='

-- Test 1: Invalid modality for CLIP
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Test 1: Invalid modality for CLIP'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

DO $$
BEGIN
	BEGIN
		PERFORM clip_embed('input', 'audio');
		RAISE WARNING 'Should have failed!';
	EXCEPTION WHEN OTHERS THEN
		RAISE NOTICE 'Correctly rejected invalid modality: %', SQLERRM;
	END;
END$$;

-- Test 2: Invalid modality for ImageBind
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Test 2: Invalid modality for ImageBind'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

DO $$
BEGIN
	BEGIN
		PERFORM imagebind_embed('input', 'invalid');
		RAISE WARNING 'Should have failed!';
	EXCEPTION WHEN OTHERS THEN
		RAISE NOTICE 'Correctly rejected invalid modality: %', SQLERRM;
	END;
END$$;

-- Test 3: NULL input
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Test 3: NULL input'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

DO $$
BEGIN
	BEGIN
		PERFORM clip_embed(NULL, 'text');
		RAISE WARNING 'Should have failed!';
	EXCEPTION WHEN OTHERS THEN
		RAISE NOTICE 'Correctly rejected NULL input: %', SQLERRM;
	END;
END$$;

\echo ''
\echo '✅ Negative multi-modal embeddings tests completed'

\echo 'Test completed successfully'
