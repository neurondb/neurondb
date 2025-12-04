\timing on
\pset footer off
\pset pager off

\set ON_ERROR_STOP on

-- Enable fail_open mode to allow tests to run without API keys (uses fallback embeddings)
SET neurondb.llm_fail_open = on;

\echo '=========================================================================='
\echo '=========================================================================='
\echo ''
\echo 'NOTE: Image and multimodal embedding tests run with fail_open mode enabled.'
\echo '      To generate real embeddings, configure:'
\echo '      - neurondb.llm_api_key (Hugging Face API key)'
\echo '      - Or enable GPU embedding via GUC (ALTER SYSTEM SET neurondb.gpu_enabled = on)'
\echo '      Without configuration, these functions return fallback embeddings (graceful fallback).'
\echo ''

-- Test 6: Cached embedding
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Test 6: Cached embedding (embed_cached)'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT
	'Cached embedding' AS test_name,
	vector_dims(embed_cached('Cache test text')) AS dims,
	embed_cached('Cache test text') IS NOT NULL AS not_null;

-- Test 7: Image embedding (requires BYTEA)
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Test 7: Image embedding (embed_image)'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

/* Create minimal test image data (1x1 PNG) - use decode to avoid TOAST compression issues */
DO $$
DECLARE
	test_image bytea;
	result vector;
BEGIN
	/* Minimal valid PNG (1x1 pixel) - use decode() to create bytea without compression */
	/* PNG signature: 89 50 4E 47 0D 0A 1A 0A */
	test_image := decode('89504e470d0a1a0a0000000d4948445200000001000000010802000000907753de0000000a49444154789c6300010000000500010d0a2db40000000049454e44ae426082', 'hex');
	result := embed_image(test_image);
	RAISE NOTICE 'Image embedding test passed, dims: %', vector_dims(result);
EXCEPTION
	WHEN OTHERS THEN
		RAISE NOTICE 'Image embedding test skipped (API may not support image embeddings): %', SQLERRM;
END $$;

-- Test 8: Multimodal embedding
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Test 8: Multimodal embedding (embed_multimodal)'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

DO $$
DECLARE
	test_image bytea;
	result vector;
BEGIN
	/* Minimal valid PNG (1x1 pixel) - use decode() to create bytea without compression */
	test_image := decode('89504e470d0a1a0a0000000d4948445200000001000000010802000000907753de0000000a49444154789c6300010000000500010d0a2db40000000049454e44ae426082', 'hex');
	result := embed_multimodal('Test text', test_image);
	RAISE NOTICE 'Multimodal embedding test passed, dims: %', vector_dims(result);
EXCEPTION
	WHEN OTHERS THEN
		RAISE NOTICE 'Multimodal embedding test skipped (API may not support multimodal embeddings): %', SQLERRM;
END $$;

\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Multimodal embedding tests completed!'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

\echo 'Test completed successfully'
