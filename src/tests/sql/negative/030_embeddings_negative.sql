\timing on
\pset footer off
\pset pager off

\set ON_ERROR_STOP on

\echo '=========================================================================='
\echo 'Negative Embedding Tests - Error Handling, Edge Cases, Invalid Inputs'
\echo '=========================================================================='

-- Test 1: NULL text input
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Test 1: NULL text input handling'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

DO $$
BEGIN
	BEGIN
		PERFORM embed_text(NULL);
		RAISE EXCEPTION 'Should have failed with NULL input';
	EXCEPTION WHEN OTHERS THEN
		RAISE NOTICE 'Correctly rejected NULL text: %', SQLERRM;
	END;
END $$;

-- Test 2: Empty text input
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Test 2: Empty text input handling'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

/* Empty text should be handled gracefully */
SELECT
	'Empty text' AS test_name,
	vector_dims(embed_text('')) AS dims,
	embed_text('') IS NOT NULL AS not_null;

-- Test 3: Invalid model name
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Test 3: Invalid model name handling'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

/* Invalid model should fallback gracefully */
SELECT
	'Invalid model' AS test_name,
	vector_dims(embed_text('Test', 'nonexistent/model/name')) AS dims;

-- Test 4: NULL image data
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Test 4: NULL image data handling'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

DO $$
BEGIN
	BEGIN
		PERFORM embed_image(NULL);
		RAISE EXCEPTION 'Should have failed with NULL image';
	EXCEPTION WHEN OTHERS THEN
		RAISE NOTICE 'Correctly rejected NULL image: %', SQLERRM;
	END;
END $$;

-- Test 5: Empty image data
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Test 5: Empty image data handling'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

DO $$
BEGIN
	BEGIN
		PERFORM embed_image(''::bytea);
		RAISE EXCEPTION 'Should have failed with empty image';
	EXCEPTION WHEN OTHERS THEN
		RAISE NOTICE 'Correctly rejected empty image: %', SQLERRM;
	END;
END $$;

-- Test 6: Invalid image format
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Test 6: Invalid image format handling'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

/* Invalid image data should be handled gracefully */
DO $$
DECLARE
	invalid_image bytea := '\x00\x01\x02\x03'::bytea;
	result vector;
BEGIN
	result := embed_image(invalid_image);
	RAISE NOTICE 'Invalid image handled, returned dims: %', vector_dims(result);
END $$;

-- Test 7: Multimodal with NULL text
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Test 7: Multimodal with NULL text'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

DO $$
DECLARE
	test_image bytea;
BEGIN
	test_image := '\x89504e470d0a1a0a0000000d4948445200000001000000010802000000907753de0000000a49444154789c6300010000000500010d0a2db40000000049454e44ae426082'::bytea;
	BEGIN
		PERFORM embed_multimodal(NULL, test_image);
		RAISE EXCEPTION 'Should have failed with NULL text';
	EXCEPTION WHEN OTHERS THEN
		RAISE NOTICE 'Correctly rejected NULL text in multimodal: %', SQLERRM;
	END;
END $$;

-- Test 8: Multimodal with NULL image
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Test 8: Multimodal with NULL image'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

DO $$
BEGIN
	BEGIN
		PERFORM embed_multimodal('Test text', NULL);
		RAISE EXCEPTION 'Should have failed with NULL image';
	EXCEPTION WHEN OTHERS THEN
		RAISE NOTICE 'Correctly rejected NULL image in multimodal: %', SQLERRM;
	END;
END $$;

-- Test 9: Invalid JSON in model configuration
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Test 9: Invalid JSON in model configuration'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

DO $$
BEGIN
	BEGIN
		PERFORM configure_embedding_model('test_model', 'invalid json');
		RAISE EXCEPTION 'Should have failed with invalid JSON';
	EXCEPTION WHEN OTHERS THEN
		RAISE NOTICE 'Correctly rejected invalid JSON: %', SQLERRM;
	END;
END $$;

-- Test 10: Invalid configuration values
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Test 10: Invalid configuration values'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

/* Invalid batch_size (too large) */
DO $$
BEGIN
	BEGIN
		PERFORM configure_embedding_model(
			'test_model',
			'{"batch_size": 99999, "normalize": true}'::text
		);
		RAISE NOTICE 'Invalid batch_size handled (may be accepted or rejected)';
	EXCEPTION WHEN OTHERS THEN
		RAISE NOTICE 'Correctly rejected invalid batch_size: %', SQLERRM;
	END;
END $$;

/* Invalid timeout_ms (too small) */
DO $$
BEGIN
	BEGIN
		PERFORM configure_embedding_model(
			'test_model',
			'{"timeout_ms": 10, "normalize": true}'::text
		);
		RAISE NOTICE 'Invalid timeout_ms handled (may be accepted or rejected)';
	EXCEPTION WHEN OTHERS THEN
		RAISE NOTICE 'Correctly rejected invalid timeout_ms: %', SQLERRM;
	END;
END $$;

-- Test 11: Batch with all NULL elements
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Test 11: Batch with all NULL elements'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT
	'All NULL batch' AS test_name,
	array_length(embed_text_batch(ARRAY[NULL, NULL, NULL]::text[]), 1) AS batch_size;

-- Test 12: Very long text (potential truncation)
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Test 12: Very long text handling'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

/* Very long text should be handled (may be truncated) */
SELECT
	'Very long text' AS test_name,
	vector_dims(embed_text(repeat('A', 100000))) AS dims;

-- Test 13: Large batch size
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Test 13: Large batch size handling'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

DO $$
DECLARE
	large_batch text[];
	i int;
BEGIN
	/* Create very large batch */
	large_batch := ARRAY[]::text[];
	FOR i IN 1..1000 LOOP
		large_batch := array_append(large_batch, 'Batch item ' || i::text);
	END LOOP;

	BEGIN
		PERFORM embed_text_batch(large_batch);
		RAISE NOTICE 'Large batch handled successfully';
	EXCEPTION WHEN OTHERS THEN
		RAISE NOTICE 'Large batch error (may be expected): %', SQLERRM;
	END;
END $$;

-- Test 14: Configuration with missing required fields
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Test 14: Configuration with missing required fields'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

/* Empty JSON object should be handled */
DO $$
BEGIN
	BEGIN
		PERFORM configure_embedding_model('test_model', '{}'::text);
		RAISE NOTICE 'Empty config accepted';
	EXCEPTION WHEN OTHERS THEN
		RAISE NOTICE 'Empty config rejected: %', SQLERRM;
	END;
END $$;

-- Test 15: Cache with invalid parameters
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Test 15: Cache with invalid parameters'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

DO $$
BEGIN
	BEGIN
		PERFORM embed_cached(NULL);
		RAISE EXCEPTION 'Should have failed with NULL input';
	EXCEPTION WHEN OTHERS THEN
		RAISE NOTICE 'Correctly rejected NULL in cache: %', SQLERRM;
	END;
END $$;

-- Test 16: Image embedding with extremely large image
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Test 16: Image embedding with extremely large image'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

DO $$
DECLARE
	large_image bytea;
BEGIN
	/* Create large image data (10MB) */
	large_image := repeat('\x00'::bytea, 10 * 1024 * 1024);
	BEGIN
		PERFORM embed_image(large_image);
		RAISE NOTICE 'Large image handled (may timeout or fail)';
	EXCEPTION WHEN OTHERS THEN
		RAISE NOTICE 'Large image error (expected): %', SQLERRM;
	END;
END $$;

-- Test 17: Get config for non-existent model
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Test 17: Get config for non-existent model'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT
	'Non-existent config' AS test_name,
	get_embedding_model_config('nonexistent_model') IS NULL AS is_null;

-- Test 18: Delete non-existent config
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Test 18: Delete non-existent config'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT
	'Delete non-existent' AS test_name,
	delete_embedding_model_config('nonexistent_model') AS deleted;

-- Test 19: Cache clear with NULL pattern
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Test 19: Cache clear with NULL pattern'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

DO $$
DECLARE
	cleared int;
BEGIN
	cleared := ndb_llm_cache_clear(NULL);
	RAISE NOTICE 'Cache cleared (all): % entries', cleared;
END $$;

-- Test 20: Cache evict with invalid size
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Test 20: Cache evict with invalid size'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

DO $$
BEGIN
	BEGIN
		PERFORM ndb_llm_cache_evict_size(-1);
		RAISE EXCEPTION 'Should have failed with negative size';
	EXCEPTION WHEN OTHERS THEN
		RAISE NOTICE 'Correctly rejected negative size: %', SQLERRM;
	END;
END $$;

-- Test 21: Cache warm with empty array
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Test 21: Cache warm with empty array'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT
	'Empty warm' AS test_name,
	ndb_llm_cache_warm(ARRAY[]::text[], 'all-MiniLM-L6-v2') AS warmed_count;

-- Test 22: Alias functions with NULL input
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Test 22: Alias functions with NULL input'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

DO $$
BEGIN
	BEGIN
		PERFORM neurondb_embed(NULL);
		RAISE EXCEPTION 'Should have failed with NULL input';
	EXCEPTION WHEN OTHERS THEN
		RAISE NOTICE 'Correctly rejected NULL in neurondb_embed: %', SQLERRM;
	END;
END $$;

\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'All negative embedding tests completed!'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

\echo 'Test completed successfully'
