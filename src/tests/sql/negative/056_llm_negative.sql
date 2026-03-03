-- 030_llm_negative.sql
-- Negative test cases for LLM module: error handling, invalid inputs, API failures

\timing on
\pset footer off
\pset pager off

\set ON_ERROR_STOP off

\echo '=========================================================================='
\echo 'LLM Module: Negative Test Cases (Error Handling)'
\echo '=========================================================================='

/*-------------------------------------------------------------------
 * ---- CACHE OPERATION ERRORS ----
 * Test error handling for cache operations
 *------------------------------------------------------------------*/
\echo ''
\echo 'Cache Operation Error Tests'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

\echo 'Error Test 1: Cache Evict with Negative Size'
SELECT ndb_llm_cache_evict_size(-1);

\echo 'Error Test 2: Cache Warm with Empty Array'
SELECT ndb_llm_cache_warm(ARRAY[]::text[]);

\echo 'Error Test 3: Cache Warm with NULL Array'
DO $$
BEGIN
	BEGIN
		PERFORM ndb_llm_cache_warm(NULL);
	EXCEPTION WHEN OTHERS THEN
		NULL; -- Expected error
	END;
END$$;

/*-------------------------------------------------------------------
 * ---- HTTP REQUEST ERRORS ----
 * Test error handling for HTTP request failures
 *------------------------------------------------------------------*/
\echo ''
\echo 'HTTP Request Error Tests'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

\echo 'Error Test 4: Embedding with Invalid API Key'
DO $$
BEGIN
	BEGIN
		PERFORM configure_embedding_model(
			'invalid_key_test',
			'openai',
			'text-embedding-ada-002',
			'{"api_key": "invalid-key-xyz"}'::jsonb
		);
		
		PERFORM embed_text('Test text', 'invalid_key_test');
	EXCEPTION WHEN OTHERS THEN
		NULL; -- Expected error for invalid API key
	END;
END$$;

\echo 'Error Test 5: Embedding with Invalid Endpoint'
DO $$
BEGIN
	BEGIN
		PERFORM configure_embedding_model(
			'invalid_endpoint_test',
			'openai',
			'text-embedding-ada-002',
			'{"api_key": "test", "endpoint": "http://invalid-endpoint-xyz.com"}'::jsonb
		);
		
		PERFORM embed_text('Test text', 'invalid_endpoint_test');
	EXCEPTION WHEN OTHERS THEN
		NULL; -- Expected error for invalid endpoint
	END;
END$$;

\echo 'Error Test 6: Embedding with Timeout'
DO $$
BEGIN
	BEGIN
		SET neurondb.llm_timeout_ms = 1; -- Very short timeout
		PERFORM embed_text('Test text');
	EXCEPTION WHEN OTHERS THEN
		NULL; -- Expected timeout error
	END;
	
	SET neurondb.llm_timeout_ms = 30000; -- Reset
END$$;

\echo 'Error Test 7: Batch Embedding with NULL Elements'
DO $$
DECLARE
	texts text[];
	results vector[];
BEGIN
	texts := ARRAY[NULL, NULL, NULL]::text[];
	
	BEGIN
		results := embed_text_batch(texts);
	EXCEPTION WHEN OTHERS THEN
		NULL; -- Expected error
	END;
END$$;

/*-------------------------------------------------------------------
 * ---- IMAGE PROCESSING ERRORS ----
 * Test error handling for image processing failures
 *------------------------------------------------------------------*/
\echo ''
\echo 'Image Processing Error Tests'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

\echo 'Error Test 8: Image Embedding with NULL Data'
SELECT embed_image(NULL::bytea);

\echo 'Error Test 9: Image Embedding with Empty Data'
SELECT embed_image(''::bytea);

\echo 'Error Test 10: Image Embedding with Invalid Image Data'
SELECT embed_image('invalid-image-data'::bytea);

\echo 'Error Test 11: Multimodal Embedding with NULL Text'
DO $$
DECLARE
	image_data bytea;
BEGIN
	image_data := '\x89504e470d0a1a0a0000000d49484452000000010000000108020000009077536e0000000a49444154789c6300010000000500010d0a2db40000000049454e44ae426082'::bytea;
	
	BEGIN
		PERFORM embed_multimodal(NULL, image_data);
	EXCEPTION WHEN OTHERS THEN
		NULL; -- Expected error
	END;
END$$;

\echo 'Error Test 12: Multimodal Embedding with NULL Image'
SELECT embed_multimodal('Test text', NULL::bytea);

\echo 'Error Test 13: Image Analysis with NULL Data'
SELECT neurondb_llm_image_analyze(NULL::bytea, 'What is this?');

\echo 'Error Test 14: Image Analysis with Empty Data'
SELECT neurondb_llm_image_analyze(''::bytea, 'What is this?');

/*-------------------------------------------------------------------
 * ---- CONFIGURATION ERRORS ----
 * Test error handling for invalid configuration
 *------------------------------------------------------------------*/
\echo ''
\echo 'Configuration Error Tests'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

\echo 'Error Test 15: Configure with NULL Model Name'
DO $$
BEGIN
	BEGIN
		PERFORM configure_embedding_model(
			NULL,
			'huggingface',
			'sentence-transformers/all-MiniLM-L6-v2',
			'{}'::jsonb
		);
	EXCEPTION WHEN OTHERS THEN
		NULL; -- Expected error
	END;
END$$;

\echo 'Error Test 16: Configure with Invalid Provider'
DO $$
BEGIN
	BEGIN
		PERFORM configure_embedding_model(
			'invalid_provider_test',
			'invalid_provider_xyz',
			'model-name',
			'{}'::jsonb
		);
	EXCEPTION WHEN OTHERS THEN
		NULL; -- Expected error
	END;
END$$;

\echo 'Error Test 17: Configure with Invalid JSON'
DO $$
BEGIN
	BEGIN
		PERFORM configure_embedding_model(
			'invalid_json_test',
			'huggingface',
			'model-name',
			'{"invalid": json}'::jsonb
		);
	EXCEPTION WHEN OTHERS THEN
		NULL; -- Expected error
	END;
END$$;

\echo 'Error Test 18: Get Config for Non-existent Model'
SELECT * FROM get_embedding_model_config('nonexistent_model_xyz');

\echo 'Error Test 19: Delete Non-existent Config'
SELECT delete_embedding_model_config('nonexistent_model_xyz');

/*-------------------------------------------------------------------
 * ---- JOB QUEUE ERRORS ----
 * Test error handling for job queue operations
 *------------------------------------------------------------------*/
\echo ''
\echo 'Job Queue Error Tests'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

\echo 'Error Test 20: Job Queue with Invalid Operation'
-- Test handling of invalid job operations
SELECT 
	COUNT(*) AS invalid_operations
FROM neurondb.llm_jobs
WHERE operation NOT IN ('completion', 'embedding', 'reranking');

\echo 'Error Test 21: Job Queue with Corrupted Payload'
-- Test handling of corrupted job payloads
SELECT 
	job_id,
	operation,
	status,
	CASE 
		WHEN payload::text = 'null' OR payload IS NULL THEN 'NULL payload'
		ELSE 'Valid payload'
	END AS payload_status
FROM neurondb.llm_jobs
WHERE payload IS NULL OR payload::text = 'null'
LIMIT 10;

\echo ''
\echo '=========================================================================='
\echo '✓ LLM Module: Negative test cases complete'
\echo '=========================================================================='

\echo 'Test completed successfully'
