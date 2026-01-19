\timing on
\pset footer off
\pset pager off

\set ON_ERROR_STOP on

-- Enable fail-open mode for graceful fallback when LLM is not configured
SET neurondb.llm_fail_open = on;

\echo '=========================================================================='
\echo '=========================================================================='
\echo ''
\echo 'Embedding Model Configuration and Function Aliases Tests'
\echo ''

-- Test 12: Model configuration
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Test 12: Model configuration (configure_embedding_model)'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT
	'Model config' AS test_name,
	configure_embedding_model(
		'test_model',
		'{"batch_size": 32, "normalize": true, "device": "cpu", "timeout_ms": 5000}'::text
	) AS config_success;

-- Verify configuration was stored
SELECT
	'Config stored' AS test_name,
	config_json->>'batch_size' AS batch_size,
	config_json->>'normalize' AS normalize
FROM neurondb.embedding_model_config
WHERE model_name = 'test_model';

-- Test 13: Function name aliases
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Test 13: Function name aliases (neurondb_embed, neurondb_embed_batch)'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT
	'Alias neurondb_embed' AS test_name,
	vector_dims(neurondb_embed('Alias test')) AS dims,
	neurondb_embed('Alias test') IS NOT NULL AS not_null;

SELECT
	'Alias neurondb_embed_batch' AS test_name,
	array_length(neurondb_embed_batch(ARRAY['Text 1', 'Text 2']), 1) AS batch_size;

-- Test 14: Get embedding model configuration
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Test 14: Get embedding model configuration (get_embedding_model_config)'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT
	'Get config' AS test_name,
	get_embedding_model_config('test_model') IS NOT NULL AS config_exists,
	(get_embedding_model_config('test_model'))->>'batch_size' AS batch_size;

-- Test 15: List embedding model configurations
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Test 15: List embedding model configurations (list_embedding_model_configs)'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT
	'List configs' AS test_name,
	COUNT(*) AS config_count
FROM list_embedding_model_configs();

-- Test 16: Delete embedding model configuration
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Test 16: Delete embedding model configuration (delete_embedding_model_config)'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT
	'Delete config' AS test_name,
	delete_embedding_model_config('test_model') AS deleted;

-- Verify deletion
SELECT
	'Config deleted' AS test_name,
	COUNT(*) AS remaining_configs
FROM neurondb.embedding_model_config
WHERE model_name = 'test_model';

\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Configuration and aliases tests completed!'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

\echo 'Test completed successfully'
