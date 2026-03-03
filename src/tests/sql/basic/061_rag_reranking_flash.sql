\timing on
\pset footer off
\pset pager off

\set ON_ERROR_STOP on

\echo '=========================================================================='
\echo 'Test: Flash Attention Reranking Basic Tests'
\echo '=========================================================================='

-- Test 1: Basic rerank_flash function
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Test 1: Basic rerank_flash function'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT
	'Rerank flash basic' AS test_name,
	COUNT(*) AS result_count
FROM rerank_flash(
	'machine learning',
	ARRAY['machine learning algorithms', 'deep learning models', 'neural networks'],
	NULL,
	3
);

-- Test 2: Rerank with model specified
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Test 2: Rerank with model specified'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT
	'Rerank with model' AS test_name,
	COUNT(*) AS result_count
FROM rerank_flash(
	'natural language processing',
	ARRAY['NLP models', 'text processing', 'language models'],
	'cross-encoder',
	2
);

-- Test 3: Long context reranking
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Test 3: Long context reranking'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT
	'Long context rerank' AS test_name,
	COUNT(*) AS result_count
FROM rerank_long_context(
	'query text',
	ARRAY['document 1', 'document 2', 'document 3'],
	8192,
	3
);

\echo ''
\echo '✅ Basic Flash Attention reranking tests completed'

\echo 'Test completed successfully'
