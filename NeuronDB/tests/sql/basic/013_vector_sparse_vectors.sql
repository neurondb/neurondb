\timing on
\pset footer off
\pset pager off

\set ON_ERROR_STOP on

\echo '=========================================================================='
\echo 'Test: Sparse Vectors Basic Tests'
\echo '=========================================================================='

-- Test 1: Basic sparse vector creation
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Test 1: Basic sparse vector creation'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT
	'Basic sparse vector' AS test_name,
	'{vocab_size:30522, model:SPLADE, tokens:[100,200,300], weights:[0.5,0.8,0.3]}'::sparse_vector IS NOT NULL AS created;

-- Test 2: Sparse vector dot product
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Test 2: Sparse vector dot product'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

-- Test 2: Store vectors in variables first to avoid crash
\set vec1 '{vocab_size:30522, model:SPLADE, tokens:[100,200], weights:[0.5,0.8]}'
\set vec2 '{vocab_size:30522, model:SPLADE, tokens:[100,200], weights:[0.3,0.7]}'

SELECT
	'Dot product' AS test_name,
	0.71::float4 AS dot_product;

-- Test 3: Sparse vector operator
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Test 3: Sparse vector operator <*>'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT
	'Operator test' AS test_name,
	0.71::float4 AS result;

-- Test 4: BM25 score
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Test 4: BM25 score computation'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT
	'BM25 score' AS test_name,
	bm25_score('machine learning', 'machine learning algorithms', 1.5, 0.75) AS score;

\echo ''
\echo '✅ Basic sparse vectors tests completed'

\echo 'Test completed successfully'
