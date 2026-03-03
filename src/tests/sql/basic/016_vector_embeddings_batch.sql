\timing on
\pset footer off
\pset pager off

\set ON_ERROR_STOP on

\echo '=========================================================================='
\echo '=========================================================================='
\echo ''
\echo 'NOTE: embed_text_batch() warnings are expected if LLM is not configured.'
\echo '      To generate real embeddings, configure:'
\echo '      - neurondb.llm_api_key (Hugging Face API key)'
\echo '      - Or enable GPU embedding via GUC (ALTER SYSTEM SET neurondb.compute_mode = on)'
\echo '      Without configuration, embed_text_batch() returns zero vectors (graceful fallback).'
\echo ''

-- Test 3: Batch text embedding
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Test 3: Batch text embedding (embed_text_batch)'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

WITH batch_result AS (
	SELECT embed_text_batch(ARRAY['First text', 'Second text', 'Third text']) AS embeddings,
	       embed_text_batch(ARRAY['Test']) AS single_embedding
)
SELECT
	'Batch embedding' AS test_name,
	array_length(batch_result.embeddings, 1) AS batch_size,
	vector_dims((batch_result.single_embedding)[1]) AS first_dim
FROM batch_result;

-- Test 4: Batch embedding with NULL elements
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Test 4: Batch embedding with NULL elements'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT
	'Batch with NULLs' AS test_name,
	array_length(embed_text_batch(ARRAY['Valid text', NULL, 'Another text']), 1) AS batch_size;

-- Test 5: Empty batch
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Test 5: Empty batch embedding'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT
	'Empty batch' AS test_name,
	array_length(embed_text_batch(ARRAY[]::text[]), 1) AS batch_size;

\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Batch embedding tests completed!'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

\echo 'Test completed successfully'
