\timing on
\pset footer off
\pset pager off

\set ON_ERROR_STOP on

-- Enable fail-open mode for graceful fallback when LLM is not configured
SET neurondb.llm_fail_open = on;

\echo '=========================================================================='
\echo '=========================================================================='
\echo ''
\echo 'NOTE: embed_text() warnings are expected if LLM is not configured.'
\echo '      To generate real embeddings, configure:'
\echo '      - neurondb.llm_api_key (Hugging Face API key)'
\echo '      - Or enable GPU embedding via GUC (ALTER SYSTEM SET neurondb.compute_mode = on)'
\echo '      Without configuration, embed_text() returns zero vectors (graceful fallback).'
\echo ''

-- Test 1: Basic text embedding
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Test 1: Basic text embedding (embed_text)'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT
	'Basic text embedding' AS test_name,
	vector_dims(embed_text('Hello, world!')) AS dims,
	embed_text('Hello, world!') IS NOT NULL AS not_null;

-- Test 2: Text embedding with custom model
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Test 2: Text embedding with custom model'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT
	'Custom model embedding' AS test_name,
	vector_dims(embed_text('Test text', 'sentence-transformers/all-MiniLM-L6-v2')) AS dims,
	embed_text('Test text', 'sentence-transformers/all-MiniLM-L6-v2') IS NOT NULL AS not_null;

-- Test 9: Vector consistency
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Test 9: Vector consistency (same text, same embedding)'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

WITH embeddings AS (
	SELECT
		embed_text('Consistency test') AS vec1,
		embed_text('Consistency test') AS vec2
)
SELECT
	'vector consistency' AS test_name,
	vector_dims(vec1) = vector_dims(vec2) AS dims_match,
	vec1 <-> vec2 AS distance
FROM embeddings;

-- Test 10: Unicode and special characters
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Test 10: Unicode and special characters'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT
	'Unicode test' AS test_name,
	vector_dims(embed_text('Hello 世界 🌍')) AS dims,
	vector_dims(embed_text('Text with "quotes" and ''apostrophes''')) AS dims2;

-- Test 11: Long text
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Test 11: Long text embedding'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT
	'Long text' AS test_name,
	vector_dims(embed_text(repeat('This is a long text. ', 100))) AS dims;

\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Basic text embedding tests completed!'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

\echo 'Test completed successfully'
