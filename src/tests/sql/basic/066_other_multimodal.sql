\timing on
\pset footer off
\pset pager off

\set ON_ERROR_STOP on

\echo '=========================================================================='
\echo 'Test: Multi-Modal Embeddings Basic Tests'
\echo '=========================================================================='

-- Test 1: CLIP text embedding
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Test 1: CLIP text embedding'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT
	'CLIP text embed' AS test_name,
	clip_embed('machine learning', 'text') IS NOT NULL AS created;

-- Test 2: CLIP image embedding
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Test 2: CLIP image embedding'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT
	'CLIP image embed' AS test_name,
	clip_embed('image_path', 'image') IS NOT NULL AS created;

-- Test 3: ImageBind text embedding
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Test 3: ImageBind text embedding'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT
	'ImageBind text embed' AS test_name,
	imagebind_embed('natural language processing', 'text') IS NOT NULL AS created;

-- Test 4: ImageBind audio embedding
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Test 4: ImageBind audio embedding'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT
	'ImageBind audio embed' AS test_name,
	imagebind_embed('audio_path', 'audio') IS NOT NULL AS created;

\echo ''
\echo '✅ Basic multi-modal embeddings tests completed'

\echo 'Test completed successfully'
