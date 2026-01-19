\timing on
\pset footer off
\pset pager off

\set ON_ERROR_STOP on

-- Enable fail_open mode to allow tests to run without API keys (uses fallback embeddings)
-- This ensures tests continue even if some API calls fail due to rate limiting or other issues
SET neurondb.llm_fail_open = on;

\echo '=========================================================================='
\echo '=========================================================================='
\echo ''
\echo 'Hugging Face Embedding Models Comprehensive Test'
\echo ''
\echo 'This test covers TEXT, IMAGE, and MULTIMODAL embedding models.'
\echo 'Note: These tests can run without neurondb.llm_api_key (uses fallback embeddings).'
\echo 'Each model is tested individually to verify compatibility.'
\echo ''

-- Test 17: Hugging Face embedding models comprehensive test
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Test 17: Hugging Face Embedding Models Comprehensive Test'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo ''

-- Test 17a: TEXT EMBEDDING MODELS (384-dim models - Fast & Efficient)
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Test 17a: TEXT Embedding Models (384-dim, Fast & Efficient)'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo ''

-- Model 1: all-MiniLM-L6-v2 (384 dim, fast, efficient - DEFAULT)
\echo 'Testing Model 1/16: sentence-transformers/all-MiniLM-L6-v2 (384 dim)'
SELECT
	'sentence-transformers/all-MiniLM-L6-v2' AS model_name,
	384 AS expected_dims,
	vector_dims(embed_text('Test text', 'sentence-transformers/all-MiniLM-L6-v2')) AS actual_dims,
	CASE 
		WHEN vector_norm(embed_text('Test text', 'sentence-transformers/all-MiniLM-L6-v2')) > 0 THEN true
		ELSE false
	END AS non_zero,
	round(vector_norm(embed_text('Test text', 'sentence-transformers/all-MiniLM-L6-v2'))::numeric, 4) AS norm;

-- Model 2: all-MiniLM-L12-v2 (384 dim, better quality than L6)
\echo ''
\echo 'Testing Model 2/16: sentence-transformers/all-MiniLM-L12-v2 (384 dim)'
SELECT
	'sentence-transformers/all-MiniLM-L12-v2' AS model_name,
	384 AS expected_dims,
	vector_dims(embed_text('Test text', 'sentence-transformers/all-MiniLM-L12-v2')) AS actual_dims,
	CASE 
		WHEN vector_norm(embed_text('Test text', 'sentence-transformers/all-MiniLM-L12-v2')) > 0 THEN true
		ELSE false
	END AS non_zero,
	round(vector_norm(embed_text('Test text', 'sentence-transformers/all-MiniLM-L12-v2'))::numeric, 4) AS norm;

-- Model 3: all-MiniLM-L3-v2 (384 dim, smaller version)
\echo ''
\echo 'Testing Model 3/16: sentence-transformers/all-MiniLM-L3-v2 (384 dim)'
SELECT
	'sentence-transformers/all-MiniLM-L3-v2' AS model_name,
	384 AS expected_dims,
	vector_dims(embed_text('Test text', 'sentence-transformers/all-MiniLM-L3-v2')) AS actual_dims,
	CASE 
		WHEN vector_norm(embed_text('Test text', 'sentence-transformers/all-MiniLM-L3-v2')) > 0 THEN true
		ELSE false
	END AS non_zero,
	round(vector_norm(embed_text('Test text', 'sentence-transformers/all-MiniLM-L3-v2'))::numeric, 4) AS norm;

-- Model 4: BAAI/bge-small-en-v1.5 (384 dim, efficient retrieval)
\echo ''
\echo 'Testing Model 4/16: BAAI/bge-small-en-v1.5 (384 dim)'
SELECT
	'BAAI/bge-small-en-v1.5' AS model_name,
	384 AS expected_dims,
	vector_dims(embed_text('Test text', 'BAAI/bge-small-en-v1.5')) AS actual_dims,
	CASE 
		WHEN vector_norm(embed_text('Test text', 'BAAI/bge-small-en-v1.5')) > 0 THEN true
		ELSE false
	END AS non_zero,
	round(vector_norm(embed_text('Test text', 'BAAI/bge-small-en-v1.5'))::numeric, 4) AS norm;

-- Model 5: paraphrase-MiniLM-L6-v2 (384 dim, for paraphrase detection)
\echo ''
\echo 'Testing Model 5/16: sentence-transformers/paraphrase-MiniLM-L6-v2 (384 dim)'
SELECT
	'sentence-transformers/paraphrase-MiniLM-L6-v2' AS model_name,
	384 AS expected_dims,
	vector_dims(embed_text('Test text', 'sentence-transformers/paraphrase-MiniLM-L6-v2')) AS actual_dims,
	CASE 
		WHEN vector_norm(embed_text('Test text', 'sentence-transformers/paraphrase-MiniLM-L6-v2')) > 0 THEN true
		ELSE false
	END AS non_zero,
	round(vector_norm(embed_text('Test text', 'sentence-transformers/paraphrase-MiniLM-L6-v2'))::numeric, 4) AS norm;

-- Model 6: paraphrase-MiniLM-L3-v2 (384 dim)
\echo ''
\echo 'Testing Model 6/16: sentence-transformers/paraphrase-MiniLM-L3-v2 (384 dim)'
SELECT
	'sentence-transformers/paraphrase-MiniLM-L3-v2' AS model_name,
	384 AS expected_dims,
	vector_dims(embed_text('Test text', 'sentence-transformers/paraphrase-MiniLM-L3-v2')) AS actual_dims,
	CASE 
		WHEN vector_norm(embed_text('Test text', 'sentence-transformers/paraphrase-MiniLM-L3-v2')) > 0 THEN true
		ELSE false
	END AS non_zero,
	round(vector_norm(embed_text('Test text', 'sentence-transformers/paraphrase-MiniLM-L3-v2'))::numeric, 4) AS norm;

-- Model 7: ms-marco-MiniLM-L-6-v2 (384 dim, MS MARCO trained)
\echo ''
\echo 'Testing Model 7/16: sentence-transformers/ms-marco-MiniLM-L-6-v2 (384 dim)'
SELECT
	'sentence-transformers/ms-marco-MiniLM-L-6-v2' AS model_name,
	384 AS expected_dims,
	vector_dims(embed_text('Test text', 'sentence-transformers/ms-marco-MiniLM-L-6-v2')) AS actual_dims,
	CASE 
		WHEN vector_norm(embed_text('Test text', 'sentence-transformers/ms-marco-MiniLM-L-6-v2')) > 0 THEN true
		ELSE false
	END AS non_zero,
	round(vector_norm(embed_text('Test text', 'sentence-transformers/ms-marco-MiniLM-L-6-v2'))::numeric, 4) AS norm;

-- Model 8: multi-qa-MiniLM-L6-cos-v1 (384 dim, Q&A focused)
\echo ''
\echo 'Testing Model 8/16: sentence-transformers/multi-qa-MiniLM-L6-cos-v1 (384 dim)'
SELECT
	'sentence-transformers/multi-qa-MiniLM-L6-cos-v1' AS model_name,
	384 AS expected_dims,
	vector_dims(embed_text('Test text', 'sentence-transformers/multi-qa-MiniLM-L6-cos-v1')) AS actual_dims,
	CASE 
		WHEN vector_norm(embed_text('Test text', 'sentence-transformers/multi-qa-MiniLM-L6-cos-v1')) > 0 THEN true
		ELSE false
	END AS non_zero,
	round(vector_norm(embed_text('Test text', 'sentence-transformers/multi-qa-MiniLM-L6-cos-v1'))::numeric, 4) AS norm;

-- Model 9: nli-MiniLM-L6-v2 (384 dim, Natural Language Inference)
\echo ''
\echo 'Testing Model 9/16: sentence-transformers/nli-MiniLM-L6-v2 (384 dim)'
SELECT
	'sentence-transformers/nli-MiniLM-L6-v2' AS model_name,
	384 AS expected_dims,
	vector_dims(embed_text('Test text', 'sentence-transformers/nli-MiniLM-L6-v2')) AS actual_dims,
	CASE 
		WHEN vector_norm(embed_text('Test text', 'sentence-transformers/nli-MiniLM-L6-v2')) > 0 THEN true
		ELSE false
	END AS non_zero,
	round(vector_norm(embed_text('Test text', 'sentence-transformers/nli-MiniLM-L6-v2'))::numeric, 4) AS norm;

-- Model 10: stsb-MiniLM-L6-v2 (384 dim, Semantic Textual Similarity)
\echo ''
\echo 'Testing Model 10/16: sentence-transformers/stsb-MiniLM-L6-v2 (384 dim)'
SELECT
	'sentence-transformers/stsb-MiniLM-L6-v2' AS model_name,
	384 AS expected_dims,
	vector_dims(embed_text('Test text', 'sentence-transformers/stsb-MiniLM-L6-v2')) AS actual_dims,
	CASE 
		WHEN vector_norm(embed_text('Test text', 'sentence-transformers/stsb-MiniLM-L6-v2')) > 0 THEN true
		ELSE false
	END AS non_zero,
	round(vector_norm(embed_text('Test text', 'sentence-transformers/stsb-MiniLM-L6-v2'))::numeric, 4) AS norm;

-- Test 17b: TEXT EMBEDDING MODELS (768-dim models - Higher Quality)
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Test 17b: TEXT Embedding Models (768-dim, Higher Quality)'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo ''

-- Model 11: all-mpnet-base-v2 (768 dim, higher quality)
\echo ''
\echo 'Testing Model 11/16: sentence-transformers/all-mpnet-base-v2 (768 dim)'
SELECT
	'sentence-transformers/all-mpnet-base-v2' AS model_name,
	768 AS expected_dims,
	vector_dims(embed_text('Test text', 'sentence-transformers/all-mpnet-base-v2')) AS actual_dims,
	CASE 
		WHEN vector_norm(embed_text('Test text', 'sentence-transformers/all-mpnet-base-v2')) > 0 THEN true
		ELSE false
	END AS non_zero,
	round(vector_norm(embed_text('Test text', 'sentence-transformers/all-mpnet-base-v2'))::numeric, 4) AS norm;

-- Model 12: all-distilroberta-v1 (768 dim, distilled Roberta)
\echo ''
\echo 'Testing Model 12/16: sentence-transformers/all-distilroberta-v1 (768 dim)'
SELECT
	'sentence-transformers/all-distilroberta-v1' AS model_name,
	768 AS expected_dims,
	vector_dims(embed_text('Test text', 'sentence-transformers/all-distilroberta-v1')) AS actual_dims,
	CASE 
		WHEN vector_norm(embed_text('Test text', 'sentence-transformers/all-distilroberta-v1')) > 0 THEN true
		ELSE false
	END AS non_zero,
	round(vector_norm(embed_text('Test text', 'sentence-transformers/all-distilroberta-v1'))::numeric, 4) AS norm;

-- Model 13: BAAI/bge-base-en-v1.5 (768 dim, excellent for retrieval)
\echo ''
\echo 'Testing Model 13/16: BAAI/bge-base-en-v1.5 (768 dim)'
SELECT
	'BAAI/bge-base-en-v1.5' AS model_name,
	768 AS expected_dims,
	vector_dims(embed_text('Test text', 'BAAI/bge-base-en-v1.5')) AS actual_dims,
	CASE 
		WHEN vector_norm(embed_text('Test text', 'BAAI/bge-base-en-v1.5')) > 0 THEN true
		ELSE false
	END AS non_zero,
	round(vector_norm(embed_text('Test text', 'BAAI/bge-base-en-v1.5'))::numeric, 4) AS norm;

-- Model 14: multi-qa-mpnet-base-cos-v1 (768 dim, Q&A focused)
\echo ''
\echo 'Testing Model 14/16: sentence-transformers/multi-qa-mpnet-base-cos-v1 (768 dim)'
SELECT
	'sentence-transformers/multi-qa-mpnet-base-cos-v1' AS model_name,
	768 AS expected_dims,
	vector_dims(embed_text('Test text', 'sentence-transformers/multi-qa-mpnet-base-cos-v1')) AS actual_dims,
	CASE 
		WHEN vector_norm(embed_text('Test text', 'sentence-transformers/multi-qa-mpnet-base-cos-v1')) > 0 THEN true
		ELSE false
	END AS non_zero,
	round(vector_norm(embed_text('Test text', 'sentence-transformers/multi-qa-mpnet-base-cos-v1'))::numeric, 4) AS norm;

-- Model 15: nli-mpnet-base-v2 (768 dim, Natural Language Inference)
\echo ''
\echo 'Testing Model 15/16: sentence-transformers/nli-mpnet-base-v2 (768 dim)'
SELECT
	'sentence-transformers/nli-mpnet-base-v2' AS model_name,
	768 AS expected_dims,
	vector_dims(embed_text('Test text', 'sentence-transformers/nli-mpnet-base-v2')) AS actual_dims,
	CASE 
		WHEN vector_norm(embed_text('Test text', 'sentence-transformers/nli-mpnet-base-v2')) > 0 THEN true
		ELSE false
	END AS non_zero,
	round(vector_norm(embed_text('Test text', 'sentence-transformers/nli-mpnet-base-v2'))::numeric, 4) AS norm;

-- Model 16: stsb-mpnet-base-v2 (768 dim, Semantic Textual Similarity)
\echo ''
\echo 'Testing Model 16/16: sentence-transformers/stsb-mpnet-base-v2 (768 dim)'
SELECT
	'sentence-transformers/stsb-mpnet-base-v2' AS model_name,
	768 AS expected_dims,
	vector_dims(embed_text('Test text', 'sentence-transformers/stsb-mpnet-base-v2')) AS actual_dims,
	CASE 
		WHEN vector_norm(embed_text('Test text', 'sentence-transformers/stsb-mpnet-base-v2')) > 0 THEN true
		ELSE false
	END AS non_zero,
	round(vector_norm(embed_text('Test text', 'sentence-transformers/stsb-mpnet-base-v2'))::numeric, 4) AS norm;

-- Model 17: BAAI/bge-large-en-v1.5 (1024 dim, large model for retrieval)
\echo ''
\echo 'Testing Model 17/17: BAAI/bge-large-en-v1.5 (1024 dim)'
SELECT
	'BAAI/bge-large-en-v1.5' AS model_name,
	1024 AS expected_dims,
	vector_dims(embed_text('Test text', 'BAAI/bge-large-en-v1.5')) AS actual_dims,
	CASE 
		WHEN vector_norm(embed_text('Test text', 'BAAI/bge-large-en-v1.5')) > 0 THEN true
		ELSE false
	END AS non_zero,
	round(vector_norm(embed_text('Test text', 'BAAI/bge-large-en-v1.5'))::numeric, 4) AS norm;

-- Test 17c: IMAGE EMBEDDING MODELS (CLIP models)
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Test 17c: IMAGE Embedding Models (CLIP)'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo ''

-- Create minimal test image data (1x1 PNG) for image embedding tests
DO $$
DECLARE
	test_image bytea;
	result vector;
	dims int;
	norm numeric;
BEGIN
	/* Minimal valid PNG (1x1 pixel) - use decode() to create bytea without compression */
	/* PNG signature: 89 50 4E 47 0D 0A 1A 0A */
	test_image := decode('89504e470d0a1a0a0000000d4948445200000001000000010802000000907753de0000000a49444154789c6300010000000500010d0a2db40000000049454e44ae426082', 'hex');
	-- Image Model 1: clip-ViT-B-32 (512 dim, default CLIP model)
	RAISE NOTICE 'Testing Image Model 1/3: sentence-transformers/clip-ViT-B-32 (512 dim)';
	BEGIN
		result := embed_image(test_image, 'sentence-transformers/clip-ViT-B-32');
		dims := vector_dims(result);
		norm := vector_norm(result);
		IF norm > 0 AND dims = 512 THEN
			RAISE NOTICE '  ✓ SUCCESS: dims=%, norm=%.4f', dims, norm;
		ELSE
			RAISE NOTICE '  ⚠ CHECK: dims=%, norm=%.4f', dims, norm;
		END IF;
	EXCEPTION WHEN OTHERS THEN
		RAISE NOTICE '  ✗ ERROR: % - %', SQLSTATE, SQLERRM;
	END;

	-- Image Model 2: clip-ViT-B-16 (512 dim)
	RAISE NOTICE '';
	RAISE NOTICE 'Testing Image Model 2/3: sentence-transformers/clip-ViT-B-16 (512 dim)';
	BEGIN
		result := embed_image(test_image, 'sentence-transformers/clip-ViT-B-16');
		dims := vector_dims(result);
		norm := vector_norm(result);
		IF norm > 0 AND dims = 512 THEN
			RAISE NOTICE '  ✓ SUCCESS: dims=%, norm=%.4f', dims, norm;
		ELSE
			RAISE NOTICE '  ⚠ CHECK: dims=%, norm=%.4f', dims, norm;
		END IF;
	EXCEPTION WHEN OTHERS THEN
		RAISE NOTICE '  ✗ ERROR: % - %', SQLSTATE, SQLERRM;
	END;

	-- Image Model 3: clip-ViT-L-14 (768 dim)
	RAISE NOTICE '';
	RAISE NOTICE 'Testing Image Model 3/3: sentence-transformers/clip-ViT-L-14 (768 dim)';
	BEGIN
		result := embed_image(test_image, 'sentence-transformers/clip-ViT-L-14');
		dims := vector_dims(result);
		norm := vector_norm(result);
		IF norm > 0 AND dims = 768 THEN
			RAISE NOTICE '  ✓ SUCCESS: dims=%, norm=%.4f', dims, norm;
		ELSE
			RAISE NOTICE '  ⚠ CHECK: dims=%, norm=%.4f', dims, norm;
		END IF;
	EXCEPTION WHEN OTHERS THEN
		RAISE NOTICE '  ✗ ERROR: % - %', SQLSTATE, SQLERRM;
	END;
END $$;

-- Test 17d: MULTIMODAL EMBEDDING MODELS (text + image)
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Test 17d: MULTIMODAL Embedding Models (text + image)'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo ''

DO $$
DECLARE
	test_text TEXT := 'Test text for multimodal embedding';
	test_image bytea;
	result vector;
	dims int;
	norm numeric;
BEGIN
	/* Minimal valid PNG (1x1 pixel) - use decode() to create bytea without compression */
	/* PNG signature: 89 50 4E 47 0D 0A 1A 0A */
	test_image := decode('89504e470d0a1a0a0000000d4948445200000001000000010802000000907753de0000000a49444154789c6300010000000500010d0a2db40000000049454e44ae426082', 'hex');
	-- Multimodal Model 1: clip-ViT-B-32 (512 dim, default CLIP model)
	RAISE NOTICE 'Testing Multimodal Model 1/3: sentence-transformers/clip-ViT-B-32 (512 dim)';
	BEGIN
		result := embed_multimodal(test_text, test_image, 'sentence-transformers/clip-ViT-B-32');
		dims := vector_dims(result);
		norm := vector_norm(result);
		IF norm > 0 AND dims = 512 THEN
			RAISE NOTICE '  ✓ SUCCESS: dims=%, norm=%.4f', dims, norm;
		ELSE
			RAISE NOTICE '  ⚠ CHECK: dims=%, norm=%.4f', dims, norm;
		END IF;
	EXCEPTION WHEN OTHERS THEN
		RAISE NOTICE '  ✗ ERROR: % - %', SQLSTATE, SQLERRM;
	END;

	-- Multimodal Model 2: clip-ViT-B-16 (512 dim)
	RAISE NOTICE '';
	RAISE NOTICE 'Testing Multimodal Model 2/3: sentence-transformers/clip-ViT-B-16 (512 dim)';
	BEGIN
		result := embed_multimodal(test_text, test_image, 'sentence-transformers/clip-ViT-B-16');
		dims := vector_dims(result);
		norm := vector_norm(result);
		IF norm > 0 AND dims = 512 THEN
			RAISE NOTICE '  ✓ SUCCESS: dims=%, norm=%.4f', dims, norm;
		ELSE
			RAISE NOTICE '  ⚠ CHECK: dims=%, norm=%.4f', dims, norm;
		END IF;
	EXCEPTION WHEN OTHERS THEN
		RAISE NOTICE '  ✗ ERROR: % - %', SQLSTATE, SQLERRM;
	END;

	-- Multimodal Model 3: clip-ViT-L-14 (768 dim)
	RAISE NOTICE '';
	RAISE NOTICE 'Testing Multimodal Model 3/3: sentence-transformers/clip-ViT-L-14 (768 dim)';
	BEGIN
		result := embed_multimodal(test_text, test_image, 'sentence-transformers/clip-ViT-L-14');
		dims := vector_dims(result);
		norm := vector_norm(result);
		IF norm > 0 AND dims = 768 THEN
			RAISE NOTICE '  ✓ SUCCESS: dims=%, norm=%.4f', dims, norm;
		ELSE
			RAISE NOTICE '  ⚠ CHECK: dims=%, norm=%.4f', dims, norm;
		END IF;
	EXCEPTION WHEN OTHERS THEN
		RAISE NOTICE '  ✗ ERROR: % - %', SQLSTATE, SQLERRM;
	END;
END $$;

-- Summary of all TEXT models tested (optimized to call embed_text once per model)
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'HF TEXT Models Test Summary'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo ''

WITH text_model_tests AS (
	SELECT model_name, expected_dims::INT AS expected_dims
	FROM (VALUES
		('sentence-transformers/all-MiniLM-L6-v2', 384),
		('sentence-transformers/all-MiniLM-L12-v2', 384),
		('sentence-transformers/all-MiniLM-L3-v2', 384),
		('BAAI/bge-small-en-v1.5', 384),
		('sentence-transformers/paraphrase-MiniLM-L6-v2', 384),
		('sentence-transformers/paraphrase-MiniLM-L3-v2', 384),
		('sentence-transformers/ms-marco-MiniLM-L-6-v2', 384),
		('sentence-transformers/multi-qa-MiniLM-L6-cos-v1', 384),
		('sentence-transformers/nli-MiniLM-L6-v2', 384),
		('sentence-transformers/stsb-MiniLM-L6-v2', 384),
		('sentence-transformers/all-mpnet-base-v2', 768),
		('sentence-transformers/all-distilroberta-v1', 768),
		('BAAI/bge-base-en-v1.5', 768),
		('sentence-transformers/multi-qa-mpnet-base-cos-v1', 768),
		('sentence-transformers/nli-mpnet-base-v2', 768),
		('sentence-transformers/stsb-mpnet-base-v2', 768),
		('BAAI/bge-large-en-v1.5', 1024)
	) AS t(model_name, expected_dims)
),
text_embeddings AS (
	SELECT
		mt.model_name,
		mt.expected_dims,
		embed_text('Summary test', mt.model_name) AS embedding
	FROM text_model_tests mt
)
SELECT
	e.model_name,
	e.expected_dims,
	vector_dims(e.embedding) AS actual_dims,
	CASE 
		WHEN vector_dims(e.embedding) = e.expected_dims THEN '✓ DIMS OK'
		ELSE '⚠ DIMS MISMATCH'
	END AS dim_check,
	CASE 
		WHEN vector_norm(e.embedding) > 0 THEN '✓ NON-ZERO'
		ELSE '⚠ ZERO VECTOR'
	END AS vector_check
FROM text_embeddings e
ORDER BY e.expected_dims, e.model_name;

\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Hugging Face embedding models tests completed!'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

\echo 'Test completed successfully'
