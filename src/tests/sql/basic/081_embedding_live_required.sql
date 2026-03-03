\timing on
\pset footer off
\pset pager off

\set ON_ERROR_STOP on

\echo '=========================================================================='
\echo 'Live-Required Embedding Quality Tests'
\echo '=========================================================================='
\echo ''
\echo 'This test requires neurondb.llm_api_key to be configured (Hugging Face API).'
\echo 'Without it, this test will FAIL with a clear error message.'
\echo ''

-- Fail fast if API key is not configured
DO $$
DECLARE
	api_key text;
BEGIN
	api_key := current_setting('neurondb.llm_api_key', true);
	IF api_key IS NULL OR api_key = '' THEN
		RAISE EXCEPTION 'This test requires neurondb.llm_api_key to be set. Please configure your Hugging Face API key: SET neurondb.llm_api_key = ''your-key-here'';';
	END IF;
END $$;

-- Force live behavior (no fail-open fallback)
SET neurondb.llm_fail_open = off;
SET neurondb.llm_provider = 'huggingface';

-- Test 1: Embedding dimensions for all-MiniLM-L6-v2
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Test 1: Embedding dimensions (expect 384 for all-MiniLM-L6-v2)'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

DO $$
DECLARE
	test_text text := 'This is a test document for embedding dimensions.';
	emb vector;
	dims int;
	model_name text := 'sentence-transformers/all-MiniLM-L6-v2';
BEGIN
	emb := embed_text(test_text, model_name);
	dims := vector_dims(emb);
	
	IF dims != 384 THEN
		RAISE EXCEPTION 'Expected embedding dimension 384 for model %, but got %', model_name, dims;
	END IF;
	
	RAISE NOTICE '✓ Embedding dimensions correct: %', dims;
END $$;

-- Test 2: Non-zero embedding norm for non-empty text
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Test 2: Non-zero embedding norm'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

DO $$
DECLARE
	test_text text := 'PostgreSQL is a powerful database system with vector search capabilities.';
	emb vector;
	norm_val float8;
BEGIN
	emb := embed_text(test_text);
	norm_val := vector_norm(emb);
	
	IF norm_val = 0.0 OR norm_val IS NULL THEN
		RAISE EXCEPTION 'Expected non-zero embedding norm, but got %', norm_val;
	END IF;
	
	RAISE NOTICE '✓ Embedding norm is non-zero: %', norm_val;
END $$;

-- Test 3: Semantic ordering (distance to relevant doc < distance to irrelevant doc)
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Test 3: Semantic ordering (relevant documents should be closer)'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

DO $$
DECLARE
	query_text text := 'machine learning algorithms';
	relevant_doc text := 'Machine learning models can be trained in SQL using neural networks and gradient descent.';
	irrelevant_doc text := 'The weather forecast for tomorrow shows sunny skies with a high of 75 degrees.';
	query_emb vector;
	relevant_emb vector;
	irrelevant_emb vector;
	relevant_dist float8;
	irrelevant_dist float8;
BEGIN
	-- Generate embeddings
	query_emb := embed_text(query_text);
	relevant_emb := embed_text(relevant_doc);
	irrelevant_emb := embed_text(irrelevant_doc);
	
	-- Calculate distances (cosine distance: 1 - cosine similarity)
	relevant_dist := query_emb <-> relevant_emb;
	irrelevant_dist := query_emb <-> irrelevant_emb;
	
	IF relevant_dist >= irrelevant_dist THEN
		RAISE EXCEPTION 'Semantic ordering failed: relevant distance (%) should be < irrelevant distance (%), but it is not', 
			relevant_dist, irrelevant_dist;
	END IF;
	
	RAISE NOTICE '✓ Semantic ordering correct: relevant_dist=%, irrelevant_dist=%', relevant_dist, irrelevant_dist;
END $$;

-- Test 4: Batch embedding consistency (same text should produce same embedding)
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Test 4: Batch embedding consistency'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

DO $$
DECLARE
	test_texts text[] := ARRAY[
		'Database systems are fundamental to modern computing.',
		'Vector search enables semantic similarity queries.',
		'Machine learning models require training data.'
	];
	embeddings vector[];
	emb1 vector;
	emb2 vector;
	dist float8;
BEGIN
	-- Get batch embeddings
	embeddings := embed_text_batch(test_texts);
	
	-- Verify we got the right number of embeddings
	IF array_length(embeddings, 1) != array_length(test_texts, 1) THEN
		RAISE EXCEPTION 'Expected % embeddings from batch, but got %', 
			array_length(test_texts, 1), array_length(embeddings, 1);
	END IF;
	
	-- Verify each embedding is non-NULL and has correct dimensions
	FOR i IN 1..array_length(embeddings, 1) LOOP
		IF embeddings[i] IS NULL THEN
			RAISE EXCEPTION 'Embedding at index % is NULL', i;
		END IF;
		
		IF vector_dims(embeddings[i]) != 384 THEN
			RAISE EXCEPTION 'Embedding at index % has wrong dimensions: expected 384, got %', 
				i, vector_dims(embeddings[i]);
		END IF;
	END LOOP;
	
	-- Verify same text produces same embedding (within floating point tolerance)
	emb1 := embed_text(test_texts[1]);
	emb2 := embeddings[1];
	dist := emb1 <-> emb2;
	
	IF dist > 0.0001 THEN  -- Very small tolerance for floating point differences
		RAISE EXCEPTION 'Batch embedding consistency failed: same text produced different embeddings (distance=%)', dist;
	END IF;
	
	RAISE NOTICE '✓ Batch embedding consistency verified: % embeddings with correct dimensions', array_length(embeddings, 1);
END $$;

-- Test 5: Different texts produce different embeddings
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Test 5: Different texts produce different embeddings'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

DO $$
DECLARE
	text1 text := 'Machine learning is a subset of artificial intelligence.';
	text2 text := 'The capital of France is Paris, a beautiful European city.';
	emb1 vector;
	emb2 vector;
	dist float8;
BEGIN
	emb1 := embed_text(text1);
	emb2 := embed_text(text2);
	dist := emb1 <-> emb2;
	
	-- Different texts should have meaningful distance (not nearly identical)
	IF dist < 0.01 THEN
		RAISE EXCEPTION 'Different texts produced nearly identical embeddings (distance=%), which suggests a problem', dist;
	END IF;
	
	RAISE NOTICE '✓ Different texts produce distinct embeddings (distance=%)', dist;
END $$;

\echo ''
\echo '=========================================================================='
\echo '✅ All embedding quality tests passed!'
\echo '=========================================================================='

\echo 'Test completed successfully'



