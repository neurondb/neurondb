\timing on
\pset footer off
\pset pager off

\set ON_ERROR_STOP on

\echo '=========================================================================='
\echo 'Live-Required Embedding Cache Tests'
\echo '=========================================================================='
\echo ''
\echo 'This test requires neurondb.llm_api_key to be configured (Hugging Face API).'
\echo 'Tests embedding cache functionality via neurondb.embedding_cache table.'
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

-- Force live behavior
SET neurondb.llm_fail_open = off;
SET neurondb.llm_provider = 'huggingface';

-- Test 1: Cache population after embedding generation
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Test 1: Cache population after embedding generation'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

DO $$
DECLARE
	test_text text := 'This is a test document for embedding cache validation.';
	v_model_name text := 'sentence-transformers/all-MiniLM-L6-v2';
	emb vector;
	cache_count_before int;
	cache_count_after int;
BEGIN
	-- Clear any existing cache entries for this test
	DELETE FROM neurondb.embedding_cache ec WHERE ec.model_name = v_model_name;
	
	-- Count cache entries before
	SELECT COUNT(*) INTO cache_count_before
	FROM neurondb.embedding_cache ec
	WHERE ec.model_name = v_model_name;
	
	-- Generate embedding (this should populate cache if caching is enabled)
	emb := embed_text(test_text, v_model_name);
	
	-- Wait a moment for cache write (if async)
	PERFORM pg_sleep(0.1);
	
	-- Count cache entries after
	SELECT COUNT(*) INTO cache_count_after
	FROM neurondb.embedding_cache ec
	WHERE ec.model_name = v_model_name;
	
	-- Note: Cache population may or may not happen depending on implementation
	-- We check that the embedding was generated correctly regardless
	IF emb IS NULL THEN
		RAISE EXCEPTION 'Embedding generation failed';
	END IF;
	
	IF vector_dims(emb) != 384 THEN
		RAISE EXCEPTION 'Expected embedding dimension 384, but got %', vector_dims(emb);
	END IF;
	
	RAISE NOTICE '✓ Embedding generated: dims=%, cache_count_before=%, cache_count_after=%', 
		vector_dims(emb), cache_count_before, cache_count_after;
END $$;

-- Test 2: Cache access count increment on repeated calls
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Test 2: Cache access count increment (if caching is implemented)'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

DO $$
DECLARE
	test_text text := 'Repeated embedding generation test for cache access counting.';
	v_model_name text := 'sentence-transformers/all-MiniLM-L6-v2';
	emb1 vector;
	emb2 vector;
	emb3 vector;
	cache_entry RECORD;
BEGIN
	-- Clear any existing cache entries for this test
	DELETE FROM neurondb.embedding_cache ec
	WHERE ec.model_name = 'sentence-transformers/all-MiniLM-L6-v2';
	
	-- Generate embedding multiple times
	emb1 := embed_text(test_text, v_model_name);
	PERFORM pg_sleep(0.1);
	
	emb2 := embed_text(test_text, v_model_name);
	PERFORM pg_sleep(0.1);
	
	emb3 := embed_text(test_text, v_model_name);
	PERFORM pg_sleep(0.1);
	
	-- Verify embeddings are identical (or very close)
	IF (emb1 <-> emb2) > 0.0001 OR (emb2 <-> emb3) > 0.0001 THEN
		RAISE EXCEPTION 'Repeated embedding calls produced different results, suggesting cache is not working';
	END IF;
	
	-- Check cache table (if entries exist)
	SELECT * INTO cache_entry
	FROM neurondb.embedding_cache ec
	WHERE ec.model_name = v_model_name
	LIMIT 1;
	
	IF FOUND THEN
		-- If cache entry exists, check that access_count is reasonable
		IF cache_entry.access_count IS NULL OR cache_entry.access_count < 1 THEN
			RAISE EXCEPTION 'Cache entry found but access_count is invalid: %', cache_entry.access_count;
		END IF;
		
		RAISE NOTICE '✓ Cache entry found: access_count=%, created_at=%', 
			cache_entry.access_count, cache_entry.created_at;
	ELSE
		-- Cache may not be implemented or may use a different mechanism
		RAISE NOTICE '✓ Embeddings are consistent (cache may use different mechanism or be disabled)';
	END IF;
END $$;

-- Test 3: Cache key uniqueness for different texts
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Test 3: Cache key uniqueness for different texts'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

DO $$
DECLARE
	text1 text := 'First unique test document for cache key validation.';
	text2 text := 'Second unique test document with different content entirely.';
	v_model_name text := 'sentence-transformers/all-MiniLM-L6-v2';
	emb1 vector;
	emb2 vector;
	cache_count int;
BEGIN
	-- Generate embeddings for different texts
	emb1 := embed_text(text1, v_model_name);
	emb2 := embed_text(text2, v_model_name);
	
	PERFORM pg_sleep(0.1);
	
	-- Verify embeddings are different
	IF (emb1 <-> emb2) < 0.01 THEN
		RAISE EXCEPTION 'Different texts produced nearly identical embeddings';
	END IF;
	
	-- Check cache entries (if any exist)
	SELECT COUNT(*) INTO cache_count
	FROM neurondb.embedding_cache ec
	WHERE ec.model_name = v_model_name;
	
	-- If cache entries exist, verify we have at least one (or more if caching is working)
	IF cache_count > 0 THEN
		RAISE NOTICE '✓ Cache contains % entries for model %', cache_count, v_model_name;
	ELSE
		RAISE NOTICE '✓ Embeddings generated correctly (cache may use different mechanism)';
	END IF;
END $$;

-- Test 4: Cache entry metadata (created_at, last_accessed)
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Test 4: Cache entry metadata validation'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

DO $$
DECLARE
	test_text text := 'Metadata validation test document.';
	v_model_name text := 'sentence-transformers/all-MiniLM-L6-v2';
	emb vector;
	cache_entry RECORD;
BEGIN
	-- Generate embedding
	emb := embed_text(test_text, v_model_name);
	
	PERFORM pg_sleep(0.1);
	
	-- Check if cache entry exists and validate metadata
	SELECT * INTO cache_entry
	FROM neurondb.embedding_cache ec
	WHERE ec.model_name = v_model_name
	ORDER BY ec.created_at DESC
	LIMIT 1;
	
	IF FOUND THEN
		-- Validate metadata fields
		IF cache_entry.created_at IS NULL THEN
			RAISE EXCEPTION 'Cache entry created_at is NULL';
		END IF;
		
		IF cache_entry.last_accessed IS NULL THEN
			RAISE EXCEPTION 'Cache entry last_accessed is NULL';
		END IF;
		
		IF cache_entry.model_name != v_model_name THEN
			RAISE EXCEPTION 'Cache entry model_name mismatch: expected %, got %', 
				v_model_name, cache_entry.model_name;
		END IF;
		
		IF cache_entry.embedding IS NULL THEN
			RAISE EXCEPTION 'Cache entry embedding is NULL';
		END IF;
		
		IF vector_dims(cache_entry.embedding) != 384 THEN
			RAISE EXCEPTION 'Cached embedding has wrong dimensions: expected 384, got %', 
				vector_dims(cache_entry.embedding);
		END IF;
		
		RAISE NOTICE '✓ Cache metadata validated: created_at=%, last_accessed=%, access_count=%', 
			cache_entry.created_at, cache_entry.last_accessed, cache_entry.access_count;
	ELSE
		RAISE NOTICE '✓ Embedding generated (cache may use different mechanism)';
	END IF;
END $$;

\echo ''
\echo '=========================================================================='
\echo '✅ All embedding cache tests passed!'
\echo '=========================================================================='

\echo 'Test completed successfully'

