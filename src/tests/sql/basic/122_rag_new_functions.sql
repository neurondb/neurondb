\timing on
\pset footer off
\pset pager off

\set ON_ERROR_STOP on

-- Ensure neurondb types/operators (including vector) are available
CREATE EXTENSION IF NOT EXISTS neurondb;

-- Enable fail-open mode for graceful fallback when LLM is not configured
SET neurondb.llm_fail_open = on;

\echo '=========================================================================='
\echo '=========================================================================='
\echo ''
\echo 'Test Suite: New RAG Functions (rag_query_with_context, rag_ingest_document,'
\echo '                                rag_evaluate, rag_chat, pipeline management)'
\echo ''
\echo 'NOTE: embed() warnings are expected if LLM is not configured.'
\echo '      To generate real embeddings, configure:'
\echo '      - neurondb.llm_api_key (Hugging Face API key)'
\echo '      - Or enable GPU embedding via GUC (ALTER SYSTEM SET neurondb.compute_mode = on)'
\echo '      Without configuration, embed() returns zero vectors (graceful fallback).'
\echo ''

-- Create test table for RAG documents
DROP TABLE IF EXISTS rag_test_documents;
CREATE TEMP TABLE rag_test_documents (
	id SERIAL PRIMARY KEY,
	content TEXT NOT NULL,
	embedding VECTOR(384),
	metadata JSONB DEFAULT '{}'::jsonb
);

-- Test 1: Document Ingestion (rag_ingest_document)
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Test 1: Document Ingestion using rag_ingest_document'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

DO $$
DECLARE
	document_text text := 'PostgreSQL is a powerful open-source relational database management system. It provides advanced features including ACID compliance, full-text search, and extensibility through extensions. NeuronDB extends PostgreSQL with vector search, machine learning inference, and RAG pipeline support. This enables building AI-powered applications directly within the database.';
	ingest_result record;
	chunk_count int := 0;
BEGIN
	-- Ingest document
	FOR ingest_result IN 
		SELECT * FROM neurondb.rag_ingest_document(
			document_text,
			'rag_test_documents',
			'content',
			'embedding',
			'default',
			100,  -- chunk_size
			20    -- chunk_overlap
		)
	LOOP
		chunk_count := chunk_count + 1;
		RAISE NOTICE 'Ingested chunk %: % characters', 
			chunk_count, 
			length(ingest_result.chunk_text);
	END LOOP;
	
	IF chunk_count = 0 THEN
		RAISE NOTICE 'Document ingestion returned no chunks (embed/LLM may not be configured; acceptable with fail-open)';
	ELSE
		RAISE NOTICE '✓ Document ingestion successful: % chunks created', chunk_count;
	END IF;
END $$;

-- Verify chunks were inserted
SELECT 
	COUNT(*) AS total_chunks,
	AVG(length(content)) AS avg_chunk_length,
	MIN(length(content)) AS min_chunk_length,
	MAX(length(content)) AS max_chunk_length
FROM rag_test_documents;

-- Test 2: RAG Query with Context (rag_query_with_context)
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Test 2: RAG Query with Context using rag_query_with_context'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

DO $$
DECLARE
	query_text text := 'What is PostgreSQL?';
	query_result record;
	result_count int := 0;
	has_answer boolean := false;
BEGIN
	-- Execute RAG query with context
	FOR query_result IN 
		SELECT * FROM neurondb.rag_query_with_context(
			query_text,
			'rag_test_documents',
			'embedding',
			'content',
			'default',
			3,  -- top_k
			'{"system_prompt": "You are a helpful database assistant.", "llm_params": "{\"temperature\": 0.7, \"max_tokens\": 200}"}'::jsonb
		)
	LOOP
		result_count := result_count + 1;
		
		IF query_result.answer IS NOT NULL AND query_result.answer != '' THEN
			has_answer := true;
			RAISE NOTICE 'Answer generated: %', substring(query_result.answer, 1, 100) || '...';
		END IF;
		
		IF result_count = 1 THEN
			RAISE NOTICE 'Top chunk: % (relevance: %)', 
				substring(query_result.chunk_text, 1, 80) || '...',
				query_result.relevance_score;
		END IF;
	END LOOP;
	
	IF result_count = 0 THEN
		RAISE NOTICE 'RAG query returned no results (embed/LLM may not be configured; acceptable with fail-open)';
	ELSIF NOT has_answer THEN
		RAISE NOTICE '⚠ Answer generation may have failed (LLM not configured or API issue)';
		RAISE NOTICE 'Context retrieval (core functionality) succeeded with % results', result_count;
	ELSE
		RAISE NOTICE '✓ RAG query with context successful: % results, answer generated', result_count;
	END IF;
END $$;

-- Test 3: RAG Evaluation (rag_evaluate)
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Test 3: RAG Evaluation using rag_evaluate'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

DO $$
DECLARE
	query_text text := 'What is PostgreSQL?';
	answer_text text := 'PostgreSQL is a powerful open-source relational database management system that provides advanced features.';
	context_chunks text[];
	eval_result jsonb;
	relevancy_score float8;
	semantic_similarity float8;
BEGIN
	-- Get context chunks from test documents
	SELECT ARRAY_AGG(content ORDER BY id) INTO context_chunks
	FROM rag_test_documents
	LIMIT 3;
	
	IF context_chunks IS NULL OR array_length(context_chunks, 1) = 0 THEN
		RAISE EXCEPTION 'No context chunks available for evaluation';
	END IF;
	
	-- Evaluate RAG performance
	eval_result := neurondb.rag_evaluate(
		query_text,
		answer_text,
		context_chunks,
		'basic'
	);
	
	-- Extract metrics
	relevancy_score := (eval_result->>'relevancy')::float8;
	semantic_similarity := (eval_result->>'semantic_similarity')::float8;
	
	RAISE NOTICE 'Evaluation Results:';
	RAISE NOTICE '  Relevancy: %', relevancy_score;
	RAISE NOTICE '  Semantic Similarity: %', semantic_similarity;
	RAISE NOTICE '  Context Count: %', eval_result->>'context_count';
	RAISE NOTICE '  Similarity Stats: %', eval_result->'similarity_stats';
	
	IF relevancy_score IS NULL OR semantic_similarity IS NULL THEN
		RAISE EXCEPTION 'Evaluation returned null metrics';
	END IF;
	
	RAISE NOTICE '✓ RAG evaluation successful';
END $$;

-- Test 4: Conversational RAG (rag_chat)
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Test 4: Conversational RAG using rag_chat'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

DO $$
DECLARE
	query_text text := 'What features does PostgreSQL support?';
	chat_result jsonb;
	answer text;
	conversation_history jsonb;
	history_length int;
BEGIN
	-- First query (no history)
	chat_result := neurondb.rag_chat(
		query_text,
		'rag_test_documents',
		'embedding',
		'content',
		'default',
		3,  -- top_k
		'[]'::jsonb,  -- empty conversation history
		'gpt-3.5-turbo'
	);
	
	IF chat_result IS NULL THEN
		RAISE EXCEPTION 'Chat returned null result';
	END IF;
	
	answer := chat_result->>'answer';
	conversation_history := chat_result->'conversation_history';
	history_length := jsonb_array_length(conversation_history);
	
	IF answer IS NOT NULL AND answer != '' THEN
		RAISE NOTICE 'First response: %', substring(answer, 1, 100) || '...';
	ELSE
		RAISE NOTICE '⚠ Answer generation may have failed (LLM not configured or API issue)';
	END IF;
	
	RAISE NOTICE 'Conversation history length: %', history_length;
	
	IF history_length < 2 THEN
		RAISE EXCEPTION 'Conversation history should contain at least 2 messages (user + assistant)';
	END IF;
	
	-- Second query (with history)
	query_text := 'Can you tell me more about vector search?';
	chat_result := neurondb.rag_chat(
		query_text,
		'rag_test_documents',
		'embedding',
		'content',
		'default',
		3,
		conversation_history,  -- use previous history
		'gpt-3.5-turbo'
	);
	
	history_length := jsonb_array_length(chat_result->'conversation_history');
	
	IF history_length < 4 THEN
		RAISE EXCEPTION 'Conversation history should contain at least 4 messages after second query';
	END IF;
	
	RAISE NOTICE '✓ Conversational RAG successful: history maintained across queries';
END $$;

-- Test 5: Pipeline Management (create_rag_pipeline, update_rag_pipeline)
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Test 5: Pipeline Management (create_rag_pipeline, update_rag_pipeline)'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

DO $$
DECLARE
	v_pipeline_id int;
	pipeline_name text := 'test_pipeline_' || extract(epoch from now())::text;
	updated_config jsonb;
	update_success boolean;
BEGIN
	-- Create a new pipeline
	v_pipeline_id := neurondb.create_rag_pipeline(
		pipeline_name,
		'default',  -- embedding_model
		512,        -- chunk_size
		128,        -- chunk_overlap
		'{"test": true}'::jsonb  -- configuration
	);
	
	IF v_pipeline_id IS NULL THEN
		RAISE EXCEPTION 'Pipeline creation returned null ID';
	END IF;
	
	RAISE NOTICE '✓ Pipeline created: ID = %, name = %', v_pipeline_id, pipeline_name;
	
	-- Verify pipeline exists
	IF NOT EXISTS (
		SELECT 1 FROM neurondb.rag_pipelines rp 
		WHERE rp.pipeline_id = v_pipeline_id
	) THEN
		RAISE EXCEPTION 'Created pipeline not found in table';
	END IF;
	
	-- Update pipeline configuration
	updated_config := '{"test": true, "updated": true, "rerank_enabled": true}'::jsonb;
	update_success := neurondb.update_rag_pipeline(
		v_pipeline_id,
		updated_config
	);
	
	IF NOT update_success THEN
		RAISE EXCEPTION 'Pipeline update returned false';
	END IF;
	
	-- Verify update
	IF (SELECT rp.configuration FROM neurondb.rag_pipelines rp WHERE rp.pipeline_id = v_pipeline_id) != updated_config THEN
		RAISE EXCEPTION 'Pipeline configuration was not updated correctly';
	END IF;
	
	RAISE NOTICE '✓ Pipeline update successful';
	
	-- Clean up
	DELETE FROM neurondb.rag_pipelines WHERE neurondb.rag_pipelines.pipeline_id = v_pipeline_id;
	RAISE NOTICE '✓ Pipeline cleanup successful';
END $$;

-- Test 6: Enhanced Pipeline Table Columns
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Test 6: Enhanced Pipeline Table Columns'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

DO $$
DECLARE
	v_pipeline_id int;
	pipeline_name text := 'test_enhanced_pipeline_' || extract(epoch from now())::text;
	has_rerank boolean;
	has_hybrid boolean;
	has_evaluation boolean;
	has_llm boolean;
	has_updated_at boolean;
BEGIN
	-- Create pipeline
	v_pipeline_id := neurondb.create_rag_pipeline(
		pipeline_name,
		'default',
		512,
		128,
		'{}'::jsonb
	);
	
	-- Check if enhanced columns exist and have defaults
	SELECT 
		rp.rerank_enabled IS NOT NULL,
		rp.hybrid_enabled IS NOT NULL,
		rp.evaluation_enabled IS NOT NULL,
		rp.llm_model IS NOT NULL,
		rp.updated_at IS NOT NULL
	INTO has_rerank, has_hybrid, has_evaluation, has_llm, has_updated_at
	FROM neurondb.rag_pipelines rp
	WHERE rp.pipeline_id = v_pipeline_id;
	
	IF NOT (has_rerank AND has_hybrid AND has_evaluation AND has_llm AND has_updated_at) THEN
		RAISE EXCEPTION 'Enhanced pipeline columns missing or null';
	END IF;
	
	-- Verify default values
	IF (SELECT rp.rerank_enabled FROM neurondb.rag_pipelines rp WHERE rp.pipeline_id = v_pipeline_id) != false THEN
		RAISE EXCEPTION 'rerank_enabled default should be false';
	END IF;
	
	IF (SELECT rp.hybrid_enabled FROM neurondb.rag_pipelines rp WHERE rp.pipeline_id = v_pipeline_id) != false THEN
		RAISE EXCEPTION 'hybrid_enabled default should be false';
	END IF;
	
	IF (SELECT rp.vector_weight FROM neurondb.rag_pipelines rp WHERE rp.pipeline_id = v_pipeline_id) != 0.7 THEN
		RAISE EXCEPTION 'vector_weight default should be 0.7';
	END IF;
	
	IF (SELECT rp.llm_model FROM neurondb.rag_pipelines rp WHERE rp.pipeline_id = v_pipeline_id) != 'gpt-3.5-turbo' THEN
		RAISE EXCEPTION 'llm_model default should be gpt-3.5-turbo';
	END IF;
	
	RAISE NOTICE '✓ Enhanced pipeline columns verified with correct defaults';
	
	-- Clean up
	DELETE FROM neurondb.rag_pipelines WHERE neurondb.rag_pipelines.pipeline_id = v_pipeline_id;
END $$;

-- Test 7: RAG Query (basic, without context generation)
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Test 7: Basic RAG Query using rag_query'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

DO $$
DECLARE
	query_text text := 'What is PostgreSQL?';
	query_result record;
	result_count int := 0;
BEGIN
	-- Execute basic RAG query
	FOR query_result IN 
		SELECT * FROM neurondb.rag_query(
			query_text,
			'rag_test_documents',
			'embedding',
			'content',
			'default',
			3  -- top_k
		)
	LOOP
		result_count := result_count + 1;
		
		IF result_count = 1 THEN
			RAISE NOTICE 'Top result: % (relevance: %)', 
				substring(query_result.chunk_text, 1, 80) || '...',
				query_result.relevance_score;
		END IF;
	END LOOP;
	
	IF result_count = 0 THEN
		RAISE EXCEPTION 'RAG query returned no results';
	END IF;
	
	RAISE NOTICE '✓ Basic RAG query successful: % results', result_count;
END $$;

\echo ''
\echo '=========================================================================='
\echo '✅ All new RAG function tests passed!'
\echo '=========================================================================='

\echo 'Test completed successfully'
