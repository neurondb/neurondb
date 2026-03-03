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
\echo 'Test Suite: Agentic RAG (Autonomous Planning and Dynamic Retrieval)'
\echo ''
\echo 'NOTE: embed() warnings are expected if LLM is not configured.'
\echo '      To generate real embeddings, configure:'
\echo '      - neurondb.llm_api_key (Hugging Face API key)'
\echo '      - Or enable GPU embedding via GUC (ALTER SYSTEM SET neurondb.compute_mode = on)'
\echo '      Without configuration, embed() returns zero vectors (graceful fallback).'
\echo ''

-- Create test table for RAG documents
DROP TABLE IF EXISTS agentic_rag_test_documents;
CREATE TEMP TABLE agentic_rag_test_documents (
	id SERIAL PRIMARY KEY,
	content TEXT NOT NULL,
	embedding VECTOR(384),
	metadata JSONB DEFAULT '{}'::jsonb
);

-- Insert sample documents
INSERT INTO agentic_rag_test_documents (content, embedding) VALUES
	('PostgreSQL is a powerful open-source relational database management system. It provides advanced features including ACID compliance, full-text search, and extensibility through extensions.', embed_text('PostgreSQL is a powerful open-source relational database management system. It provides advanced features including ACID compliance, full-text search, and extensibility through extensions.', 'all-MiniLM-L6-v2')),
	('NeuronDB extends PostgreSQL with vector search, machine learning inference, and RAG pipeline support. This enables building AI-powered applications directly within the database.', embed_text('NeuronDB extends PostgreSQL with vector search, machine learning inference, and RAG pipeline support. This enables building AI-powered applications directly within the database.', 'all-MiniLM-L6-v2')),
	('Vector search enables semantic similarity queries. HNSW indexes provide fast approximate nearest neighbor search for high-dimensional vectors.', embed_text('Vector search enables semantic similarity queries. HNSW indexes provide fast approximate nearest neighbor search for high-dimensional vectors.', 'all-MiniLM-L6-v2')),
	('Machine learning algorithms can be trained and deployed directly in PostgreSQL using NeuronDB. This includes classification, regression, clustering, and deep learning models.', embed_text('Machine learning algorithms can be trained and deployed directly in PostgreSQL using NeuronDB. This includes classification, regression, clustering, and deep learning models.', 'all-MiniLM-L6-v2')),
	('RAG (Retrieval-Augmented Generation) combines vector search with language models to provide accurate, context-aware answers grounded in retrieved documents.', embed_text('RAG (Retrieval-Augmented Generation) combines vector search with language models to provide accurate, context-aware answers grounded in retrieved documents.', 'all-MiniLM-L6-v2')),
	('Agentic RAG uses autonomous planning to break down complex queries into multiple retrieval steps. Each step verifies evidence sufficiency before proceeding.', embed_text('Agentic RAG uses autonomous planning to break down complex queries into multiple retrieval steps. Each step verifies evidence sufficiency before proceeding.', 'all-MiniLM-L6-v2'));

\echo 'Sample documents inserted'
\echo ''

-- Test 1: Basic Agentic RAG Query
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Test 1: Basic Agentic RAG Query'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

DO $$
DECLARE
	query_text text := 'What database features does PostgreSQL support and how does NeuronDB extend them?';
	agentic_result record;
	result_count int := 0;
	has_answer boolean := false;
	has_execution_trace boolean := false;
	has_reasoning_path boolean := false;
BEGIN
	-- Execute Agentic RAG query
	FOR agentic_result IN 
		SELECT * FROM neurondb.rag_agentic(
			query_text,
			'agentic_rag_test_documents',
			'embedding',
			'content',
			'default',
			'default',
			3,  -- top_k
			5,  -- max_steps
			0.7,  -- evidence_threshold
			2000,  -- max_tokens
			'{}'::jsonb  -- custom_context
		)
	LOOP
		result_count := result_count + 1;
		
		IF agentic_result.answer IS NOT NULL AND agentic_result.answer != '' THEN
			has_answer := true;
			IF result_count = 1 THEN
				RAISE NOTICE 'Answer generated: %', substring(agentic_result.answer, 1, 100) || '...';
			END IF;
		END IF;
		
		IF agentic_result.execution_trace IS NOT NULL AND jsonb_array_length(agentic_result.execution_trace) > 0 THEN
			has_execution_trace := true;
			IF result_count = 1 THEN
				RAISE NOTICE 'Execution trace: % steps', jsonb_array_length(agentic_result.execution_trace);
			END IF;
		END IF;
		
		IF agentic_result.reasoning_path IS NOT NULL AND array_length(agentic_result.reasoning_path, 1) > 0 THEN
			has_reasoning_path := true;
			IF result_count = 1 THEN
				RAISE NOTICE 'Reasoning path: %', array_to_string(agentic_result.reasoning_path, ' -> ');
			END IF;
		END IF;
		
		IF result_count = 1 THEN
			RAISE NOTICE 'Top chunk: % (relevance: %)', 
				substring(agentic_result.chunk_text, 1, 80) || '...',
				agentic_result.relevance_score;
		END IF;
	END LOOP;
	
	IF result_count = 0 THEN
		RAISE EXCEPTION 'Agentic RAG query returned no results';
	END IF;
	
	IF NOT has_answer THEN
		RAISE NOTICE '⚠ Answer generation may have failed (LLM not configured or API issue)';
		RAISE NOTICE 'Context retrieval (core functionality) succeeded with % results', result_count;
	ELSE
		RAISE NOTICE '✓ Agentic RAG query successful: % results, answer generated', result_count;
	END IF;
	
	IF has_execution_trace THEN
		RAISE NOTICE '✓ Execution trace generated successfully';
	ELSE
		RAISE NOTICE '⚠ Execution trace may not have been generated (fallback used)';
	END IF;
	
	IF has_reasoning_path THEN
		RAISE NOTICE '✓ Reasoning path generated successfully';
	ELSE
		RAISE NOTICE '⚠ Reasoning path may not have been generated (fallback used)';
	END IF;
END $$;

-- Test 2: Agentic RAG with Custom Parameters
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Test 2: Agentic RAG with Custom Parameters'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

DO $$
DECLARE
	query_text text := 'How does RAG work and what are its advantages?';
	agentic_result record;
	result_count int := 0;
BEGIN
	-- Execute Agentic RAG with custom parameters
	FOR agentic_result IN 
		SELECT * FROM neurondb.rag_agentic(
			query_text,
			'agentic_rag_test_documents',
			'embedding',
			'content',
			'default',
			'default',
			5,  -- top_k
			3,  -- max_steps (fewer steps)
			0.8,  -- higher evidence_threshold
			1500,  -- lower max_tokens
			'{"system_prompt": "You are a technical documentation assistant.", "llm_params": "{\"temperature\": 0.5, \"max_tokens\": 300}"}'::jsonb
		)
	LOOP
		result_count := result_count + 1;
	END LOOP;
	
	IF result_count = 0 THEN
		RAISE EXCEPTION 'Agentic RAG with custom parameters returned no results';
	END IF;
	
	RAISE NOTICE '✓ Agentic RAG with custom parameters successful: % results', result_count;
END $$;

-- Test 3: Agentic RAG with Complex Multi-Step Query
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Test 3: Agentic RAG with Complex Multi-Step Query'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

DO $$
DECLARE
	query_text text := 'Compare PostgreSQL features, NeuronDB extensions, and RAG capabilities. What are the key differences and use cases?';
	agentic_result record;
	result_count int := 0;
	steps_executed int;
BEGIN
	-- Execute Agentic RAG with complex query that should trigger multiple steps
	FOR agentic_result IN 
		SELECT * FROM neurondb.rag_agentic(
			query_text,
			'agentic_rag_test_documents',
			'embedding',
			'content',
			'default',
			'default',
			3,
			5,  -- allow more steps
			0.6,  -- lower threshold to allow more steps
			3000,  -- more tokens
			'{}'::jsonb
		)
	LOOP
		result_count := result_count + 1;
		
		IF result_count = 1 AND agentic_result.execution_trace IS NOT NULL THEN
			steps_executed := jsonb_array_length(agentic_result.execution_trace);
			RAISE NOTICE 'Steps executed: %', steps_executed;
		END IF;
	END LOOP;
	
	IF result_count = 0 THEN
		RAISE EXCEPTION 'Agentic RAG with complex query returned no results';
	END IF;
	
	RAISE NOTICE '✓ Agentic RAG with complex multi-step query successful: % results', result_count;
END $$;

\echo ''
\echo '=========================================================================='
\echo '✅ All Agentic RAG tests passed!'
\echo '=========================================================================='

\echo 'Test completed successfully'
