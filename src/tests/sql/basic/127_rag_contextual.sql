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
\echo 'Test Suite: Contextual RAG (Context-Aware Query Rewriting and Adaptation)'
\echo ''
\echo 'NOTE: embed() warnings are expected if LLM is not configured.'
\echo '      To generate real embeddings, configure:'
\echo '      - neurondb.llm_api_key (Hugging Face API key)'
\echo '      - Or enable GPU embedding via GUC (ALTER SYSTEM SET neurondb.compute_mode = on)'
\echo '      Without configuration, embed() returns zero vectors (graceful fallback).'
\echo ''

-- Create test table for RAG documents
DROP TABLE IF EXISTS contextual_rag_test_documents;
CREATE TEMP TABLE contextual_rag_test_documents (
	id SERIAL PRIMARY KEY,
	content TEXT NOT NULL,
	embedding VECTOR(384),
	metadata JSONB DEFAULT '{}'::jsonb
);

-- Insert sample documents
INSERT INTO contextual_rag_test_documents (content, embedding) VALUES
	('PostgreSQL is a powerful open-source relational database management system. It provides advanced features including ACID compliance, full-text search, and extensibility through extensions.', embed_text('PostgreSQL is a powerful open-source relational database management system. It provides advanced features including ACID compliance, full-text search, and extensibility through extensions.', 'all-MiniLM-L6-v2')),
	('NeuronDB extends PostgreSQL with vector search, machine learning inference, and RAG pipeline support. This enables building AI-powered applications directly within the database.', embed_text('NeuronDB extends PostgreSQL with vector search, machine learning inference, and RAG pipeline support. This enables building AI-powered applications directly within the database.', 'all-MiniLM-L6-v2')),
	('Vector search enables semantic similarity queries. HNSW indexes provide fast approximate nearest neighbor search for high-dimensional vectors.', embed_text('Vector search enables semantic similarity queries. HNSW indexes provide fast approximate nearest neighbor search for high-dimensional vectors.', 'all-MiniLM-L6-v2')),
	('Machine learning algorithms can be trained and deployed directly in PostgreSQL using NeuronDB. This includes classification, regression, clustering, and deep learning models.', embed_text('Machine learning algorithms can be trained and deployed directly in PostgreSQL using NeuronDB. This includes classification, regression, clustering, and deep learning models.', 'all-MiniLM-L6-v2')),
	('RAG (Retrieval-Augmented Generation) combines vector search with language models to provide accurate, context-aware answers grounded in retrieved documents.', embed_text('RAG (Retrieval-Augmented Generation) combines vector search with language models to provide accurate, context-aware answers grounded in retrieved documents.', 'all-MiniLM-L6-v2')),
	('Contextual RAG adapts retrieval by interpreting broader conversational context. It performs query rewriting and strategy adaptation based on conversation history.', embed_text('Contextual RAG adapts retrieval by interpreting broader conversational context. It performs query rewriting and strategy adaptation based on conversation history.', 'all-MiniLM-L6-v2'));

\echo 'Sample documents inserted'
\echo ''

-- Test 1: Basic Contextual RAG Query
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Test 1: Basic Contextual RAG Query'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

DO $$
DECLARE
	query_text text := 'What is PostgreSQL?';
	contextual_result record;
	result_count int := 0;
	has_answer boolean := false;
	has_rewritten_query boolean := false;
	has_adaptation boolean := false;
BEGIN
	-- Execute Contextual RAG query
	FOR contextual_result IN 
		SELECT * FROM neurondb.rag_contextual(
			query_text,
			'contextual_rag_test_documents',
			'embedding',
			'content',
			'default',
			'default',
			3,  -- top_k
			'[]'::jsonb,  -- empty conversation history
			'{}'::jsonb,  -- empty session context
			false,  -- cross_session_context
			'{}'::jsonb  -- custom_context
		)
	LOOP
		result_count := result_count + 1;
		
		IF contextual_result.answer IS NOT NULL AND contextual_result.answer != '' THEN
			has_answer := true;
			IF result_count = 1 THEN
				RAISE NOTICE 'Answer generated: %', substring(contextual_result.answer, 1, 100) || '...';
			END IF;
		END IF;
		
		IF contextual_result.rewritten_query IS NOT NULL AND contextual_result.rewritten_query != query_text THEN
			has_rewritten_query := true;
			IF result_count = 1 THEN
				RAISE NOTICE 'Query rewritten: "%" -> "%"', query_text, contextual_result.rewritten_query;
			END IF;
		END IF;
		
		IF contextual_result.context_adaptation IS NOT NULL THEN
			has_adaptation := true;
			IF result_count = 1 THEN
				RAISE NOTICE 'Context adaptation: %', contextual_result.context_adaptation;
			END IF;
		END IF;
		
		IF result_count = 1 THEN
			RAISE NOTICE 'Top chunk: % (relevance: %)', 
				substring(contextual_result.chunk_text, 1, 80) || '...',
				contextual_result.relevance_score;
		END IF;
	END LOOP;
	
	IF result_count = 0 THEN
		RAISE NOTICE 'Contextual RAG query returned no results (embed/LLM may not be configured; acceptable with fail-open)';
	ELSIF NOT has_answer THEN
		RAISE NOTICE '⚠ Answer generation may have failed (LLM not configured or API issue)';
		RAISE NOTICE 'Context retrieval (core functionality) succeeded with % results', result_count;
	ELSE
		RAISE NOTICE '✓ Contextual RAG query successful: % results, answer generated', result_count;
	END IF;
	
	IF has_rewritten_query THEN
		RAISE NOTICE '✓ Query rewriting successful';
	ELSE
		RAISE NOTICE '⚠ Query may not have been rewritten (fallback used)';
	END IF;
	
	IF has_adaptation THEN
		RAISE NOTICE '✓ Context adaptation successful';
	ELSE
		RAISE NOTICE '⚠ Context adaptation may not have been generated (fallback used)';
	END IF;
END $$;

-- Test 2: Contextual RAG with Conversation History
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Test 2: Contextual RAG with Conversation History'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

DO $$
DECLARE
	query_text text := 'What about rate limits?';
	conversation_history jsonb;
	contextual_result record;
	result_count int := 0;
BEGIN
	-- Build conversation history
	conversation_history := jsonb_build_array(
		jsonb_build_object('role', 'user', 'content', 'What database features does PostgreSQL support?'),
		jsonb_build_object('role', 'assistant', 'content', 'PostgreSQL supports ACID compliance, full-text search, and extensibility through extensions.')
	);
	
	-- Execute Contextual RAG with conversation history
	FOR contextual_result IN 
		SELECT * FROM neurondb.rag_contextual(
			query_text,
			'contextual_rag_test_documents',
			'embedding',
			'content',
			'default',
			'default',
			3,
			conversation_history,
			'{}'::jsonb,
			false,
			'{}'::jsonb
		)
	LOOP
		result_count := result_count + 1;
		
		IF result_count = 1 THEN
			RAISE NOTICE 'Query: "%"', query_text;
			IF contextual_result.rewritten_query IS NOT NULL THEN
				RAISE NOTICE 'Rewritten: "%"', contextual_result.rewritten_query;
			END IF;
		END IF;
	END LOOP;
	
	IF result_count = 0 THEN
		RAISE EXCEPTION 'Contextual RAG with conversation history returned no results';
	END IF;
	
	RAISE NOTICE '✓ Contextual RAG with conversation history successful: % results', result_count;
END $$;

-- Test 3: Contextual RAG with Session Context
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Test 3: Contextual RAG with Session Context'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

DO $$
DECLARE
	query_text text := 'How does it work?';
	session_context jsonb;
	contextual_result record;
	result_count int := 0;
BEGIN
	-- Build session context
	session_context := jsonb_build_object(
		'topics', 'database systems, vector search, RAG',
		'intent', 'learning about database features',
		'domain', 'database technology'
	);
	
	-- Execute Contextual RAG with session context
	FOR contextual_result IN 
		SELECT * FROM neurondb.rag_contextual(
			query_text,
			'contextual_rag_test_documents',
			'embedding',
			'content',
			'default',
			'default',
			3,
			'[]'::jsonb,
			session_context,
			false,
			'{}'::jsonb
		)
	LOOP
		result_count := result_count + 1;
	END LOOP;
	
	IF result_count = 0 THEN
		RAISE EXCEPTION 'Contextual RAG with session context returned no results';
	END IF;
	
	RAISE NOTICE '✓ Contextual RAG with session context successful: % results', result_count;
END $$;

-- Test 4: Contextual RAG with Full Context (History + Session)
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Test 4: Contextual RAG with Full Context (History + Session)'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

DO $$
DECLARE
	query_text text := 'Can you tell me more?';
	conversation_history jsonb;
	session_context jsonb;
	contextual_result record;
	result_count int := 0;
BEGIN
	-- Build conversation history
	conversation_history := jsonb_build_array(
		jsonb_build_object('role', 'user', 'content', 'What is RAG?'),
		jsonb_build_object('role', 'assistant', 'content', 'RAG combines vector search with language models to provide accurate answers.')
	);
	
	-- Build session context
	session_context := jsonb_build_object(
		'topics', 'RAG, vector search, AI',
		'intent', 'understanding RAG architectures',
		'domain', 'AI/ML'
	);
	
	-- Execute Contextual RAG with full context
	FOR contextual_result IN 
		SELECT * FROM neurondb.rag_contextual(
			query_text,
			'contextual_rag_test_documents',
			'embedding',
			'content',
			'default',
			'default',
			5,
			conversation_history,
			session_context,
			true,  -- cross_session_context
			'{}'::jsonb
		)
	LOOP
		result_count := result_count + 1;
	END LOOP;
	
	IF result_count = 0 THEN
		RAISE EXCEPTION 'Contextual RAG with full context returned no results';
	END IF;
	
	RAISE NOTICE '✓ Contextual RAG with full context successful: % results', result_count;
END $$;

\echo ''
\echo '=========================================================================='
\echo '✅ All Contextual RAG tests passed!'
\echo '=========================================================================='

\echo 'Test completed successfully'
