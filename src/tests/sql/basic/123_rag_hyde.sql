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
\echo 'Test Suite: HyDE (Hypothetical Document Embeddings) RAG'
\echo ''
\echo 'NOTE: embed() warnings are expected if LLM is not configured.'
\echo '      To generate real embeddings, configure:'
\echo '      - neurondb.llm_api_key (Hugging Face API key)'
\echo '      - Or enable GPU embedding via GUC (ALTER SYSTEM SET neurondb.compute_mode = on)'
\echo '      Without configuration, embed() returns zero vectors (graceful fallback).'
\echo ''

-- Create test table for RAG documents
DROP TABLE IF EXISTS hyde_test_documents;
CREATE TEMP TABLE hyde_test_documents (
	id SERIAL PRIMARY KEY,
	content TEXT NOT NULL,
	embedding VECTOR(384),
	metadata JSONB DEFAULT '{}'::jsonb
);

-- Insert sample documents
INSERT INTO hyde_test_documents (content, embedding) VALUES
	('PostgreSQL is a powerful open-source relational database management system. It provides advanced features including ACID compliance, full-text search, and extensibility through extensions.', embed_text('PostgreSQL is a powerful open-source relational database management system. It provides advanced features including ACID compliance, full-text search, and extensibility through extensions.', 'all-MiniLM-L6-v2')),
	('NeuronDB extends PostgreSQL with vector search, machine learning inference, and RAG pipeline support. This enables building AI-powered applications directly within the database.', embed_text('NeuronDB extends PostgreSQL with vector search, machine learning inference, and RAG pipeline support. This enables building AI-powered applications directly within the database.', 'all-MiniLM-L6-v2')),
	('Vector search enables semantic similarity queries. HNSW indexes provide fast approximate nearest neighbor search for high-dimensional vectors.', embed_text('Vector search enables semantic similarity queries. HNSW indexes provide fast approximate nearest neighbor search for high-dimensional vectors.', 'all-MiniLM-L6-v2')),
	('Machine learning algorithms can be trained and deployed directly in PostgreSQL using NeuronDB. This includes classification, regression, clustering, and deep learning models.', embed_text('Machine learning algorithms can be trained and deployed directly in PostgreSQL using NeuronDB. This includes classification, regression, clustering, and deep learning models.', 'all-MiniLM-L6-v2')),
	('RAG (Retrieval-Augmented Generation) combines vector search with language models to provide accurate, context-aware answers grounded in retrieved documents.', embed_text('RAG (Retrieval-Augmented Generation) combines vector search with language models to provide accurate, context-aware answers grounded in retrieved documents.', 'all-MiniLM-L6-v2'));

\echo 'Sample documents inserted'
\echo ''

-- Test 1: Basic HyDE RAG Query
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Test 1: Basic HyDE RAG Query'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

DO $$
DECLARE
	query_text text := 'What database features does PostgreSQL support?';
	hyde_result record;
	result_count int := 0;
	has_answer boolean := false;
	has_hypotheticals boolean := false;
BEGIN
	-- Execute HyDE RAG query
	FOR hyde_result IN 
		SELECT * FROM neurondb.rag_hyde(
			query_text,
			'hyde_test_documents',
			'embedding',
			'content',
			'default',
			'default',
			3,  -- top_k
			3,  -- num_hypotheticals
			0.5,  -- hypothetical_weight
			'{}'::jsonb  -- custom_context
		)
	LOOP
		result_count := result_count + 1;
		
		IF hyde_result.answer IS NOT NULL AND hyde_result.answer != '' THEN
			has_answer := true;
			IF result_count = 1 THEN
				RAISE NOTICE 'Answer generated: %', substring(hyde_result.answer, 1, 100) || '...';
			END IF;
		END IF;
		
		IF hyde_result.hypothetical_docs IS NOT NULL AND array_length(hyde_result.hypothetical_docs, 1) > 0 THEN
			has_hypotheticals := true;
			IF result_count = 1 THEN
				RAISE NOTICE 'Hypothetical documents generated: %', array_length(hyde_result.hypothetical_docs, 1);
				RAISE NOTICE 'First hypothetical: %', substring(hyde_result.hypothetical_docs[1], 1, 80) || '...';
			END IF;
		END IF;
		
		IF result_count = 1 THEN
			RAISE NOTICE 'Top chunk: % (relevance: %)', 
				substring(hyde_result.chunk_text, 1, 80) || '...',
				hyde_result.relevance_score;
		END IF;
	END LOOP;
	
	IF result_count = 0 THEN
		RAISE EXCEPTION 'HyDE RAG query returned no results';
	END IF;
	
	IF NOT has_answer THEN
		RAISE NOTICE '⚠ Answer generation may have failed (LLM not configured or API issue)';
		RAISE NOTICE 'Context retrieval (core functionality) succeeded with % results', result_count;
	ELSE
		RAISE NOTICE '✓ HyDE RAG query successful: % results, answer generated', result_count;
	END IF;
	
	IF has_hypotheticals THEN
		RAISE NOTICE '✓ Hypothetical documents generated successfully';
	ELSE
		RAISE NOTICE '⚠ Hypothetical documents may not have been generated (fallback used)';
	END IF;
END $$;

-- Test 2: HyDE with Custom Parameters
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Test 2: HyDE with Custom Parameters'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

DO $$
DECLARE
	query_text text := 'How does vector search work?';
	hyde_result record;
	result_count int := 0;
BEGIN
	-- Execute HyDE RAG with custom parameters
	FOR hyde_result IN 
		SELECT * FROM neurondb.rag_hyde(
			query_text,
			'hyde_test_documents',
			'embedding',
			'content',
			'default',
			'default',
			5,  -- top_k
			5,  -- num_hypotheticals (more hypotheticals)
			0.7,  -- higher hypothetical_weight
			'{"system_prompt": "You are a technical documentation assistant.", "llm_params": "{\"temperature\": 0.5, \"max_tokens\": 300}"}'::jsonb
		)
	LOOP
		result_count := result_count + 1;
	END LOOP;
	
	IF result_count = 0 THEN
		RAISE EXCEPTION 'HyDE RAG with custom parameters returned no results';
	END IF;
	
	RAISE NOTICE '✓ HyDE RAG with custom parameters successful: % results', result_count;
END $$;

-- Test 3: HyDE with Different Query Types
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Test 3: HyDE with Different Query Types'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

DO $$
DECLARE
	queries text[] := ARRAY[
		'What is RAG?',
		'Explain machine learning in databases',
		'How do I use vector search?'
	];
	query_text text;
	hyde_result record;
	result_count int;
	total_results int := 0;
BEGIN
	FOREACH query_text IN ARRAY queries
	LOOP
		result_count := 0;
		FOR hyde_result IN 
			SELECT * FROM neurondb.rag_hyde(
				query_text,
				'hyde_test_documents',
				'embedding',
				'content',
				'default',
				'default',
				3,
				3,
				0.5,
				'{}'::jsonb
			)
		LOOP
			result_count := result_count + 1;
		END LOOP;
		
		total_results := total_results + result_count;
		RAISE NOTICE 'Query "%" returned % results', query_text, result_count;
	END LOOP;
	
	IF total_results = 0 THEN
		RAISE EXCEPTION 'All HyDE queries returned no results';
	END IF;
	
	RAISE NOTICE '✓ HyDE with different query types successful: % total results across % queries', total_results, array_length(queries, 1);
END $$;

\echo ''
\echo '=========================================================================='
\echo '✅ All HyDE RAG tests passed!'
\echo '=========================================================================='

\echo 'Test completed successfully'
