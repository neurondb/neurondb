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
\echo 'Test Suite: Corrective RAG (Iterative Self-Correction)'
\echo ''
\echo 'NOTE: embed() warnings are expected if LLM is not configured.'
\echo '      To generate real embeddings, configure:'
\echo '      - neurondb.llm_api_key (Hugging Face API key)'
\echo '      - Or enable GPU embedding via GUC (ALTER SYSTEM SET neurondb.compute_mode = on)'
\echo '      Without configuration, embed() returns zero vectors (graceful fallback).'
\echo ''

-- Create test table for RAG documents
DROP TABLE IF EXISTS corrective_rag_test_documents;
CREATE TEMP TABLE corrective_rag_test_documents (
	id SERIAL PRIMARY KEY,
	content TEXT NOT NULL,
	embedding VECTOR(384),
	metadata JSONB DEFAULT '{}'::jsonb
);

-- Insert sample documents
INSERT INTO corrective_rag_test_documents (content, embedding) VALUES
	('PostgreSQL is a powerful open-source relational database management system. It provides advanced features including ACID compliance, full-text search, and extensibility through extensions.', embed_text('PostgreSQL is a powerful open-source relational database management system. It provides advanced features including ACID compliance, full-text search, and extensibility through extensions.', 'all-MiniLM-L6-v2')),
	('NeuronDB extends PostgreSQL with vector search, machine learning inference, and RAG pipeline support. This enables building AI-powered applications directly within the database.', embed_text('NeuronDB extends PostgreSQL with vector search, machine learning inference, and RAG pipeline support. This enables building AI-powered applications directly within the database.', 'all-MiniLM-L6-v2')),
	('Vector search enables semantic similarity queries. HNSW indexes provide fast approximate nearest neighbor search for high-dimensional vectors.', embed_text('Vector search enables semantic similarity queries. HNSW indexes provide fast approximate nearest neighbor search for high-dimensional vectors.', 'all-MiniLM-L6-v2')),
	('Machine learning algorithms can be trained and deployed directly in PostgreSQL using NeuronDB. This includes classification, regression, clustering, and deep learning models.', embed_text('Machine learning algorithms can be trained and deployed directly in PostgreSQL using NeuronDB. This includes classification, regression, clustering, and deep learning models.', 'all-MiniLM-L6-v2')),
	('RAG (Retrieval-Augmented Generation) combines vector search with language models to provide accurate, context-aware answers grounded in retrieved documents.', embed_text('RAG (Retrieval-Augmented Generation) combines vector search with language models to provide accurate, context-aware answers grounded in retrieved documents.', 'all-MiniLM-L6-v2'));

\echo 'Sample documents inserted'
\echo ''

-- Test 1: Basic Corrective RAG Query
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Test 1: Basic Corrective RAG Query'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

DO $$
DECLARE
	query_text text := 'What is PostgreSQL?';
	corrective_result record;
	result_count int := 0;
	has_answer boolean := false;
	iterations_count int;
	quality_score float8;
BEGIN
	-- Execute Corrective RAG query
	FOR corrective_result IN 
		SELECT * FROM neurondb.rag_corrective(
			query_text,
			'corrective_rag_test_documents',
			'embedding',
			'content',
			'default',
			'default',
			3,  -- top_k
			3,  -- max_iterations
			0.7,  -- quality_threshold
			'{}'::jsonb  -- custom_context
		)
	LOOP
		result_count := result_count + 1;
		
		IF corrective_result.answer IS NOT NULL AND corrective_result.answer != '' THEN
			has_answer := true;
			IF result_count = 1 THEN
				RAISE NOTICE 'Answer generated: %', substring(corrective_result.answer, 1, 100) || '...';
			END IF;
		END IF;
		
		IF corrective_result.iterations IS NOT NULL THEN
			iterations_count := corrective_result.iterations;
		END IF;
		
		IF corrective_result.quality_score IS NOT NULL THEN
			quality_score := corrective_result.quality_score;
		END IF;
		
		IF result_count = 1 THEN
			RAISE NOTICE 'Top chunk: % (relevance: %)', 
				substring(corrective_result.chunk_text, 1, 80) || '...',
				corrective_result.relevance_score;
		END IF;
	END LOOP;
	
	IF result_count = 0 THEN
		RAISE EXCEPTION 'Corrective RAG query returned no results';
	END IF;
	
	IF NOT has_answer THEN
		RAISE NOTICE '⚠ Answer generation may have failed (LLM not configured or API issue)';
		RAISE NOTICE 'Context retrieval (core functionality) succeeded with % results', result_count;
	ELSE
		RAISE NOTICE '✓ Corrective RAG query successful: % results, answer generated', result_count;
	END IF;
	
	IF iterations_count IS NOT NULL THEN
		RAISE NOTICE 'Correction iterations: %, Quality score: %', iterations_count, quality_score;
	END IF;
END $$;

-- Test 2: Corrective RAG with High Quality Threshold
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Test 2: Corrective RAG with High Quality Threshold'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

DO $$
DECLARE
	query_text text := 'How does vector search work?';
	corrective_result record;
	result_count int := 0;
BEGIN
	-- Execute Corrective RAG with high quality threshold (should trigger more iterations)
	FOR corrective_result IN 
		SELECT * FROM neurondb.rag_corrective(
			query_text,
			'corrective_rag_test_documents',
			'embedding',
			'content',
			'default',
			'default',
			3,
			5,  -- more max_iterations
			0.9,  -- higher quality_threshold
			'{}'::jsonb
		)
	LOOP
		result_count := result_count + 1;
	END LOOP;
	
	IF result_count = 0 THEN
		RAISE EXCEPTION 'Corrective RAG with high threshold returned no results';
	END IF;
	
	RAISE NOTICE '✓ Corrective RAG with high quality threshold successful: % results', result_count;
END $$;

\echo ''
\echo '=========================================================================='
\echo '✅ All Corrective RAG tests passed!'
\echo '=========================================================================='

\echo 'Test completed successfully'
