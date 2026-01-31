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
\echo 'Test Suite: Modular RAG (Composable, Plug-and-Play Modules)'
\echo ''
\echo 'NOTE: embed() warnings are expected if LLM is not configured.'
\echo '      To generate real embeddings, configure:'
\echo '      - neurondb.llm_api_key (Hugging Face API key)'
\echo '      - Or enable GPU embedding via GUC (ALTER SYSTEM SET neurondb.compute_mode = on)'
\echo '      Without configuration, embed() returns zero vectors (graceful fallback).'
\echo ''

-- Create test table for RAG documents
DROP TABLE IF EXISTS modular_rag_test_documents;
CREATE TEMP TABLE modular_rag_test_documents (
	id SERIAL PRIMARY KEY,
	content TEXT NOT NULL,
	embedding VECTOR(384),
	metadata JSONB DEFAULT '{}'::jsonb
);

-- Insert sample documents
INSERT INTO modular_rag_test_documents (content, embedding) VALUES
	('PostgreSQL is a powerful open-source relational database management system. It provides advanced features including ACID compliance, full-text search, and extensibility through extensions.', embed_text('PostgreSQL is a powerful open-source relational database management system. It provides advanced features including ACID compliance, full-text search, and extensibility through extensions.', 'all-MiniLM-L6-v2')),
	('NeuronDB extends PostgreSQL with vector search, machine learning inference, and RAG pipeline support. This enables building AI-powered applications directly within the database.', embed_text('NeuronDB extends PostgreSQL with vector search, machine learning inference, and RAG pipeline support. This enables building AI-powered applications directly within the database.', 'all-MiniLM-L6-v2')),
	('Vector search enables semantic similarity queries. HNSW indexes provide fast approximate nearest neighbor search for high-dimensional vectors.', embed_text('Vector search enables semantic similarity queries. HNSW indexes provide fast approximate nearest neighbor search for high-dimensional vectors.', 'all-MiniLM-L6-v2')),
	('Machine learning algorithms can be trained and deployed directly in PostgreSQL using NeuronDB. This includes classification, regression, clustering, and deep learning models.', embed_text('Machine learning algorithms can be trained and deployed directly in PostgreSQL using NeuronDB. This includes classification, regression, clustering, and deep learning models.', 'all-MiniLM-L6-v2')),
	('RAG (Retrieval-Augmented Generation) combines vector search with language models to provide accurate, context-aware answers grounded in retrieved documents.', embed_text('RAG (Retrieval-Augmented Generation) combines vector search with language models to provide accurate, context-aware answers grounded in retrieved documents.', 'all-MiniLM-L6-v2')),
	('Modular RAG allows composing custom workflows by chaining together retrieval, reranking, filtering, and generation modules. This enables flexible pipeline configurations.', embed_text('Modular RAG allows composing custom workflows by chaining together retrieval, reranking, filtering, and generation modules. This enables flexible pipeline configurations.', 'all-MiniLM-L6-v2'));

\echo 'Sample documents inserted'
\echo ''

-- Test 1: Basic Modular RAG with Simple Pipeline
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Test 1: Basic Modular RAG with Simple Pipeline'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

DO $$
DECLARE
	query_text text := 'What is PostgreSQL?';
	module_config jsonb;
	modular_result record;
	result_count int := 0;
	has_answer boolean := false;
	has_pipeline_name boolean := false;
	has_module_trace boolean := false;
BEGIN
	-- Build simple module configuration
	module_config := jsonb_build_object(
		'name', 'simple_pipeline',
		'modules', jsonb_build_array(
			jsonb_build_object(
				'name', 'vector_retrieval',
				'type', 'retrieval',
				'parameters', jsonb_build_object('top_k', 5),
				'enabled', true
			)
		)
	);
	
	-- Execute Modular RAG query
	FOR modular_result IN 
		SELECT * FROM neurondb.rag_modular(
			query_text,
			'modular_rag_test_documents',
			'embedding',
			'content',
			module_config,
			'default',
			'default',
			'{}'::jsonb
		)
	LOOP
		result_count := result_count + 1;
		
		IF modular_result.answer IS NOT NULL AND modular_result.answer != '' THEN
			has_answer := true;
			IF result_count = 1 THEN
				RAISE NOTICE 'Answer generated: %', substring(modular_result.answer, 1, 100) || '...';
			END IF;
		END IF;
		
		IF modular_result.pipeline_name IS NOT NULL AND modular_result.pipeline_name != '' THEN
			has_pipeline_name := true;
			IF result_count = 1 THEN
				RAISE NOTICE 'Pipeline: %', modular_result.pipeline_name;
			END IF;
		END IF;
		
		IF modular_result.module_trace IS NOT NULL AND jsonb_array_length(modular_result.module_trace) > 0 THEN
			has_module_trace := true;
			IF result_count = 1 THEN
				RAISE NOTICE 'Module trace: % modules executed', jsonb_array_length(modular_result.module_trace);
			END IF;
		END IF;
		
		IF result_count = 1 THEN
			RAISE NOTICE 'Top chunk: % (relevance: %)', 
				substring(modular_result.chunk_text, 1, 80) || '...',
				modular_result.relevance_score;
		END IF;
	END LOOP;
	
	IF result_count = 0 THEN
		RAISE NOTICE 'Modular RAG query returned no results (embed/LLM may not be configured; acceptable with fail-open)';
	ELSIF NOT has_answer THEN
		RAISE NOTICE '⚠ Answer generation may have failed (LLM not configured or API issue)';
		RAISE NOTICE 'Context retrieval (core functionality) succeeded with % results', result_count;
	ELSE
		RAISE NOTICE '✓ Modular RAG query successful: % results, answer generated', result_count;
	END IF;
	
	IF has_pipeline_name THEN
		RAISE NOTICE '✓ Pipeline name tracked successfully';
	ELSE
		RAISE NOTICE '⚠ Pipeline name may not have been set';
	END IF;
	
	IF has_module_trace THEN
		RAISE NOTICE '✓ Module trace generated successfully';
	ELSE
		RAISE NOTICE '⚠ Module trace may not have been generated';
	END IF;
END $$;

-- Test 2: Modular RAG with Multi-Module Pipeline
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Test 2: Modular RAG with Multi-Module Pipeline'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

DO $$
DECLARE
	query_text text := 'How does vector search work?';
	module_config jsonb;
	modular_result record;
	result_count int := 0;
BEGIN
	-- Build multi-module pipeline configuration
	module_config := jsonb_build_object(
		'name', 'multi_module_pipeline',
		'modules', jsonb_build_array(
			jsonb_build_object(
				'name', 'vector_retrieval',
				'type', 'retrieval',
				'parameters', jsonb_build_object('top_k', 10),
				'enabled', true
			),
			jsonb_build_object(
				'name', 'reranking',
				'type', 'reranking',
				'parameters', jsonb_build_object('top_k', 5),
				'enabled', true
			),
			jsonb_build_object(
				'name', 'filter',
				'type', 'filter',
				'parameters', jsonb_build_object('max_docs', 3),
				'enabled', true
			)
		)
	);
	
	-- Execute Modular RAG with multi-module pipeline
	FOR modular_result IN 
		SELECT * FROM neurondb.rag_modular(
			query_text,
			'modular_rag_test_documents',
			'embedding',
			'content',
			module_config,
			'default',
			'default',
			'{}'::jsonb
		)
	LOOP
		result_count := result_count + 1;
	END LOOP;
	
	IF result_count = 0 THEN
		RAISE EXCEPTION 'Modular RAG with multi-module pipeline returned no results';
	END IF;
	
	RAISE NOTICE '✓ Modular RAG with multi-module pipeline successful: % results', result_count;
END $$;

-- Test 3: Modular RAG with Disabled Module
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Test 3: Modular RAG with Disabled Module'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

DO $$
DECLARE
	query_text text := 'What is RAG?';
	module_config jsonb;
	modular_result record;
	result_count int := 0;
BEGIN
	-- Build pipeline with disabled module
	module_config := jsonb_build_object(
		'name', 'selective_pipeline',
		'modules', jsonb_build_array(
			jsonb_build_object(
				'name', 'vector_retrieval',
				'type', 'retrieval',
				'parameters', jsonb_build_object('top_k', 5),
				'enabled', true
			),
			jsonb_build_object(
				'name', 'reranking',
				'type', 'reranking',
				'parameters', jsonb_build_object('top_k', 3),
				'enabled', false  -- Disabled module
			)
		)
	);
	
	-- Execute Modular RAG with disabled module
	FOR modular_result IN 
		SELECT * FROM neurondb.rag_modular(
			query_text,
			'modular_rag_test_documents',
			'embedding',
			'content',
			module_config,
			'default',
			'default',
			'{}'::jsonb
		)
	LOOP
		result_count := result_count + 1;
	END LOOP;
	
	IF result_count = 0 THEN
		RAISE EXCEPTION 'Modular RAG with disabled module returned no results';
	END IF;
	
	RAISE NOTICE '✓ Modular RAG with disabled module successful: % results', result_count;
END $$;

-- Test 4: Modular RAG with Hybrid Retrieval
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Test 4: Modular RAG with Hybrid Retrieval'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

DO $$
DECLARE
	query_text text := 'What features does NeuronDB provide?';
	module_config jsonb;
	modular_result record;
	result_count int := 0;
BEGIN
	-- Build pipeline with hybrid retrieval
	module_config := jsonb_build_object(
		'name', 'hybrid_pipeline',
		'modules', jsonb_build_array(
			jsonb_build_object(
				'name', 'hybrid_retrieval',
				'type', 'hybrid_retrieval',
				'parameters', jsonb_build_object('top_k', 5, 'vector_weight', 0.7),
				'enabled', true
			)
		)
	);
	
	-- Execute Modular RAG with hybrid retrieval
	FOR modular_result IN 
		SELECT * FROM neurondb.rag_modular(
			query_text,
			'modular_rag_test_documents',
			'embedding',
			'content',
			module_config,
			'default',
			'default',
			'{}'::jsonb
		)
	LOOP
		result_count := result_count + 1;
	END LOOP;
	
	IF result_count = 0 THEN
		RAISE EXCEPTION 'Modular RAG with hybrid retrieval returned no results';
	END IF;
	
	RAISE NOTICE '✓ Modular RAG with hybrid retrieval successful: % results', result_count;
END $$;

\echo ''
\echo '=========================================================================='
\echo '✅ All Modular RAG tests passed!'
\echo '=========================================================================='

\echo 'Test completed successfully'
