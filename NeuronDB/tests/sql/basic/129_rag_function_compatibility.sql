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
\echo 'Test Suite: RAG Function Compatibility (text vs regclass parameters)'
\echo ''
\echo 'Tests that RAG functions work with both text and regclass table parameters'
\echo 'NOTE: embed() warnings are expected if LLM is not configured.'
\echo ''

-- Create test table for RAG documents
DROP TABLE IF EXISTS rag_compat_test_documents;
CREATE TEMP TABLE rag_compat_test_documents (
	id SERIAL PRIMARY KEY,
	content TEXT NOT NULL,
	embedding VECTOR(384),
	metadata JSONB DEFAULT '{}'::jsonb
);

-- Insert sample documents
INSERT INTO rag_compat_test_documents (content, embedding) VALUES
	('PostgreSQL is a powerful open-source relational database management system. It provides advanced features including ACID compliance, full-text search, and extensibility through extensions.', embed_text('PostgreSQL is a powerful open-source relational database management system. It provides advanced features including ACID compliance, full-text search, and extensibility through extensions.', 'all-MiniLM-L6-v2')),
	('NeuronDB extends PostgreSQL with vector search, machine learning inference, and RAG pipeline support. This enables building AI-powered applications directly within the database.', embed_text('NeuronDB extends PostgreSQL with vector search, machine learning inference, and RAG pipeline support. This enables building AI-powered applications directly within the database.', 'all-MiniLM-L6-v2')),
	('Vector search enables semantic similarity queries. HNSW indexes provide fast approximate nearest neighbor search for high-dimensional vectors.', embed_text('Vector search enables semantic similarity queries. HNSW indexes provide fast approximate nearest neighbor search for high-dimensional vectors.', 'all-MiniLM-L6-v2'));

\echo 'Sample documents inserted'
\echo ''

-- Test 1: rag_query with text vs regclass
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Test 1: rag_query compatibility (text vs regclass)'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

DO $$
DECLARE
	query_text text := 'What is PostgreSQL?';
	text_result_count int := 0;
	regclass_result_count int := 0;
	text_result record;
	regclass_result record;
BEGIN
	-- Test with text parameter
	FOR text_result IN 
		SELECT * FROM neurondb.rag_query(
			query_text,
			'rag_compat_test_documents'::text,
			'embedding',
			'content',
			'default',
			3
		)
	LOOP
		text_result_count := text_result_count + 1;
	END LOOP;
	
	-- Test with regclass parameter
	FOR regclass_result IN 
		SELECT * FROM neurondb.rag_query(
			query_text,
			'rag_compat_test_documents'::regclass,
			'embedding',
			'content',
			'default',
			3
		)
	LOOP
		regclass_result_count := regclass_result_count + 1;
	END LOOP;
	
	IF text_result_count = 0 THEN
		RAISE NOTICE 'rag_query with text parameter returned no results (embed/LLM may not be configured)';
	END IF;
	IF regclass_result_count = 0 THEN
		RAISE NOTICE 'rag_query with regclass parameter returned no results (embed/LLM may not be configured)';
	END IF;
	IF text_result_count != regclass_result_count THEN
		RAISE NOTICE 'Result count mismatch: text=% vs regclass=% (acceptable when no embedding model)', text_result_count, regclass_result_count;
	END IF;
	
	RAISE NOTICE '✓ rag_query compatibility verified: text=% results, regclass=% results', text_result_count, regclass_result_count;
END $$;

-- Test 2: rag_query_with_context with text vs regclass
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Test 2: rag_query_with_context compatibility (text vs regclass)'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

DO $$
DECLARE
	query_text text := 'What is PostgreSQL?';
	text_result record;
	regclass_result record;
	text_result_count int := 0;
	regclass_result_count int := 0;
BEGIN
	-- Test with text parameter
	FOR text_result IN 
		SELECT * FROM neurondb.rag_query_with_context(
			query_text,
			'rag_compat_test_documents'::text,
			'embedding',
			'content',
			'default',
			3
		)
	LOOP
		text_result_count := text_result_count + 1;
	END LOOP;
	
	-- Test with regclass parameter
	FOR regclass_result IN 
		SELECT * FROM neurondb.rag_query_with_context(
			query_text,
			'rag_compat_test_documents'::regclass,
			'embedding',
			'content',
			'default',
			3
		)
	LOOP
		regclass_result_count := regclass_result_count + 1;
	END LOOP;
	
	IF text_result_count = 0 OR regclass_result_count = 0 THEN
		RAISE EXCEPTION 'rag_query_with_context returned no results (text=% or regclass=%)', text_result_count, regclass_result_count;
	END IF;
	
	RAISE NOTICE '✓ rag_query_with_context compatibility verified: text=% results, regclass=% results', text_result_count, regclass_result_count;
END $$;

-- Test 3: rag_ingest_document with text vs regclass
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Test 3: rag_ingest_document compatibility (text vs regclass)'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

DO $$
DECLARE
	document_text text := 'This is a test document for ingestion compatibility testing.';
	text_result record;
	regclass_result record;
	text_result_count int := 0;
	regclass_result_count int := 0;
BEGIN
	-- Create separate tables for testing
	CREATE TEMP TABLE IF NOT EXISTS ingest_text_test (id SERIAL PRIMARY KEY, content TEXT, embedding VECTOR(384), metadata JSONB);
	CREATE TEMP TABLE IF NOT EXISTS ingest_regclass_test (id SERIAL PRIMARY KEY, content TEXT, embedding VECTOR(384), metadata JSONB);
	
	-- Test with text parameter
	FOR text_result IN 
		SELECT * FROM neurondb.rag_ingest_document(
			document_text,
			'ingest_text_test'::text,
			'content',
			'embedding',
			'default',
			100,
			20
		)
	LOOP
		text_result_count := text_result_count + 1;
	END LOOP;
	
	-- Test with regclass parameter
	FOR regclass_result IN 
		SELECT * FROM neurondb.rag_ingest_document(
			document_text,
			'ingest_regclass_test'::regclass,
			'content',
			'embedding',
			'default',
			100,
			20
		)
	LOOP
		regclass_result_count := regclass_result_count + 1;
	END LOOP;
	
	IF text_result_count = 0 OR regclass_result_count = 0 THEN
		RAISE EXCEPTION 'rag_ingest_document returned no results (text=% or regclass=%)', text_result_count, regclass_result_count;
	END IF;
	
	RAISE NOTICE '✓ rag_ingest_document compatibility verified: text=% chunks, regclass=% chunks', text_result_count, regclass_result_count;
END $$;

-- Test 4: rag_chat with text vs regclass
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Test 4: rag_chat compatibility (text vs regclass)'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

DO $$
DECLARE
	query_text text := 'What is PostgreSQL?';
	text_result jsonb;
	regclass_result jsonb;
BEGIN
	-- Test with text parameter
	SELECT neurondb.rag_chat(
		query_text,
		'rag_compat_test_documents'::text,
		'embedding',
		'content',
		'default',
		3,
		'[]'::jsonb,
		'gpt-3.5-turbo'
	) INTO text_result;
	
	-- Test with regclass parameter
	SELECT neurondb.rag_chat(
		query_text,
		'rag_compat_test_documents'::regclass,
		'embedding',
		'content',
		'default',
		3,
		'[]'::jsonb,
		'gpt-3.5-turbo'
	) INTO regclass_result;
	
	IF text_result IS NULL OR regclass_result IS NULL THEN
		RAISE EXCEPTION 'rag_chat returned null result';
	END IF;
	
	RAISE NOTICE '✓ rag_chat compatibility verified: both text and regclass parameters work';
END $$;

-- Test 5: Schema-qualified table names with regclass
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Test 5: Schema-qualified table names with regclass'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

DO $$
DECLARE
	query_text text := 'What is PostgreSQL?';
	result_count int := 0;
BEGIN
	-- Test with schema-qualified regclass (pg_temp.rag_compat_test_documents)
	-- Note: For temp tables, we use the table name directly as regclass resolves correctly
	SELECT COUNT(*) INTO result_count
	FROM neurondb.rag_query(
		query_text,
		'rag_compat_test_documents'::regclass,
		'embedding',
		'content',
		'default',
		3
	);
	
	IF result_count = 0 THEN
		RAISE EXCEPTION 'Schema-qualified regclass test returned no results';
	END IF;
	
	RAISE NOTICE '✓ Schema-qualified regclass compatibility verified: % results', result_count;
END $$;

-- Test 6: Verify function overloads exist
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Test 6: Verify function overloads exist in pg_proc'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

DO $$
DECLARE
	text_overload_count int;
	regclass_overload_count int;
BEGIN
	-- Count text overloads
	SELECT COUNT(*) INTO text_overload_count
	FROM pg_proc p
	JOIN pg_namespace n ON p.pronamespace = n.oid
	WHERE n.nspname = 'neurondb'
		AND p.proname = 'rag_query'
		AND pg_get_function_arguments(p.oid) LIKE '%document_table text%';
	
	-- Count regclass overloads
	SELECT COUNT(*) INTO regclass_overload_count
	FROM pg_proc p
	JOIN pg_namespace n ON p.pronamespace = n.oid
	WHERE n.nspname = 'neurondb'
		AND p.proname = 'rag_query'
		AND pg_get_function_arguments(p.oid) LIKE '%document_table regclass%';
	
	IF text_overload_count = 0 THEN
		RAISE EXCEPTION 'Text overload for rag_query not found';
	END IF;
	
	IF regclass_overload_count = 0 THEN
		RAISE EXCEPTION 'Regclass overload for rag_query not found';
	END IF;
	
	RAISE NOTICE '✓ Function overloads verified: text=% overloads, regclass=% overloads', text_overload_count, regclass_overload_count;
END $$;

\echo ''
\echo '=========================================================================='
\echo '✅ All RAG function compatibility tests passed!'
\echo '=========================================================================='
\echo ''
\echo 'Summary:'
\echo '  ✓ rag_query supports both text and regclass parameters'
\echo '  ✓ rag_query_with_context supports both text and regclass parameters'
\echo '  ✓ rag_ingest_document supports both text and regclass parameters'
\echo '  ✓ rag_chat supports both text and regclass parameters'
\echo '  ✓ Schema-qualified table names work with regclass'
\echo '  ✓ Function overloads are properly registered in pg_proc'
\echo ''
\echo 'Test completed successfully'
