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
\echo 'Test Suite: Graph RAG (Knowledge Graph-based Retrieval)'
\echo ''
\echo 'NOTE: embed() warnings are expected if LLM is not configured.'
\echo '      To generate real embeddings, configure:'
\echo '      - neurondb.llm_api_key (Hugging Face API key)'
\echo '      - Or enable GPU embedding via GUC (ALTER SYSTEM SET neurondb.compute_mode = on)'
\echo '      Without configuration, embed() returns zero vectors (graceful fallback).'
\echo ''

-- Create test table for Graph RAG documents with entities and relations
DROP TABLE IF EXISTS graph_rag_test_documents;
CREATE TEMP TABLE graph_rag_test_documents (
	id SERIAL PRIMARY KEY,
	content TEXT NOT NULL,
	embedding VECTOR(384),
	entities JSONB DEFAULT '[]'::jsonb,
	relations JSONB DEFAULT '[]'::jsonb,
	metadata JSONB DEFAULT '{}'::jsonb
);

-- Insert sample documents with entity and relation information
INSERT INTO graph_rag_test_documents (content, embedding, entities, relations) VALUES
	('PostgreSQL is a powerful open-source relational database management system. It provides ACID compliance and full-text search.', 
	 embed_text('PostgreSQL is a powerful open-source relational database management system. It provides ACID compliance and full-text search.', 'all-MiniLM-L6-v2'),
	 '[{"name": "PostgreSQL", "type": "database"}, {"name": "ACID", "type": "concept"}]'::jsonb,
	 '[{"source": "PostgreSQL", "target": "ACID", "relation": "supports"}]'::jsonb),
	('NeuronDB extends PostgreSQL with vector search and machine learning. It enables RAG pipelines and semantic search.', 
	 embed_text('NeuronDB extends PostgreSQL with vector search and machine learning. It enables RAG pipelines and semantic search.', 'all-MiniLM-L6-v2'),
	 '[{"name": "NeuronDB", "type": "extension"}, {"name": "PostgreSQL", "type": "database"}, {"name": "RAG", "type": "concept"}]'::jsonb,
	 '[{"source": "NeuronDB", "target": "PostgreSQL", "relation": "extends"}, {"source": "NeuronDB", "target": "RAG", "relation": "enables"}]'::jsonb),
	('Vector search uses HNSW indexes for fast similarity queries. It enables semantic search over embeddings.', 
	 embed_text('Vector search uses HNSW indexes for fast similarity queries. It enables semantic search over embeddings.', 'all-MiniLM-L6-v2'),
	 '[{"name": "Vector search", "type": "feature"}, {"name": "HNSW", "type": "algorithm"}, {"name": "semantic search", "type": "concept"}]'::jsonb,
	 '[{"source": "Vector search", "target": "HNSW", "relation": "uses"}, {"source": "Vector search", "target": "semantic search", "relation": "enables"}]'::jsonb),
	('Machine learning in databases includes classification, regression, and clustering algorithms. These can be trained directly in PostgreSQL.', 
	 embed_text('Machine learning in databases includes classification, regression, and clustering algorithms. These can be trained directly in PostgreSQL.', 'all-MiniLM-L6-v2'),
	 '[{"name": "Machine learning", "type": "concept"}, {"name": "classification", "type": "algorithm"}, {"name": "PostgreSQL", "type": "database"}]'::jsonb,
	 '[{"source": "Machine learning", "target": "classification", "relation": "includes"}, {"source": "Machine learning", "target": "PostgreSQL", "relation": "runs_in"}]'::jsonb),
	('RAG combines retrieval with generation. It uses vector search to find relevant documents and LLMs to generate answers.', 
	 embed_text('RAG combines retrieval with generation. It uses vector search to find relevant documents and LLMs to generate answers.', 'all-MiniLM-L6-v2'),
	 '[{"name": "RAG", "type": "concept"}, {"name": "vector search", "type": "feature"}, {"name": "LLM", "type": "model"}]'::jsonb,
	 '[{"source": "RAG", "target": "vector search", "relation": "uses"}, {"source": "RAG", "target": "LLM", "relation": "uses"}]'::jsonb);

\echo 'Sample documents with entities and relations inserted'
\echo ''

-- Test 1: Basic Graph RAG Query
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Test 1: Basic Graph RAG Query'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

DO $$
DECLARE
	query_text text := 'What is PostgreSQL?';
	graph_result record;
	result_count int := 0;
	has_answer boolean := false;
	has_graph_path boolean := false;
BEGIN
	-- Execute Graph RAG query
	FOR graph_result IN 
		SELECT * FROM neurondb.rag_graph(
			query_text,
			'graph_rag_test_documents',
			'embedding',
			'content',
			'entities',
			'relations',
			'default',
			3,  -- top_k
			2,  -- max_depth
			'bfs',  -- traversal_method
			'{}'::jsonb  -- custom_context
		)
	LOOP
		result_count := result_count + 1;
		
		IF graph_result.answer IS NOT NULL AND graph_result.answer != '' THEN
			has_answer := true;
			IF result_count = 1 THEN
				RAISE NOTICE 'Answer generated: %', substring(graph_result.answer, 1, 100) || '...';
			END IF;
		END IF;
		
		IF graph_result.graph_path IS NOT NULL AND array_length(graph_result.graph_path, 1) > 0 THEN
			has_graph_path := true;
			IF result_count = 1 THEN
				RAISE NOTICE 'Graph path found: %', array_to_string(graph_result.graph_path, ' -> ');
			END IF;
		END IF;
		
		IF result_count = 1 THEN
			RAISE NOTICE 'Top chunk: % (relevance: %)', 
				substring(graph_result.chunk_text, 1, 80) || '...',
				graph_result.relevance_score;
		END IF;
	END LOOP;
	
	IF result_count = 0 THEN
		RAISE EXCEPTION 'Graph RAG query returned no results';
	END IF;
	
	IF NOT has_answer THEN
		RAISE NOTICE '⚠ Answer generation may have failed (LLM not configured or API issue)';
		RAISE NOTICE 'Context retrieval (core functionality) succeeded with % results', result_count;
	ELSE
		RAISE NOTICE '✓ Graph RAG query successful: % results, answer generated', result_count;
	END IF;
	
	IF has_graph_path THEN
		RAISE NOTICE '✓ Graph traversal successful';
	ELSE
		RAISE NOTICE '⚠ Graph path may not have been generated (no relations found)';
	END IF;
END $$;

-- Test 2: Graph RAG with Entity Traversal
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Test 2: Graph RAG with Entity Traversal'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

DO $$
DECLARE
	query_text text := 'How does NeuronDB relate to RAG?';
	graph_result record;
	result_count int := 0;
BEGIN
	-- Execute Graph RAG query that should traverse entity relationships
	FOR graph_result IN 
		SELECT * FROM neurondb.rag_graph(
			query_text,
			'graph_rag_test_documents',
			'embedding',
			'content',
			'entities',
			'relations',
			'default',
			5,  -- top_k
			3,  -- max_depth (deeper traversal)
			'bfs',
			'{}'::jsonb
		)
	LOOP
		result_count := result_count + 1;
	END LOOP;
	
	IF result_count = 0 THEN
		RAISE EXCEPTION 'Graph RAG with entity traversal returned no results';
	END IF;
	
	RAISE NOTICE '✓ Graph RAG with entity traversal successful: % results', result_count;
END $$;

-- Test 3: Graph RAG with DFS Traversal
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Test 3: Graph RAG with DFS Traversal'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

DO $$
DECLARE
	query_text text := 'What algorithms are used in machine learning?';
	graph_result record;
	result_count int := 0;
BEGIN
	-- Execute Graph RAG with DFS traversal
	FOR graph_result IN 
		SELECT * FROM neurondb.rag_graph(
			query_text,
			'graph_rag_test_documents',
			'embedding',
			'content',
			'entities',
			'relations',
			'default',
			5,
			2,
			'dfs',  -- DFS traversal
			'{}'::jsonb
		)
	LOOP
		result_count := result_count + 1;
	END LOOP;
	
	IF result_count = 0 THEN
		RAISE EXCEPTION 'Graph RAG with DFS traversal returned no results';
	END IF;
	
	RAISE NOTICE '✓ Graph RAG with DFS traversal successful: % results', result_count;
END $$;

\echo ''
\echo '=========================================================================='
\echo '✅ All Graph RAG tests passed!'
\echo '=========================================================================='

\echo 'Test completed successfully'
