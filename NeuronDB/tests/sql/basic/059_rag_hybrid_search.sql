\timing on
\pset footer off
\pset pager off

\set ON_ERROR_STOP on

-- Enable fail_open mode to allow tests to run without API keys (uses fallback embeddings)
SET neurondb.llm_fail_open = on;

\echo '=========================================================================='
\echo '=========================================================================='

-- Create test table with text and vector columns
DROP TABLE IF EXISTS hybrid_search_test;
CREATE TEMP TABLE hybrid_search_test (
	id SERIAL PRIMARY KEY,
	title TEXT NOT NULL,
	content TEXT NOT NULL,
	embedding VECTOR(384),
	fts_vector tsvector,
	metadata jsonb DEFAULT '{}'::jsonb
);

-- Insert sample documents
INSERT INTO hybrid_search_test (title, content, embedding, fts_vector) VALUES
	('PostgreSQL Database', 'PostgreSQL is a powerful open-source relational database management system', embed_text('PostgreSQL is a powerful open-source relational database management system', 'all-MiniLM-L6-v2'), to_tsvector('PostgreSQL is a powerful open-source relational database management system')),
	('Machine Learning', 'Machine learning algorithms learn patterns from data to make predictions', embed_text('Machine learning algorithms learn patterns from data to make predictions', 'all-MiniLM-L6-v2'), to_tsvector('Machine learning algorithms learn patterns from data to make predictions')),
	('Vector Search', 'Vector databases enable semantic similarity search using embeddings', embed_text('Vector databases enable semantic similarity search using embeddings', 'all-MiniLM-L6-v2'), to_tsvector('Vector databases enable semantic similarity search using embeddings')),
	('GPU Computing', 'GPU acceleration speeds up parallel processing tasks significantly', embed_text('GPU acceleration speeds up parallel processing tasks significantly', 'all-MiniLM-L6-v2'), to_tsvector('GPU acceleration speeds up parallel processing tasks significantly')),
	('Database Systems', 'Database systems store and manage structured data efficiently', embed_text('Database systems store and manage structured data efficiently', 'all-MiniLM-L6-v2'), to_tsvector('Database systems store and manage structured data efficiently'));

\echo 'Sample documents inserted'

-- Test 1: Hybrid Search with Text and Vector
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT 
	search_result.id,
	hybrid_search_test.title,
	hybrid_search_test.content,
	search_result.score
FROM hybrid_search_test,
	LATERAL hybrid_search(
		'hybrid_search_test',
		embed_text('database systems', 'all-MiniLM-L6-v2'),
		'database systems',
		'{}'::text,
		0.7,
		5
	) AS search_result(id, score)
WHERE hybrid_search_test.id = search_result.id
ORDER BY search_result.score DESC
LIMIT 5;

-- Test 2: Hybrid Search Fusion
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

WITH vector_results AS (
	SELECT 
		id,
		embedding <-> embed_text('database systems', 'all-MiniLM-L6-v2') AS distance
	FROM hybrid_search_test
	ORDER BY distance
	LIMIT 5
),
text_results AS (
	SELECT 
		id,
		ts_rank(to_tsvector('english', title || ' ' || content), to_tsquery('english', 'database & systems')) AS rank
	FROM hybrid_search_test
	WHERE to_tsvector('english', title || ' ' || content) @@ to_tsquery('english', 'database & systems')
	ORDER BY rank DESC
	LIMIT 5
)
SELECT 
	COALESCE(v.id, t.id) AS id,
	hybrid_search_test.title,
	hybrid_search_test.content
FROM vector_results v
FULL OUTER JOIN text_results t ON v.id = t.id
JOIN hybrid_search_test ON hybrid_search_test.id = COALESCE(v.id, t.id)
ORDER BY COALESCE(v.distance, 0.0) + COALESCE(t.rank, 0.0) DESC
LIMIT 5;

\echo ''
\echo '=========================================================================='

\echo 'Test completed successfully'
