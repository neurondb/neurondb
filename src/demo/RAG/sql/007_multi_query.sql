-- ============================================================================
-- Test 007: Multi-Query Retrieval
-- ============================================================================
-- Demonstrates: Query expansion, multiple perspective retrieval
-- ============================================================================

\echo 'Testing multi-query retrieval...'

\echo ''
\echo 'Original Query: "database performance optimization"'
\echo ''
\echo 'Expanding to multiple query perspectives...'

-- Simulate query expansion (in production, use LLM to generate variations)
CREATE TEMP TABLE IF NOT EXISTS query_variations (
    variation_id SERIAL,
    original_query TEXT,
    query_variation TEXT
);

INSERT INTO query_variations (original_query, query_variation) VALUES
    ('database performance optimization', 'database performance optimization'),
    ('database performance optimization', 'how to make databases faster'),
    ('database performance optimization', 'SQL query optimization techniques'),
    ('database performance optimization', 'database indexing strategies');

\echo 'Query variations:'
SELECT * FROM query_variations;

\echo ''
\echo 'Retrieving documents for each query variation...'

WITH multi_query_results AS (
    SELECT 
        qv.variation_id,
        qv.query_variation,
        dc.chunk_id,
        d.title,
        dc.chunk_text,
        1 - (dc.embedding <=> embed_text(qv.query_variation,
            'sentence-transformers/all-MiniLM-L6-v2'::text
        )) AS similarity,
        ROW_NUMBER() OVER (PARTITION BY qv.variation_id 
                          ORDER BY dc.embedding <=> embed_text(qv.query_variation,
                              'sentence-transformers/all-MiniLM-L6-v2'::text
                          )) AS rank_in_query
    FROM query_variations qv
    CROSS JOIN document_chunks dc
    JOIN documents d ON dc.doc_id = d.doc_id
),
ranked_results AS (
    SELECT 
        chunk_id,
        title,
        chunk_text,
        AVG(similarity) AS avg_similarity,
        COUNT(DISTINCT variation_id) AS found_in_queries,
        MIN(rank_in_query) AS best_rank
    FROM multi_query_results
    WHERE rank_in_query <= 5
    GROUP BY chunk_id, title, chunk_text
)
SELECT 
    chunk_id,
    title,
    left(chunk_text, 100) || '...' AS preview,
    ROUND(avg_similarity::numeric, 4) AS avg_score,
    found_in_queries,
    best_rank
FROM ranked_results
ORDER BY found_in_queries DESC, avg_similarity DESC
LIMIT 5;

\echo ''
\echo 'Multi-query retrieval complete!'

DROP TABLE query_variations;


