-- ============================================================================
-- Test 008: Contextual Retrieval with Metadata Filtering
-- ============================================================================
-- Demonstrates: Metadata filtering, category-based retrieval
-- ============================================================================

\echo 'Testing contextual retrieval with metadata filtering...'

\echo ''
\echo 'Query 1: Search within "machine_learning" category'

WITH query_embedding AS (
    SELECT embed_text('training models with data',
        'sentence-transformers/all-MiniLM-L6-v2'::text
    ) AS embedding
)
SELECT 
    d.title,
    d.metadata->>'category' AS category,
    dc.chunk_text,
    ROUND((1 - (dc.embedding <=> qe.embedding))::numeric, 4) AS similarity
FROM document_chunks dc
JOIN documents d ON dc.doc_id = d.doc_id
CROSS JOIN query_embedding qe
WHERE d.metadata->>'category' = 'machine_learning'
ORDER BY dc.embedding <=> qe.embedding
LIMIT 5;

\echo ''
\echo 'Query 2: Search within "database" category'

WITH query_embedding AS (
    SELECT embed_text('performance tuning',
        'sentence-transformers/all-MiniLM-L6-v2'::text
    ) AS embedding
)
SELECT 
    d.title,
    d.metadata->>'category' AS category,
    dc.chunk_text,
    ROUND((1 - (dc.embedding <=> qe.embedding))::numeric, 4) AS similarity
FROM document_chunks dc
JOIN documents d ON dc.doc_id = d.doc_id
CROSS JOIN query_embedding qe
WHERE d.metadata->>'category' = 'database'
ORDER BY dc.embedding <=> qe.embedding
LIMIT 5;

\echo ''
\echo 'Query 3: Time-based filtering (recent documents)'

SELECT 
    d.title,
    d.created_at,
    d.metadata->>'category' AS category,
    left(d.content, 100) || '...' AS preview
FROM documents d
WHERE d.created_at >= CURRENT_DATE - INTERVAL '7 days'
ORDER BY d.created_at DESC;

\echo ''
\echo 'Contextual retrieval complete!'


