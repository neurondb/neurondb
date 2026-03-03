-- ============================================================================
-- Test 009: Document Summarization
-- ============================================================================
-- Demonstrates: Retrieving and aggregating content for summarization
-- ============================================================================

\echo 'Testing document summarization capabilities...'

\echo ''
\echo 'Summarization Task: "Summarize all information about PostgreSQL"'

-- Retrieve all relevant chunks
WITH relevant_chunks AS (
    SELECT 
        d.title,
        dc.chunk_text,
        1 - (dc.embedding <=> embed_text('PostgreSQL database',
            'sentence-transformers/all-MiniLM-L6-v2'::text
        )) AS relevance
    FROM document_chunks dc
    JOIN documents d ON dc.doc_id = d.doc_id
    WHERE 1 - (dc.embedding <=> embed_text('PostgreSQL database',
        'sentence-transformers/all-MiniLM-L6-v2'::text
    )) > 0.3
    ORDER BY dc.embedding <=> embed_text('PostgreSQL database',
        'sentence-transformers/all-MiniLM-L6-v2'::text
    )
    LIMIT 10
)
SELECT 
    title,
    COUNT(*) AS chunk_count,
    string_agg(left(chunk_text, 80), ' | ') AS content_preview,
    ROUND(AVG(relevance)::numeric, 4) AS avg_relevance
FROM relevant_chunks
GROUP BY title
ORDER BY avg_relevance DESC;

\echo ''
\echo 'Full content for summarization:'

WITH relevant_chunks AS (
    SELECT 
        d.doc_id,
        d.title,
        dc.chunk_text,
        dc.chunk_index
    FROM document_chunks dc
    JOIN documents d ON dc.doc_id = d.doc_id
    WHERE d.title LIKE '%PostgreSQL%'
    ORDER BY d.doc_id, dc.chunk_index
)
SELECT 
    title,
    string_agg(chunk_text, ' ' ORDER BY chunk_index) AS full_content_for_summary
FROM relevant_chunks
GROUP BY doc_id, title;

\echo ''
\echo 'Document summarization complete!'


