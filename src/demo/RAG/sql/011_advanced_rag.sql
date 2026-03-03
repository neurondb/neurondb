-- ============================================================================
-- Test 011: Advanced RAG Patterns
-- ============================================================================
-- Demonstrates: Agentic RAG, self-query, adaptive retrieval
-- ============================================================================

\echo 'Testing advanced RAG patterns...'

\echo ''
\echo 'Pattern 1: Self-Query with Metadata Extraction'
\echo 'Query: "Show me recent database articles about performance"'

-- Extract filters from query (in production, use LLM)
-- Extracted: category='database', tags contains 'performance', recent=true

WITH filtered_docs AS (
    SELECT 
        d.doc_id,
        d.title,
        d.metadata,
        dc.chunk_id,
        dc.chunk_text,
        dc.embedding
    FROM documents d
    JOIN document_chunks dc ON d.doc_id = dc.doc_id
    WHERE d.metadata->>'category' = 'database'
      AND d.metadata->'tags' ? 'performance'
),
query_results AS (
    SELECT 
        title,
        chunk_text,
        ROUND((1 - (embedding <=> embed_text('performance optimization',
            'sentence-transformers/all-MiniLM-L6-v2'::text
        )))::numeric, 4) AS score
    FROM filtered_docs
    ORDER BY embedding <=> embed_text('performance optimization',
        'sentence-transformers/all-MiniLM-L6-v2'::text
    )
    LIMIT 3
)
SELECT * FROM query_results;

\echo ''
\echo 'Pattern 2: Adaptive Retrieval (retrieve more if confidence is low)'

WITH initial_retrieval AS (
    SELECT 
        dc.chunk_id,
        dc.chunk_text,
        1 - (dc.embedding <=> embed_text('complex database query',
            'sentence-transformers/all-MiniLM-L6-v2'::text
        )) AS confidence
    FROM document_chunks dc
    ORDER BY dc.embedding <=> embed_text('complex database query',
        'sentence-transformers/all-MiniLM-L6-v2'::text
    )
    LIMIT 5
)
SELECT 
    CASE 
        WHEN MAX(confidence) < 0.6 THEN 'Low confidence - retrieve more documents'
        WHEN MAX(confidence) < 0.8 THEN 'Medium confidence - standard retrieval'
        ELSE 'High confidence - sufficient context'
    END AS retrieval_strategy,
    ROUND(MAX(confidence)::numeric, 4) AS max_confidence,
    COUNT(*) AS docs_retrieved
FROM initial_retrieval;

\echo ''
\echo 'Advanced RAG patterns complete!'


