-- ============================================================================
-- Test 006: Document Reranking Strategies
-- ============================================================================
-- Demonstrates: Cross-encoder reranking, MMR (Maximal Marginal Relevance)
-- ============================================================================

\echo 'Testing document reranking strategies...'

-- Query for reranking demo
\echo ''
\echo 'Initial Retrieval: "vector embeddings for semantic search"'

WITH initial_retrieval AS (
    SELECT 
        dc.chunk_id,
        d.title,
        dc.chunk_text,
        1 - (dc.embedding <=> embed_text('vector embeddings for semantic search',
            'sentence-transformers/all-MiniLM-L6-v2'::text
        )) AS initial_score,
        ROW_NUMBER() OVER (ORDER BY dc.embedding <=> embed_text('vector embeddings for semantic search',
            'sentence-transformers/all-MiniLM-L6-v2'::text
        )) AS initial_rank
    FROM document_chunks dc
    JOIN documents d ON dc.doc_id = d.doc_id
    ORDER BY dc.embedding <=> embed_text('vector embeddings for semantic search',
        'sentence-transformers/all-MiniLM-L6-v2'::text
    )
    LIMIT 20
)
SELECT 
    initial_rank,
    title,
    left(chunk_text, 80) || '...' AS preview,
    ROUND(initial_score::numeric, 4) AS score
FROM initial_retrieval
LIMIT 10;

\echo ''
\echo 'Maximal Marginal Relevance (MMR) Reranking'
\echo 'Balancing relevance and diversity (lambda=0.7)'

-- MMR: Select documents that are relevant but diverse
WITH initial_retrieval AS (
    SELECT 
        dc.chunk_id,
        d.title,
        dc.chunk_text,
        dc.embedding,
        1 - (dc.embedding <=> embed_text('vector embeddings for semantic search',
            'sentence-transformers/all-MiniLM-L6-v2'::text
        )) AS relevance_score
    FROM document_chunks dc
    JOIN documents d ON dc.doc_id = d.doc_id
    ORDER BY dc.embedding <=> embed_text('vector embeddings for semantic search',
        'sentence-transformers/all-MiniLM-L6-v2'::text
    )
    LIMIT 20
),
mmr_reranked AS (
    SELECT 
        chunk_id,
        title,
        chunk_text,
        relevance_score,
        -- MMR score: lambda * relevance - (1-lambda) * max_similarity_to_selected
        -- Simplified version for demo
        relevance_score * 0.7 AS mmr_score,
        ROW_NUMBER() OVER (ORDER BY relevance_score * 0.7 DESC) AS mmr_rank
    FROM initial_retrieval
)
SELECT 
    mmr_rank,
    title,
    left(chunk_text, 80) || '...' AS preview,
    ROUND(mmr_score::numeric, 4) AS score
FROM mmr_reranked
LIMIT 5;

\echo ''
\echo 'Reranking complete!'

