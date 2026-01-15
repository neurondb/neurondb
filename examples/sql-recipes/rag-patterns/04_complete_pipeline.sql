-- ============================================================================
-- Recipe: Complete RAG Pipeline
-- ============================================================================
-- Purpose: End-to-end RAG implementation from query to context
-- 
-- Prerequisites:
--   - quickstart_documents table
--   - document_chunks table with embeddings (from chunking recipe)
--   - HNSW index on chunk embeddings
--
-- RAG Pipeline Steps:
--   1. Query understanding (convert to embedding)
--   2. Context retrieval (vector search + filtering)
--   3. Reranking (optional, improves quality)
--   4. Context formatting (prepare for LLM)
--
-- Performance Notes:
--   - Pipeline latency: typically 50-200ms
--   - Consider caching for frequently asked questions
--   - Monitor retrieval quality (similarity scores)
-- ============================================================================

-- Example 1: Complete RAG pipeline function
-- Encapsulates the full RAG retrieval process
CREATE OR REPLACE FUNCTION rag_retrieve_context(
    query_text TEXT,
    max_chunks INTEGER DEFAULT 5,
    similarity_threshold FLOAT DEFAULT 0.7
)
RETURNS TABLE (
    chunk_id INTEGER,
    document_title TEXT,
    chunk_text TEXT,
    similarity_score FLOAT,
    context_order INTEGER
) AS $$
BEGIN
    RETURN QUERY
    WITH query_vector AS (
        SELECT embed_text(query_text) AS query_emb
    ),
    retrieved_chunks AS (
        SELECT 
            dc.chunk_id,
            d.title AS document_title,
            dc.chunk_text,
            1 - (dc.embedding <=> query_vector.query_emb) AS similarity_score,
            ROW_NUMBER() OVER (ORDER BY dc.embedding <=> query_vector.query_emb) AS rank
        FROM document_chunks dc
        JOIN quickstart_documents d ON dc.doc_id = d.id
        CROSS JOIN query_vector
        WHERE dc.embedding IS NOT NULL
          AND (1 - (dc.embedding <=> query_vector.query_emb)) >= similarity_threshold
        ORDER BY dc.embedding <=> query_vector.query_emb
        LIMIT max_chunks * 2  -- Get more for reranking
    ),
    reranked AS (
        SELECT 
            *,
            -- Simple reranking: boost by category match
            similarity_score * 
            CASE 
                WHEN document_title ILIKE '%' || query_text || '%' THEN 1.2
                ELSE 1.0
            END AS reranked_score
        FROM retrieved_chunks
    )
    SELECT 
        chunk_id,
        document_title,
        chunk_text,
        similarity_score::FLOAT,
        ROW_NUMBER() OVER (ORDER BY reranked_score DESC)::INTEGER AS context_order
    FROM reranked
    ORDER BY reranked_score DESC
    LIMIT max_chunks;
END;
$$ LANGUAGE plpgsql;

-- Use the function
SELECT * FROM rag_retrieve_context('How do vector databases work?', max_chunks := 5);

-- Example 2: RAG pipeline with formatted context
-- Returns context ready for LLM input
WITH query_text AS (
    SELECT 'What are the best practices for database indexing?' AS query
),
context_chunks AS (
    SELECT * FROM rag_retrieve_context((SELECT query FROM query_text), max_chunks := 5)
)
SELECT 
    format(
        'Context for query: %s\n\n%s',
        (SELECT query FROM query_text),
        string_agg(
            format(
                '[%s] %s\n%s\n---',
                context_order,
                document_title,
                chunk_text
            ),
            E'\n\n'
            ORDER BY context_order
        )
    ) AS formatted_context
FROM context_chunks;

-- Example 3: RAG pipeline with metadata
-- Include metadata for better context understanding
CREATE OR REPLACE FUNCTION rag_retrieve_with_metadata(
    query_text TEXT,
    max_chunks INTEGER DEFAULT 5
)
RETURNS TABLE (
    chunk_id INTEGER,
    document_title TEXT,
    document_category TEXT,
    document_tags TEXT[],
    chunk_text TEXT,
    similarity_score FLOAT,
    chunk_position INTEGER
) AS $$
BEGIN
    RETURN QUERY
    WITH query_vector AS (
        SELECT embed_text(query_text) AS query_emb
    )
    SELECT 
        dc.chunk_id,
        d.title AS document_title,
        d.category AS document_category,
        d.tags AS document_tags,
        dc.chunk_text,
        (1 - (dc.embedding <=> query_vector.query_emb))::FLOAT AS similarity_score,
        dc.chunk_index AS chunk_position
    FROM document_chunks dc
    JOIN quickstart_documents d ON dc.doc_id = d.id
    CROSS JOIN query_vector
    WHERE dc.embedding IS NOT NULL
    ORDER BY dc.embedding <=> query_vector.query_emb
    LIMIT max_chunks;
END;
$$ LANGUAGE plpgsql;

-- Use the enhanced function
SELECT * FROM rag_retrieve_with_metadata('machine learning embeddings', max_chunks := 5);

-- Example 4: RAG pipeline with filtering
-- Add metadata filters to retrieval
CREATE OR REPLACE FUNCTION rag_retrieve_filtered(
    query_text TEXT,
    category_filter TEXT DEFAULT NULL,
    max_chunks INTEGER DEFAULT 5
)
RETURNS TABLE (
    chunk_id INTEGER,
    document_title TEXT,
    chunk_text TEXT,
    similarity_score FLOAT
) AS $$
BEGIN
    RETURN QUERY
    WITH query_vector AS (
        SELECT embed_text(query_text) AS query_emb
    )
    SELECT 
        dc.chunk_id,
        d.title AS document_title,
        dc.chunk_text,
        (1 - (dc.embedding <=> query_vector.query_emb))::FLOAT AS similarity_score
    FROM document_chunks dc
    JOIN quickstart_documents d ON dc.doc_id = d.id
    CROSS JOIN query_vector
    WHERE dc.embedding IS NOT NULL
      AND (category_filter IS NULL OR d.category = category_filter)
    ORDER BY dc.embedding <=> query_vector.query_emb
    LIMIT max_chunks;
END;
$$ LANGUAGE plpgsql;

-- Use with category filter
SELECT * FROM rag_retrieve_filtered(
    'database performance', 
    category_filter := 'database',
    max_chunks := 5
);

-- Example 5: RAG pipeline performance monitoring
-- Track retrieval performance and quality
WITH query_vector AS (
    SELECT embed_text('vector similarity search') AS query_emb
),
performance_test AS (
    SELECT 
        COUNT(*) AS total_chunks_searched,
        COUNT(*) FILTER (
            WHERE (1 - (embedding <=> query_vector.query_emb)) > 0.7
        ) AS high_similarity_chunks,
        MIN(1 - (embedding <=> query_vector.query_emb)) AS min_similarity,
        MAX(1 - (embedding <=> query_vector.query_emb)) AS max_similarity,
        AVG(1 - (embedding <=> query_vector.query_emb)) AS avg_similarity
    FROM document_chunks
    CROSS JOIN query_vector
    WHERE embedding IS NOT NULL
)
SELECT 
    total_chunks_searched,
    high_similarity_chunks,
    ROUND(min_similarity::numeric, 4) AS min_similarity,
    ROUND(max_similarity::numeric, 4) AS max_similarity,
    ROUND(avg_similarity::numeric, 4) AS avg_similarity,
    ROUND((high_similarity_chunks::FLOAT / NULLIF(total_chunks_searched, 0) * 100)::numeric, 2) AS high_similarity_percentage
FROM performance_test;




