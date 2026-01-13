-- ============================================================================
-- Recipe: Reranking for RAG
-- ============================================================================
-- Purpose: Improve context quality by reranking retrieved chunks
-- 
-- Prerequisites:
--   - document_chunks table with embeddings
--   - Initial retrieval results
--
-- Reranking Strategies:
--   1. Cross-encoder reranking (more accurate, slower)
--   2. Metadata-based reranking (fast, uses document attributes)
--   3. Multi-factor reranking (combines multiple signals)
--
-- Performance Notes:
--   - Reranking improves quality but adds latency
--   - Typically rerank top 20-50, return top 5-10
--   - Consider caching reranked results for common queries
-- ============================================================================

-- Example 1: Metadata-based reranking
-- Boost chunks based on document metadata (recency, category, etc.)
WITH query_vector AS (
    SELECT embed_text('vector search optimization') AS query_emb
),
initial_results AS (
    SELECT 
        dc.chunk_id,
        dc.doc_id,
        d.title,
        d.category,
        dc.chunk_text,
        1 - (dc.embedding <=> query_vector.query_emb) AS base_similarity,
        d.created_at
    FROM document_chunks dc
    JOIN quickstart_documents d ON dc.doc_id = d.id
    CROSS JOIN query_vector
    WHERE dc.embedding IS NOT NULL
    ORDER BY dc.embedding <=> query_vector.query_emb
    LIMIT 20  -- Get more candidates for reranking
),
reranked AS (
    SELECT 
        *,
        -- Boost by category match
        CASE 
            WHEN category = 'search' THEN base_similarity * 1.2
            WHEN category = 'algorithms' THEN base_similarity * 1.1
            ELSE base_similarity
        END AS reranked_score
    FROM initial_results
)
SELECT 
    chunk_id,
    title,
    category,
    ROUND(base_similarity::numeric, 4) AS base_similarity,
    ROUND(reranked_score::numeric, 4) AS reranked_score
FROM reranked
ORDER BY reranked_score DESC
LIMIT 5;

-- Example 2: Recency-based reranking
-- Boost newer documents
WITH query_vector AS (
    SELECT embed_text('database performance') AS query_emb
),
initial_results AS (
    SELECT 
        dc.chunk_id,
        d.title,
        dc.chunk_text,
        1 - (dc.embedding <=> query_vector.query_emb) AS base_similarity,
        d.created_at,
        EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP - d.created_at)) / 86400.0 AS days_old
    FROM document_chunks dc
    JOIN quickstart_documents d ON dc.doc_id = d.id
    CROSS JOIN query_vector
    WHERE dc.embedding IS NOT NULL
    ORDER BY dc.embedding <=> query_vector.query_emb
    LIMIT 20
),
reranked AS (
    SELECT 
        *,
        -- Boost newer documents (decay factor)
        base_similarity * (1.0 / (1.0 + days_old / 365.0)) AS reranked_score
    FROM initial_results
)
SELECT 
    chunk_id,
    title,
    ROUND(days_old::numeric, 1) AS days_old,
    ROUND(base_similarity::numeric, 4) AS base_similarity,
    ROUND(reranked_score::numeric, 4) AS reranked_score
FROM reranked
ORDER BY reranked_score DESC
LIMIT 5;

-- Example 3: Multi-factor reranking
-- Combine similarity, metadata, and other signals
WITH query_vector AS (
    SELECT embed_text('neural networks') AS query_emb
),
initial_results AS (
    SELECT 
        dc.chunk_id,
        d.title,
        d.category,
        dc.chunk_text,
        d.tags,
        1 - (dc.embedding <=> query_vector.query_emb) AS similarity,
        CASE WHEN d.category = 'machine_learning' THEN 1.0 ELSE 0.5 END AS category_match,
        CASE WHEN 'neural_networks' = ANY(d.tags) THEN 1.0 ELSE 0.0 END AS tag_match,
        length(dc.chunk_text) AS chunk_length
    FROM document_chunks dc
    JOIN quickstart_documents d ON dc.doc_id = d.id
    CROSS JOIN query_vector
    WHERE dc.embedding IS NOT NULL
    ORDER BY dc.embedding <=> query_vector.query_emb
    LIMIT 20
),
reranked AS (
    SELECT 
        *,
        -- Combined score: similarity + category + tags - length penalty
        (similarity * 0.7) + 
        (category_match * 0.2) + 
        (tag_match * 0.1) - 
        (CASE WHEN chunk_length < 100 THEN 0.1 ELSE 0.0 END) AS combined_score
    FROM initial_results
)
SELECT 
    chunk_id,
    title,
    category,
    ROUND(similarity::numeric, 4) AS similarity,
    category_match,
    tag_match,
    ROUND(combined_score::numeric, 4) AS combined_score
FROM reranked
ORDER BY combined_score DESC
LIMIT 5;

-- Example 4: Reciprocal Rank Fusion for reranking
-- Combine multiple ranking signals using RRF
WITH query_vector AS (
    SELECT embed_text('vector databases') AS query_emb
),
similarity_ranked AS (
    SELECT 
        dc.chunk_id,
        ROW_NUMBER() OVER (ORDER BY dc.embedding <=> query_vector.query_emb) AS similarity_rank
    FROM document_chunks dc
    CROSS JOIN query_vector
    WHERE dc.embedding IS NOT NULL
    LIMIT 20
),
category_ranked AS (
    SELECT 
        dc.chunk_id,
        ROW_NUMBER() OVER (
            ORDER BY 
                CASE WHEN d.category = 'database' THEN 1 ELSE 2 END,
                dc.embedding <=> query_vector.query_emb
        ) AS category_rank
    FROM document_chunks dc
    JOIN quickstart_documents d ON dc.doc_id = d.id
    CROSS JOIN query_vector
    WHERE dc.embedding IS NOT NULL
    LIMIT 20
),
rrf_scores AS (
    SELECT 
        COALESCE(sr.chunk_id, cr.chunk_id) AS chunk_id,
        (1.0 / (60.0 + COALESCE(sr.similarity_rank, 1000))) + 
        (1.0 / (60.0 + COALESCE(cr.category_rank, 1000))) AS rrf_score
    FROM similarity_ranked sr
    FULL OUTER JOIN category_ranked cr ON sr.chunk_id = cr.chunk_id
)
SELECT 
    dc.chunk_id,
    d.title,
    dc.chunk_text,
    ROUND(rrf.rrf_score::numeric, 6) AS rrf_score
FROM document_chunks dc
JOIN quickstart_documents d ON dc.doc_id = d.id
JOIN rrf_scores rrf ON dc.chunk_id = rrf.chunk_id
ORDER BY rrf.rrf_score DESC
LIMIT 5;

-- Example 5: Compare reranked vs original results
-- See how reranking changes the order
WITH query_vector AS (
    SELECT embed_text('database optimization') AS query_emb
),
original_ranking AS (
    SELECT 
        dc.chunk_id,
        d.title,
        1 - (dc.embedding <=> query_vector.query_emb) AS similarity,
        ROW_NUMBER() OVER (ORDER BY dc.embedding <=> query_vector.query_emb) AS original_rank
    FROM document_chunks dc
    JOIN quickstart_documents d ON dc.doc_id = d.id
    CROSS JOIN query_vector
    WHERE dc.embedding IS NOT NULL
    ORDER BY dc.embedding <=> query_vector.query_emb
    LIMIT 10
),
reranked AS (
    SELECT 
        chunk_id,
        title,
        similarity,
        original_rank,
        -- Simple reranking: boost database category
        CASE 
            WHEN title ILIKE '%database%' THEN similarity * 1.2
            ELSE similarity
        END AS reranked_score,
        ROW_NUMBER() OVER (ORDER BY 
            CASE 
                WHEN title ILIKE '%database%' THEN similarity * 1.2
                ELSE similarity
            END DESC
        ) AS reranked_rank
    FROM original_ranking
)
SELECT 
    chunk_id,
    title,
    ROUND(similarity::numeric, 4) AS original_score,
    original_rank,
    ROUND(reranked_score::numeric, 4) AS reranked_score,
    reranked_rank,
    (original_rank - reranked_rank) AS rank_change
FROM reranked
ORDER BY reranked_rank;



