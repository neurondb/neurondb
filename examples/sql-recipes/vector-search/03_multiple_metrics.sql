-- ============================================================================
-- Recipe: Multiple Distance Metrics
-- ============================================================================
-- Purpose: Compare different distance metrics (L2, Cosine, Inner Product)
-- 
-- Prerequisites:
--   - quickstart_documents table
--   - Embeddings column populated
--
-- Distance Operators:
--   - <->  : L2 (Euclidean) distance
--   - <=>  : Cosine distance
--   - <#>  : Negative inner product (for maximum inner product search)
--
-- Use Cases:
--   - Compare metric performance
--   - Choose appropriate metric for your use case
--   - Combine multiple metrics
--
-- Performance Notes:
--   - Cosine is most common for text embeddings
--   - L2 works well for normalized embeddings
--   - Inner product for recommendation systems
-- ============================================================================

-- Example 1: Compare all three metrics
-- See how rankings differ between metrics
WITH query_vector AS (
    SELECT embed_text('machine learning') AS query_emb
)
SELECT 
    id,
    title,
    embedding <-> query_vector.query_emb AS l2_distance,
    embedding <=> query_vector.query_emb AS cosine_distance,
    embedding <#> query_vector.query_emb AS neg_inner_product,
    RANK() OVER (ORDER BY embedding <=> query_vector.query_emb) AS cosine_rank,
    RANK() OVER (ORDER BY embedding <-> query_vector.query_emb) AS l2_rank
FROM quickstart_documents, query_vector
WHERE embedding IS NOT NULL
ORDER BY embedding <=> query_vector.query_emb
LIMIT 5;

-- Example 2: L2 (Euclidean) Distance
-- Best for: General purpose, unnormalized vectors
WITH query_vector AS (
    SELECT embed_text('database performance') AS query_emb
)
SELECT 
    id,
    title,
    embedding <-> query_vector.query_emb AS l2_distance
FROM quickstart_documents, query_vector
WHERE embedding IS NOT NULL
ORDER BY embedding <-> query_vector.query_emb
LIMIT 5;

-- Example 3: Cosine Distance
-- Best for: Text embeddings (most common)
-- Returns distance; similarity = 1 - distance
WITH query_vector AS (
    SELECT embed_text('vector search') AS query_emb
)
SELECT 
    id,
    title,
    embedding <=> query_vector.query_emb AS cosine_distance,
    1 - (embedding <=> query_vector.query_emb) AS cosine_similarity
FROM quickstart_documents, query_vector
WHERE embedding IS NOT NULL
ORDER BY embedding <=> query_vector.query_emb
LIMIT 5;

-- Example 4: Inner Product (Maximum)
-- Best for: Recommendation systems, normalized embeddings
-- Higher inner product = more similar
-- Note: <#> returns negative inner product, so ORDER BY ASC gives max
WITH query_vector AS (
    SELECT embed_text('neural networks') AS query_emb
)
SELECT 
    id,
    title,
    embedding <#> query_vector.query_emb AS neg_inner_product,
    -(embedding <#> query_vector.query_emb) AS inner_product
FROM quickstart_documents, query_vector
WHERE embedding IS NOT NULL
ORDER BY embedding <#> query_vector.query_emb  -- ASC = maximum inner product
LIMIT 5;

-- Example 5: Hybrid ranking using multiple metrics
-- Combine cosine and L2 for potentially better results
WITH query_vector AS (
    SELECT embed_text('semantic search') AS query_emb
),
ranked AS (
    SELECT 
        id,
        title,
        embedding <=> query_vector.query_emb AS cosine_dist,
        embedding <-> query_vector.query_emb AS l2_dist,
        -- Normalized combined score (both metrics contribute equally)
        (embedding <=> query_vector.query_emb) / 
            (MAX(embedding <=> query_vector.query_emb) OVER ()) +
        (embedding <-> query_vector.query_emb) / 
            (MAX(embedding <-> query_vector.query_emb) OVER ()) AS combined_score
    FROM quickstart_documents, query_vector
    WHERE embedding IS NOT NULL
)
SELECT 
    id,
    title,
    ROUND(cosine_dist::numeric, 4) AS cosine_dist,
    ROUND(l2_dist::numeric, 4) AS l2_dist,
    ROUND(combined_score::numeric, 4) AS combined_score
FROM ranked
ORDER BY combined_score
LIMIT 5;



