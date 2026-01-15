-- ============================================================================
-- Recipe: Faceted Hybrid Search
-- ============================================================================
-- Purpose: Combine vector/text search with faceted filtering by metadata
-- 
-- Prerequisites:
--   - quickstart_documents table with category and tags columns
--   - Full-text search index
--   - Embeddings column populated
--
-- Use Cases:
--   - E-commerce product search
--   - Content filtering by category/tags
--   - Multi-dimensional search results
--
-- Performance Notes:
--   - Facets computed on filtered result set
--   - Consider indexes on facet columns
-- ============================================================================

-- Example 1: Search with category facets
-- Return search results and category distribution
WITH query_vector AS (
    SELECT embed_text('database performance') AS query_emb
),
search_results AS (
    SELECT 
        id,
        title,
        category,
        1 - (embedding <=> query_vector.query_emb) AS similarity
    FROM quickstart_documents, query_vector
    WHERE embedding IS NOT NULL
    ORDER BY embedding <=> query_vector.query_emb
    LIMIT 20
),
category_facets AS (
    SELECT 
        category,
        COUNT(*) AS count,
        MAX(similarity) AS max_similarity
    FROM search_results
    WHERE category IS NOT NULL
    GROUP BY category
    ORDER BY count DESC
)
SELECT 
    'Results' AS type,
    NULL::text AS category,
    COUNT(*)::text AS count,
    NULL::numeric AS max_similarity
FROM search_results
UNION ALL
SELECT 
    'Facet' AS type,
    category,
    count::text,
    max_similarity
FROM category_facets;

-- Example 2: Filter by facet, then search
-- Common pattern: user selects category, then searches
WITH query_vector AS (
    SELECT embed_text('vector algorithms') AS query_emb
),
filtered_docs AS (
    SELECT *
    FROM quickstart_documents
    WHERE category = 'algorithms'  -- User-selected facet
      AND embedding IS NOT NULL
)
SELECT 
    id,
    title,
    category,
    1 - (embedding <=> query_vector.query_emb) AS similarity
FROM filtered_docs, query_vector
ORDER BY embedding <=> query_vector.query_emb
LIMIT 10;

-- Example 3: Multi-faceted search
-- Filter by multiple facets (category + tags)
WITH query_vector AS (
    SELECT embed_text('machine learning') AS query_emb
),
filtered_docs AS (
    SELECT *
    FROM quickstart_documents
    WHERE category = 'machine_learning'
      AND 'neural_networks' = ANY(tags)
      AND embedding IS NOT NULL
)
SELECT 
    id,
    title,
    category,
    tags,
    1 - (embedding <=> query_vector.query_emb) AS similarity
FROM filtered_docs, query_vector
ORDER BY embedding <=> query_vector.query_emb
LIMIT 10;

-- Example 4: Dynamic facets based on search results
-- Show available facets for current search results
WITH query_vector AS (
    SELECT embed_text('database indexing') AS query_emb
),
top_results AS (
    SELECT 
        id,
        category,
        tags,
        1 - (embedding <=> query_vector.query_emb) AS similarity
    FROM quickstart_documents, query_vector
    WHERE embedding IS NOT NULL
    ORDER BY embedding <=> query_vector.query_emb
    LIMIT 50  -- Consider top 50 for facet calculation
),
category_facets AS (
    SELECT 
        category,
        COUNT(*) AS doc_count,
        AVG(similarity) AS avg_similarity
    FROM top_results
    WHERE category IS NOT NULL
    GROUP BY category
    ORDER BY doc_count DESC
),
tag_facets AS (
    SELECT 
        unnest(tags) AS tag,
        COUNT(*) AS doc_count,
        AVG(similarity) AS avg_similarity
    FROM top_results
    WHERE tags IS NOT NULL
    GROUP BY unnest(tags)
    ORDER BY doc_count DESC
    LIMIT 10
)
SELECT 
    'category' AS facet_type,
    category AS facet_value,
    doc_count,
    ROUND(avg_similarity::numeric, 4) AS avg_similarity
FROM category_facets
UNION ALL
SELECT 
    'tag' AS facet_type,
    tag AS facet_value,
    doc_count,
    ROUND(avg_similarity::numeric, 4) AS avg_similarity
FROM tag_facets
ORDER BY facet_type, doc_count DESC;

-- Example 5: Hybrid search with faceted filtering
-- Combine text + vector search, then apply facets
WITH query_vector AS (
    SELECT embed_text('PostgreSQL optimization') AS query_emb
),
hybrid_results AS (
    SELECT 
        d.id,
        d.title,
        d.category,
        d.tags,
        -- Combined score from vector and text
        COALESCE(
            (1 - (d.embedding <=> query_vector.query_emb)) * 0.7 +
            ts_rank(d.fts_vector, plainto_tsquery('english', 'PostgreSQL optimization')) * 0.3,
            0
        ) AS hybrid_score
    FROM quickstart_documents d, query_vector
    WHERE d.embedding IS NOT NULL
    ORDER BY hybrid_score DESC
    LIMIT 50
),
faceted_results AS (
    SELECT 
        hr.*,
        -- Apply facet boost: increase score for matching facets
        CASE 
            WHEN hr.category = 'database' THEN hr.hybrid_score * 1.2
            ELSE hr.hybrid_score
        END AS boosted_score
    FROM hybrid_results hr
)
SELECT 
    id,
    title,
    category,
    ROUND(hybrid_score::numeric, 4) AS hybrid_score,
    ROUND(boosted_score::numeric, 4) AS boosted_score
FROM faceted_results
ORDER BY boosted_score DESC
LIMIT 10;




