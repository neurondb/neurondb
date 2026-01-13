-- ============================================================================
-- Recipe: Filtered Vector Similarity Search
-- ============================================================================
-- Purpose: Combine vector similarity search with metadata filtering
-- 
-- Prerequisites:
--   - quickstart_documents table with category, tags columns
--   - Embeddings column populated
--
-- Use Cases:
--   - Search within specific categories
--   - Filter by tags or metadata
--   - Date range filtering
--   - Combining multiple filters
--
-- Performance Notes:
--   - Filters applied after vector search for best performance
--   - Consider partial indexes on filter columns
-- ============================================================================

-- Example 1: Filter by category
-- Find similar documents within a specific category
WITH query_vector AS (
    SELECT embed_text('database indexing') AS query_emb
)
SELECT 
    id,
    title,
    category,
    1 - (embedding <=> query_vector.query_emb) AS similarity
FROM quickstart_documents, query_vector
WHERE embedding IS NOT NULL
  AND category = 'database'  -- Filter by category
ORDER BY embedding <=> query_vector.query_emb
LIMIT 5;

-- Example 2: Filter by tags
-- Search within documents that have specific tags
WITH query_vector AS (
    SELECT embed_text('vector similarity') AS query_emb
)
SELECT 
    id,
    title,
    tags,
    1 - (embedding <=> query_vector.query_emb) AS similarity
FROM quickstart_documents, query_vector
WHERE embedding IS NOT NULL
  AND 'vectors' = ANY(tags)  -- Documents with 'vectors' tag
ORDER BY embedding <=> query_vector.query_emb
LIMIT 5;

-- Example 3: Multiple filters
-- Combine category and tag filtering
WITH query_vector AS (
    SELECT embed_text('neural networks') AS query_emb
)
SELECT 
    id,
    title,
    category,
    tags,
    1 - (embedding <=> query_vector.query_emb) AS similarity
FROM quickstart_documents, query_vector
WHERE embedding IS NOT NULL
  AND category = 'machine_learning'
  AND 'neural_networks' = ANY(tags)
ORDER BY embedding <=> query_vector.query_emb
LIMIT 5;

-- Example 4: Filter with date range
-- Search within documents from a specific time period
WITH query_vector AS (
    SELECT embed_text('database optimization') AS query_emb
)
SELECT 
    id,
    title,
    created_at,
    1 - (embedding <=> query_vector.query_emb) AS similarity
FROM quickstart_documents, query_vector
WHERE embedding IS NOT NULL
  AND created_at >= CURRENT_DATE - INTERVAL '30 days'
ORDER BY embedding <=> query_vector.query_emb
LIMIT 5;

-- Example 5: Filtered search with OR conditions
-- Multiple category options
WITH query_vector AS (
    SELECT embed_text('search algorithms') AS query_emb
)
SELECT 
    id,
    title,
    category,
    1 - (embedding <=> query_vector.query_emb) AS similarity
FROM quickstart_documents, query_vector
WHERE embedding IS NOT NULL
  AND category IN ('algorithms', 'search', 'database')
ORDER BY embedding <=> query_vector.query_emb
LIMIT 5;

-- Example 6: Filtered search with threshold
-- Combine similarity threshold with metadata filters
WITH query_vector AS (
    SELECT embed_text('vector quantization') AS query_emb
)
SELECT 
    id,
    title,
    category,
    1 - (embedding <=> query_vector.query_emb) AS similarity
FROM quickstart_documents, query_vector
WHERE embedding IS NOT NULL
  AND category = 'algorithms'
  AND (1 - (embedding <=> query_vector.query_emb)) > 0.6  -- Similarity threshold
ORDER BY embedding <=> query_vector.query_emb;



