-- ============================================================================
-- SQL Recipe Library: Filtered Vector Search
-- ============================================================================
-- Ready-to-run queries for vector search with SQL filters
-- 
-- Prerequisites:
--   - NeuronDB extension installed
--   - Quickstart data pack loaded (or your own table with embeddings)
--
-- Usage:
--   psql -f 03_filtered_search.sql
--   Or copy individual queries to run them
--
-- Table: quickstart_documents (from quickstart data pack)
-- ============================================================================

-- ============================================================================
-- Setup: Add category column for filtering examples
-- ============================================================================
-- Add a category column to enable filtering examples
ALTER TABLE quickstart_documents ADD COLUMN IF NOT EXISTS category TEXT;

-- Populate category based on title (if not already populated)
UPDATE quickstart_documents
SET category = CASE
    WHEN title ILIKE '%Machine Learning%' OR title ILIKE '%Neural%' THEN 'ML'
    WHEN title ILIKE '%Language%' OR title ILIKE '%NLP%' OR title ILIKE '%Processing%' THEN 'NLP'
    WHEN title ILIKE '%Vision%' OR title ILIKE '%Image%' OR title ILIKE '%Computer%' THEN 'Vision'
    WHEN title ILIKE '%Learning%' OR title ILIKE '%Reinforcement%' THEN 'Learning'
    WHEN title ILIKE '%Database%' OR title ILIKE '%Vector%' OR title ILIKE '%Search%' THEN 'Database'
    WHEN title ILIKE '%PostgreSQL%' OR title ILIKE '%SQL%' THEN 'SQL'
    ELSE 'General'
END
WHERE category IS NULL;

-- ============================================================================
-- Recipe 1: Basic Filtered Vector Search
-- ============================================================================
-- Use case: Find similar documents within a specific category
-- Complexity: ⭐

-- Find top 10 similar documents, but only in the 'ML' category
SELECT 
    id,
    title,
    category,
    embedding <=> (SELECT embedding FROM quickstart_documents WHERE id = 1) AS distance
FROM quickstart_documents
WHERE id != 1
  AND category = 'ML'
ORDER BY embedding <=> (SELECT embedding FROM quickstart_documents WHERE id = 1)
LIMIT 10;

-- ============================================================================
-- Recipe 2: Multiple Filter Conditions
-- ============================================================================
-- Use case: Filter by multiple criteria before vector search
-- Complexity: ⭐⭐

-- Search in specific categories with text filter
SELECT 
    id,
    title,
    category,
    embedding <=> (SELECT embedding FROM quickstart_documents WHERE id = 1) AS distance
FROM quickstart_documents
WHERE id != 1
  AND category IN ('ML', 'NLP', 'Learning')
  AND title ILIKE '%learning%'
ORDER BY embedding <=> (SELECT embedding FROM quickstart_documents WHERE id = 1)
LIMIT 10;

-- ============================================================================
-- Recipe 3: Filtered Search with Distance Threshold
-- ============================================================================
-- Use case: Combine category filter with maximum distance threshold
-- Complexity: ⭐⭐

-- Only return results within distance threshold and specific category
SELECT 
    id,
    title,
    category,
    embedding <=> (SELECT embedding FROM quickstart_documents WHERE id = 1) AS distance
FROM quickstart_documents
WHERE id != 1
  AND category = 'Database'
  AND embedding <=> (SELECT embedding FROM quickstart_documents WHERE id = 1) < 0.8
ORDER BY embedding <=> (SELECT embedding FROM quickstart_documents WHERE id = 1)
LIMIT 20;

-- ============================================================================
-- Recipe 4: Filtered Search with Date Range (if date column exists)
-- ============================================================================
-- Use case: Filter by date range and then search
-- Complexity: ⭐⭐
-- Note: This example adds a date column for demonstration

-- Add date column if it doesn't exist
ALTER TABLE quickstart_documents ADD COLUMN IF NOT EXISTS created_at TIMESTAMP;

-- Populate with random dates if empty
UPDATE quickstart_documents
SET created_at = NOW() - (random() * interval '365 days')
WHERE created_at IS NULL;

-- Search in recent documents (last 30 days)
SELECT 
    id,
    title,
    category,
    created_at,
    embedding <=> (SELECT embedding FROM quickstart_documents WHERE id = 1) AS distance
FROM quickstart_documents
WHERE id != 1
  AND created_at >= NOW() - interval '30 days'
ORDER BY embedding <=> (SELECT embedding FROM quickstart_documents WHERE id = 1)
LIMIT 10;

-- ============================================================================
-- Recipe 5: Filtered Search with Text Matching
-- ============================================================================
-- Use case: Combine vector search with text pattern matching
-- Complexity: ⭐⭐

-- Search similar documents where title contains specific text
SELECT 
    id,
    title,
    category,
    embedding <=> (SELECT embedding FROM quickstart_documents WHERE id = 1) AS distance
FROM quickstart_documents
WHERE id != 1
  AND (title ILIKE '%neural%' OR title ILIKE '%network%' OR content ILIKE '%neural%')
ORDER BY embedding <=> (SELECT embedding FROM quickstart_documents WHERE id = 1)
LIMIT 10;

-- ============================================================================
-- Recipe 6: Top-K Per Category (Filtered Grouping)
-- ============================================================================
-- Use case: Get top-K similar documents per category
-- Complexity: ⭐⭐⭐

-- Get top 3 similar documents per category
WITH query_vector AS (
    SELECT embedding AS q_vec FROM quickstart_documents WHERE id = 1
),
ranked AS (
    SELECT 
        id,
        title,
        category,
        embedding <=> (SELECT q_vec FROM query_vector) AS distance,
        ROW_NUMBER() OVER (PARTITION BY category ORDER BY embedding <=> (SELECT q_vec FROM query_vector)) AS rn
    FROM quickstart_documents
    WHERE id != 1
)
SELECT 
    id,
    title,
    category,
    distance
FROM ranked
WHERE rn <= 3
ORDER BY category, distance
LIMIT 20;

-- ============================================================================
-- Recipe 7: Filtered Search with Aggregation
-- ============================================================================
-- Use case: Analyze similarity within filtered groups
-- Complexity: ⭐⭐⭐

-- Find average similarity per category
SELECT 
    category,
    COUNT(*) AS document_count,
    MIN(embedding <=> (SELECT embedding FROM quickstart_documents WHERE id = 1)) AS min_distance,
    MAX(embedding <=> (SELECT embedding FROM quickstart_documents WHERE id = 1)) AS max_distance,
    AVG(embedding <=> (SELECT embedding FROM quickstart_documents WHERE id = 1)) AS avg_distance
FROM quickstart_documents
WHERE id != 1
GROUP BY category
ORDER BY avg_distance;

-- ============================================================================
-- Recipe 8: Complex Filtered Search
-- ============================================================================
-- Use case: Multiple filters combined with vector search
-- Complexity: ⭐⭐⭐

-- Complex filter: category, text match, and distance threshold
SELECT 
    id,
    title,
    category,
    created_at,
    embedding <=> (SELECT embedding FROM quickstart_documents WHERE id = 1) AS distance
FROM quickstart_documents
WHERE id != 1
  AND category IN ('ML', 'NLP')
  AND (title ILIKE '%learning%' OR content ILIKE '%algorithm%')
  AND created_at >= NOW() - interval '180 days'
  AND embedding <=> (SELECT embedding FROM quickstart_documents WHERE id = 1) < 0.9
ORDER BY embedding <=> (SELECT embedding FROM quickstart_documents WHERE id = 1)
LIMIT 10;

-- ============================================================================
-- Recipe 9: Filtered Search with NOT Conditions
-- ============================================================================
-- Use case: Exclude specific categories or documents
-- Complexity: ⭐⭐

-- Search but exclude specific categories
SELECT 
    id,
    title,
    category,
    embedding <=> (SELECT embedding FROM quickstart_documents WHERE id = 1) AS distance
FROM quickstart_documents
WHERE id != 1
  AND category NOT IN ('SQL', 'General')
ORDER BY embedding <=> (SELECT embedding FROM quickstart_documents WHERE id = 1)
LIMIT 10;

-- ============================================================================
-- Recipe 10: Filtered Search with EXISTS/NOT EXISTS
-- ============================================================================
-- Use case: Filter based on related tables (if you have relationships)
-- Complexity: ⭐⭐⭐
-- Note: This is a template - adapt to your schema

-- Example: Filter documents that have related items
-- (This assumes you might have a related table - adjust to your schema)
-- SELECT 
--     d.id,
--     d.title,
--     d.category,
--     d.embedding <=> (SELECT embedding FROM quickstart_documents WHERE id = 1) AS distance
-- FROM quickstart_documents d
-- WHERE d.id != 1
--   AND EXISTS (
--       SELECT 1 FROM related_items r 
--       WHERE r.doc_id = d.id AND r.status = 'active'
--   )
-- ORDER BY d.embedding <=> (SELECT embedding FROM quickstart_documents WHERE id = 1)
-- LIMIT 10;

-- ============================================================================
-- Recipe 11: Filtered Search Performance Tips
-- ============================================================================
-- Use case: Optimize filtered vector search
-- Complexity: ⭐⭐⭐

-- Tip: Create indexes on filter columns for better performance
-- CREATE INDEX idx_documents_category ON quickstart_documents(category);
-- CREATE INDEX idx_documents_created_at ON quickstart_documents(created_at);

-- Tip: Use partial indexes for common filter patterns
-- CREATE INDEX idx_documents_ml_category ON quickstart_documents(category, id)
-- WHERE category = 'ML';

-- Combined index for common filter + vector search pattern
-- (HNSW index already exists on embedding, this adds filter index)
CREATE INDEX IF NOT EXISTS idx_documents_category_created 
ON quickstart_documents(category, created_at);

-- Example: This query will benefit from the index above
EXPLAIN ANALYZE
SELECT 
    id,
    title,
    category,
    embedding <=> (SELECT embedding FROM quickstart_documents WHERE id = 1) AS distance
FROM quickstart_documents
WHERE id != 1
  AND category = 'ML'
  AND created_at >= NOW() - interval '90 days'
ORDER BY embedding <=> (SELECT embedding FROM quickstart_documents WHERE id = 1)
LIMIT 10;

-- ============================================================================
-- Notes:
-- ============================================================================
-- 
-- 1. Filters are applied BEFORE vector search ordering, so they can
--    significantly reduce the search space and improve performance.
--
-- 2. Create indexes on filter columns (category, created_at, etc.) for
--    better performance with filtered searches.
--
-- 3. The HNSW index on embeddings works with filters - PostgreSQL will
--    use both indexes together when possible.
--
-- 4. For best performance, combine vector search index (HNSW) with
--    B-tree indexes on filter columns.
--
-- 5. Consider using partial indexes for common filter patterns to save
--    space and improve performance.
--
-- 6. Filtered search is useful for:
--    - Category-based recommendations
--    - Time-based search (recent items)
--    - User-specific filtering
--    - Excluding unwanted results
--
-- ============================================================================



