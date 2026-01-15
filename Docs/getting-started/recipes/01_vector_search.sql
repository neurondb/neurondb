-- ============================================================================
-- SQL Recipe Library: Vector Search
-- ============================================================================
-- Ready-to-run queries for vector similarity search
-- 
-- Prerequisites:
--   - NeuronDB extension installed
--   - Quickstart data pack loaded (or your own table with embeddings)
--
-- Usage:
--   psql -f 01_vector_search.sql
--   Or copy individual queries to run them
--
-- Table: quickstart_documents (from quickstart data pack)
-- ============================================================================

-- ============================================================================
-- Recipe 1: Basic Cosine Similarity Search
-- ============================================================================
-- Use case: Find documents most similar to a query vector
-- Complexity: ⭐
-- Distance metric: Cosine distance (<=> operator)

-- Find top 10 documents most similar to document #1
SELECT 
    id,
    title,
    embedding <=> (SELECT embedding FROM quickstart_documents WHERE id = 1) AS distance
FROM quickstart_documents
WHERE id != 1
ORDER BY embedding <=> (SELECT embedding FROM quickstart_documents WHERE id = 1)
LIMIT 10;

-- ============================================================================
-- Recipe 2: Basic L2 (Euclidean) Distance Search
-- ============================================================================
-- Use case: Find documents closest by L2 distance
-- Complexity: ⭐
-- Distance metric: L2/Euclidean distance (<-> operator)

-- Find top 10 documents closest to document #1 using L2 distance
SELECT 
    id,
    title,
    embedding <-> (SELECT embedding FROM quickstart_documents WHERE id = 1) AS distance
FROM quickstart_documents
WHERE id != 1
ORDER BY embedding <-> (SELECT embedding FROM quickstart_documents WHERE id = 1)
LIMIT 10;

-- ============================================================================
-- Recipe 3: Similarity Search with Query Vector
-- ============================================================================
-- Use case: Search using a manually constructed query vector
-- Complexity: ⭐⭐
-- Note: In practice, use embed_text() to generate query vectors from text

-- Example: Search with a query vector (384 dimensions for quickstart data)
WITH query_vector AS (
    SELECT array_agg(random()::real ORDER BY j)::real[]::vector(384) AS q_vec
    FROM generate_series(1, 384) j
)
SELECT 
    id,
    title,
    embedding <=> q.q_vec AS distance
FROM quickstart_documents, query_vector q
ORDER BY embedding <=> q.q_vec
LIMIT 10;

-- ============================================================================
-- Recipe 4: Similarity Search with Distance Threshold
-- ============================================================================
-- Use case: Find documents within a maximum distance threshold
-- Complexity: ⭐⭐

-- Find documents within distance threshold of 0.5
SELECT 
    id,
    title,
    embedding <=> (SELECT embedding FROM quickstart_documents WHERE id = 1) AS distance
FROM quickstart_documents
WHERE id != 1
  AND embedding <=> (SELECT embedding FROM quickstart_documents WHERE id = 1) < 0.5
ORDER BY embedding <=> (SELECT embedding FROM quickstart_documents WHERE id = 1)
LIMIT 20;

-- ============================================================================
-- Recipe 5: Inner Product Search
-- ============================================================================
-- Use case: Find documents with highest inner product (useful for normalized vectors)
-- Complexity: ⭐⭐
-- Distance metric: Inner product (<#> operator, returns negative for ordering)

-- Find top 10 documents by inner product
WITH query_vector AS (
    SELECT array_agg(random()::real ORDER BY j)::real[]::vector(384) AS q_vec
    FROM generate_series(1, 384) j
)
SELECT 
    id,
    title,
    embedding <#> q.q_vec AS neg_inner_product,
    -(embedding <#> q.q_vec) AS inner_product
FROM quickstart_documents, query_vector q
ORDER BY embedding <#> q.q_vec
LIMIT 10;

-- ============================================================================
-- Recipe 6: Similarity Search with Ranking
-- ============================================================================
-- Use case: Get ranked results with row numbers
-- Complexity: ⭐⭐

-- Ranked similarity search with row numbers
SELECT 
    id,
    title,
    embedding <=> (SELECT embedding FROM quickstart_documents WHERE id = 1) AS distance,
    RANK() OVER (ORDER BY embedding <=> (SELECT embedding FROM quickstart_documents WHERE id = 1)) AS rank,
    ROW_NUMBER() OVER (ORDER BY embedding <=> (SELECT embedding FROM quickstart_documents WHERE id = 1)) AS row_num
FROM quickstart_documents
WHERE id != 1
ORDER BY embedding <=> (SELECT embedding FROM quickstart_documents WHERE id = 1)
LIMIT 10;

-- ============================================================================
-- Recipe 7: Similarity Search with Aggregation
-- ============================================================================
-- Use case: Analyze similarity distribution
-- Complexity: ⭐⭐⭐

-- Analyze similarity statistics
SELECT 
    COUNT(*) AS total_documents,
    MIN(embedding <=> (SELECT embedding FROM quickstart_documents WHERE id = 1)) AS min_distance,
    MAX(embedding <=> (SELECT embedding FROM quickstart_documents WHERE id = 1)) AS max_distance,
    AVG(embedding <=> (SELECT embedding FROM quickstart_documents WHERE id = 1)) AS avg_distance,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY embedding <=> (SELECT embedding FROM quickstart_documents WHERE id = 1)) AS median_distance
FROM quickstart_documents
WHERE id != 1;

-- ============================================================================
-- Recipe 8: Batch Similarity Search (Multiple Query Vectors)
-- ============================================================================
-- Use case: Compare multiple query vectors against documents
-- Complexity: ⭐⭐⭐

-- Compare document #1 and #2 as query vectors
WITH queries AS (
    SELECT 1 AS query_id, (SELECT embedding FROM quickstart_documents WHERE id = 1) AS q_vec
    UNION ALL
    SELECT 2 AS query_id, (SELECT embedding FROM quickstart_documents WHERE id = 2) AS q_vec
)
SELECT 
    q.query_id AS query_document_id,
    d.id AS document_id,
    d.title,
    d.embedding <=> q.q_vec AS distance
FROM quickstart_documents d
CROSS JOIN queries q
WHERE d.id NOT IN (q.query_id)
ORDER BY q.query_id, distance
LIMIT 10;

-- ============================================================================
-- Recipe 9: Similarity Search Performance (with Timing)
-- ============================================================================
-- Use case: Benchmark query performance
-- Complexity: ⭐⭐

-- Enable timing to measure query performance
\timing on

-- Run similarity search
SELECT 
    id,
    title,
    embedding <=> (SELECT embedding FROM quickstart_documents WHERE id = 1) AS distance
FROM quickstart_documents
WHERE id != 1
ORDER BY embedding <=> (SELECT embedding FROM quickstart_documents WHERE id = 1)
LIMIT 10;

\timing off

-- ============================================================================
-- Recipe 10: Top-K Search with Multiple Metrics
-- ============================================================================
-- Use case: Compare different distance metrics on the same query
-- Complexity: ⭐⭐⭐

-- Compare cosine, L2, and inner product for the same query
WITH query_doc AS (
    SELECT embedding AS q_vec FROM quickstart_documents WHERE id = 1
)
SELECT 
    d.id,
    d.title,
    d.embedding <=> q.q_vec AS cosine_distance,
    d.embedding <-> q.q_vec AS l2_distance,
    -(d.embedding <#> q.q_vec) AS inner_product
FROM quickstart_documents d, query_doc q
WHERE d.id != 1
ORDER BY d.embedding <=> q.q_vec
LIMIT 10;

-- ============================================================================
-- Notes:
-- ============================================================================
-- 
-- 1. These recipes assume the quickstart_documents table exists.
--    If using your own table, replace 'quickstart_documents' with your table name.
--
-- 2. In production, use embed_text() to generate query vectors from text:
--    WITH q AS (SELECT embed_text('your search query') AS query_vec)
--    SELECT ... FROM your_table, q
--    ORDER BY embedding <=> q.query_vec
--
-- 3. Distance operators:
--    - <=> : Cosine distance (lower = more similar)
--    - <-> : L2/Euclidean distance (lower = more similar)
--    - <#> : Negative inner product (lower = higher similarity for normalized vectors)
--
-- 4. For better performance, ensure you have an HNSW index on your embedding column.
--    See recipes/04_indexing.sql for index creation recipes.
--
-- ============================================================================




