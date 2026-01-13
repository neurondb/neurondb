-- ============================================================================
-- Recipe: Basic Vector Similarity Search
-- ============================================================================
-- Purpose: Find similar documents using cosine similarity
-- 
-- Prerequisites:
--   - quickstart_documents table (from quickstart data pack)
--   - OR documents table with embedding vector column
--   - embed_text() function available (or pre-computed query embedding)
--
-- Expected Results:
--   - Top 5 most similar documents to the query
--   - Similarity scores (1.0 = identical, 0.0 = orthogonal)
--
-- Performance Notes:
--   - Uses cosine distance (<=>) operator for normalized embeddings
--   - Works best with HNSW index on embedding column
-- ============================================================================

-- Example 1: Basic similarity search with text query
-- Converts query text to embedding, then finds similar documents
WITH query_vector AS (
    SELECT embed_text('machine learning embeddings') AS query_emb
)
SELECT 
    id,
    title,
    LEFT(content, 100) || '...' AS content_preview,
    1 - (embedding <=> query_vector.query_emb) AS similarity_score,
    embedding <=> query_vector.query_emb AS cosine_distance
FROM quickstart_documents, query_vector
WHERE embedding IS NOT NULL
ORDER BY embedding <=> query_vector.query_emb
LIMIT 5;

-- Example 2: Using pre-computed query vector
-- More efficient if you're running multiple searches with the same query
WITH query_vector AS (
    SELECT '[0.1,0.2,0.3,...]'::vector(384) AS query_emb  -- Your pre-computed vector
)
SELECT 
    id,
    title,
    1 - (embedding <=> query_vector.query_emb) AS similarity
FROM quickstart_documents, query_vector
ORDER BY embedding <=> query_vector.query_emb
LIMIT 5;

-- Example 3: Similarity search with threshold
-- Only return results above a similarity threshold
WITH query_vector AS (
    SELECT embed_text('vector databases') AS query_emb
)
SELECT 
    id,
    title,
    1 - (embedding <=> query_vector.query_emb) AS similarity
FROM quickstart_documents, query_vector
WHERE embedding IS NOT NULL
  AND (1 - (embedding <=> query_vector.query_emb)) > 0.7  -- 70% similarity threshold
ORDER BY embedding <=> query_vector.query_emb;

-- Example 4: Batch similarity search
-- Compare multiple queries at once
WITH queries AS (
    SELECT 
        query_text,
        embed_text(query_text) AS query_emb
    FROM (VALUES 
        ('vector databases'),
        ('machine learning'),
        ('database optimization')
    ) AS q(query_text)
)
SELECT 
    q.query_text,
    d.id,
    d.title,
    1 - (d.embedding <=> q.query_emb) AS similarity
FROM quickstart_documents d
CROSS JOIN queries q
WHERE d.embedding IS NOT NULL
ORDER BY q.query_text, similarity DESC
LIMIT 5;



