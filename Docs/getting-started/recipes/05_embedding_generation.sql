-- ============================================================================
-- SQL Recipe Library: Embedding Generation
-- ============================================================================
-- Ready-to-run queries for generating embeddings from text
-- 
-- Prerequisites:
--   - NeuronDB extension installed
--   - Embedding model configured (default: all-MiniLM-L6-v2)
--
-- Usage:
--   psql -f 05_embedding_generation.sql
--   Or copy individual recipes to run them
--
-- Note: These recipes use embed_text() which requires embedding model setup.
--       For quickstart, you can use pre-generated embeddings instead.
-- ============================================================================

-- ============================================================================
-- Recipe 1: Generate Single Embedding
-- ============================================================================
-- Use case: Generate embedding for a single piece of text
-- Complexity: ⭐

-- Generate embedding from text (uses default model)
SELECT embed_text('machine learning algorithms') AS embedding;

-- Generate embedding with specific model
SELECT embed_text('machine learning algorithms', 'all-MiniLM-L6-v2') AS embedding;

-- ============================================================================
-- Recipe 2: Insert Document with Generated Embedding
-- ============================================================================
-- Use case: Insert new document and generate its embedding in one step
-- Complexity: ⭐

-- Create example table
CREATE TABLE IF NOT EXISTS my_documents (
    id SERIAL PRIMARY KEY,
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    embedding vector(384)  -- Adjust dimension based on model
);

-- Insert document with generated embedding
INSERT INTO my_documents (title, content, embedding)
VALUES (
    'Introduction to AI',
    'Artificial intelligence is transforming technology',
    embed_text('Artificial intelligence is transforming technology')
);

-- ============================================================================
-- Recipe 3: Batch Embedding Generation (Single INSERT)
-- ============================================================================
-- Use case: Generate embeddings for multiple documents at once
-- Complexity: ⭐⭐

-- Insert multiple documents with embeddings
INSERT INTO my_documents (title, content, embedding)
VALUES 
    ('Machine Learning', 'Machine learning is a subset of AI', 
     embed_text('Machine learning is a subset of AI')),
    ('Deep Learning', 'Deep learning uses neural networks', 
     embed_text('Deep learning uses neural networks')),
    ('Natural Language Processing', 'NLP enables computers to understand language', 
     embed_text('NLP enables computers to understand language'));

-- ============================================================================
-- Recipe 4: Update Existing Documents with Embeddings
-- ============================================================================
-- Use case: Generate embeddings for existing documents that don't have them
-- Complexity: ⭐⭐

-- Update documents without embeddings
UPDATE my_documents
SET embedding = embed_text(content)
WHERE embedding IS NULL;

-- ============================================================================
-- Recipe 5: Batch Embedding Generation (SELECT + INSERT)
-- ============================================================================
-- Use case: Generate embeddings from source table and insert into target table
-- Complexity: ⭐⭐

-- Generate embeddings from source and insert into target
INSERT INTO my_documents (title, content, embedding)
SELECT 
    title,
    content,
    embed_text(content) AS embedding
FROM source_documents
WHERE embedding IS NULL;

-- ============================================================================
-- Recipe 6: Generate Query Embedding for Search
-- ============================================================================
-- Use case: Generate embedding from search query for similarity search
-- Complexity: ⭐

-- Generate query embedding and use in search
WITH query_embedding AS (
    SELECT embed_text('machine learning') AS q_vec
)
SELECT 
    id,
    title,
    embedding <=> q.q_vec AS distance
FROM my_documents, query_embedding q
ORDER BY embedding <=> q.q_vec
LIMIT 10;

-- ============================================================================
-- Recipe 7: Generate Embeddings with Different Models
-- ============================================================================
-- Use case: Compare embeddings from different models
-- Complexity: ⭐⭐
-- Note: Model availability depends on your NeuronDB configuration

-- Generate embeddings with different models
SELECT 
    'all-MiniLM-L6-v2' AS model,
    embed_text('machine learning', 'all-MiniLM-L6-v2') AS embedding_1
UNION ALL
SELECT 
    'all-mpnet-base-v2' AS model,
    embed_text('machine learning', 'all-mpnet-base-v2') AS embedding_2;

-- ============================================================================
-- Recipe 8: Batch Embedding with Progress Tracking
-- ============================================================================
-- Use case: Generate embeddings in batches with progress monitoring
-- Complexity: ⭐⭐⭐

-- Create function to track progress (if needed)
-- Update embeddings in batches of 100
DO $$
DECLARE
    batch_size INTEGER := 100;
    total_rows INTEGER;
    processed_rows INTEGER := 0;
BEGIN
    -- Get total count
    SELECT COUNT(*) INTO total_rows
    FROM my_documents
    WHERE embedding IS NULL;
    
    -- Process in batches
    WHILE processed_rows < total_rows LOOP
        UPDATE my_documents
        SET embedding = embed_text(content)
        WHERE embedding IS NULL
        AND id IN (
            SELECT id FROM my_documents
            WHERE embedding IS NULL
            ORDER BY id
            LIMIT batch_size
        );
        
        processed_rows := processed_rows + batch_size;
        RAISE NOTICE 'Processed % rows of %', processed_rows, total_rows;
    END LOOP;
END $$;

-- ============================================================================
-- Recipe 9: Verify Embedding Generation
-- ============================================================================
-- Use case: Check that embeddings were generated correctly
-- Complexity: ⭐⭐

-- Check embedding statistics
SELECT 
    COUNT(*) AS total_documents,
    COUNT(embedding) AS documents_with_embeddings,
    COUNT(*) FILTER (WHERE embedding IS NULL) AS missing_embeddings,
    -- Check dimensions (adjust based on your model)
    MIN(array_length(vector_to_array(embedding), 1)) AS min_dimensions,
    MAX(array_length(vector_to_array(embedding), 1)) AS max_dimensions
FROM my_documents;

-- View sample embeddings (first 10 dimensions)
SELECT 
    id,
    title,
    LEFT(content, 50) AS content_preview,
    (vector_to_array(embedding))[1:10] AS embedding_preview
FROM my_documents
WHERE embedding IS NOT NULL
LIMIT 5;

-- ============================================================================
-- Recipe 10: Generate Embeddings with Text Preprocessing
-- ============================================================================
-- Use case: Clean or preprocess text before generating embeddings
-- Complexity: ⭐⭐

-- Generate embeddings with cleaned/preprocessed text
INSERT INTO my_documents (title, content, embedding)
SELECT 
    title,
    content,
    embed_text(
        -- Example: Convert to lowercase and remove extra spaces
        regexp_replace(lower(content), '\s+', ' ', 'g')
    ) AS embedding
FROM source_documents
WHERE embedding IS NULL;

-- ============================================================================
-- Recipe 11: Generate Embeddings for Multiple Columns
-- ============================================================================
-- Use case: Create embeddings from title + content combination
-- Complexity: ⭐⭐

-- Generate embedding from combined title and content
UPDATE my_documents
SET embedding = embed_text(title || ' ' || content)
WHERE embedding IS NULL;

-- Or create separate embeddings for title and content
ALTER TABLE my_documents ADD COLUMN IF NOT EXISTS title_embedding vector(384);

UPDATE my_documents
SET title_embedding = embed_text(title)
WHERE title_embedding IS NULL;

-- ============================================================================
-- Recipe 12: Embedding Generation Performance Tips
-- ============================================================================
-- Use case: Optimize embedding generation for large datasets
-- Complexity: ⭐⭐⭐

-- Tip 1: Use batch processing for better performance
-- Process in batches of 100-1000 depending on available memory

-- Tip 2: Generate embeddings during insert when possible
INSERT INTO my_documents (title, content, embedding)
SELECT title, content, embed_text(content)
FROM source_documents;

-- Tip 3: Use background workers for large batches (if available)
-- See NeuronDB worker documentation for background embedding generation

-- Tip 4: Cache frequently used embeddings
-- NeuronDB automatically caches embeddings - reuse when possible

-- ============================================================================
-- Recipe 13: Check Embedding Cache Statistics
-- ============================================================================
-- Use case: Monitor embedding generation caching
-- Complexity: ⭐⭐

-- View embedding cache statistics (if available)
-- SELECT * FROM neurondb.embedding_cache_stats;

-- ============================================================================
-- Recipe 14: Error Handling for Embedding Generation
-- ============================================================================
-- Use case: Handle errors gracefully during batch generation
-- Complexity: ⭐⭐⭐

-- Example: Update with error handling
DO $$
DECLARE
    doc_record RECORD;
BEGIN
    FOR doc_record IN 
        SELECT id, content 
        FROM my_documents 
        WHERE embedding IS NULL
    LOOP
        BEGIN
            UPDATE my_documents
            SET embedding = embed_text(doc_record.content)
            WHERE id = doc_record.id;
        EXCEPTION
            WHEN OTHERS THEN
                RAISE NOTICE 'Failed to generate embedding for document %: %', 
                    doc_record.id, SQLERRM;
        END;
    END LOOP;
END $$;

-- ============================================================================
-- Notes:
-- ============================================================================
-- 
-- 1. embed_text() is an alias for neurondb_embed() - both functions work.
--
-- 2. Default model: 'all-MiniLM-L6-v2' (384 dimensions)
--    - Fast, good quality for most use cases
--    - Change model by passing as second parameter
--
-- 3. Model dimensions:
--    - all-MiniLM-L6-v2: 384 dimensions
--    - all-mpnet-base-v2: 768 dimensions
--    - text-embedding-ada-002: 1536 dimensions (OpenAI)
--    - Adjust vector column dimension to match model
--
-- 4. Performance:
--    - Batch generation is more efficient than individual calls
--    - Embeddings are automatically cached
--    - Use background workers for very large batches
--
-- 5. Error handling:
--    - Network timeouts may occur with external API models
--    - Handle errors gracefully in batch processing
--    - Check embedding cache for frequently used text
--
-- 6. Best practices:
--    - Generate embeddings during data insertion when possible
--    - Use batch processing for large datasets
--    - Monitor embedding cache hit rates
--    - Choose model dimensions based on your use case
--
-- 7. For quickstart examples (without embedding generation):
--    - Use pre-generated embeddings from quickstart data pack
--    - Or use array_to_vector() with pre-computed embeddings
--
-- ============================================================================



