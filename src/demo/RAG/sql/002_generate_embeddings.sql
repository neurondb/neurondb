-- ============================================================================
-- Test 002: Generate Vector Embeddings
-- ============================================================================
-- Demonstrates: Text embedding generation, vector storage
-- ============================================================================

\echo 'Generating embeddings for all document chunks...'

-- Update chunks with embeddings using NeuronDB embedding functions
-- Using sentence-transformers/all-MiniLM-L6-v2 (384 dimensions)

UPDATE document_chunks
SET embedding = embed_text(chunk_text, 'sentence-transformers/all-MiniLM-L6-v2'::text)
WHERE embedding IS NULL;

\echo 'Verifying embeddings...'

SELECT 
    COUNT(*) AS chunks_with_embeddings,
    COUNT(*) FILTER (WHERE embedding IS NOT NULL) AS embedded_chunks,
    COUNT(*) FILTER (WHERE embedding IS NULL) AS missing_embeddings
FROM document_chunks;

\echo ''
\echo 'Sample embedding vectors (first 10 dimensions):'
SELECT 
    chunk_id,
    left(chunk_text, 60) || '...' AS text_preview,
    (vector_to_array(embedding))[1:10] AS embedding_preview
FROM document_chunks
WHERE embedding IS NOT NULL
LIMIT 3;

\echo ''
\echo 'Embedding statistics:'
SELECT 
    MIN(vector_dims(embedding)) AS min_dimensions,
    MAX(vector_dims(embedding)) AS max_dimensions,
    AVG(vector_dims(embedding)) AS avg_dimensions
FROM document_chunks
WHERE embedding IS NOT NULL;

