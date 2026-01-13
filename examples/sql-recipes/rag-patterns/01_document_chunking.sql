-- ============================================================================
-- Recipe: Document Chunking for RAG
-- ============================================================================
-- Purpose: Split documents into chunks for RAG (Retrieval-Augmented Generation)
-- 
-- Prerequisites:
--   - Documents table with text content
--   - Understanding of your documents' structure
--
-- Chunking Strategies:
--   1. Fixed-size chunks (simple, fast)
--   2. Sentence-based chunks (respects sentence boundaries)
--   3. Semantic chunks (groups related sentences)
--   4. Sliding window (overlapping chunks for context)
--
-- Performance Notes:
--   - Chunk size: 200-500 tokens typically works well
--   - Overlap: 20-50 tokens helps maintain context
--   - Too small: May lose context
--   - Too large: May dilute relevance
-- ============================================================================

-- Example 1: Simple fixed-size chunking
-- Split documents into fixed-size chunks
CREATE TABLE IF NOT EXISTS document_chunks (
    chunk_id SERIAL PRIMARY KEY,
    doc_id INTEGER,
    chunk_index INTEGER,
    chunk_text TEXT NOT NULL,
    chunk_tokens INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Chunk a document into 500-character pieces
INSERT INTO document_chunks (doc_id, chunk_index, chunk_text, chunk_tokens)
SELECT 
    id AS doc_id,
    ROW_NUMBER() OVER (PARTITION BY id ORDER BY chunk_num) - 1 AS chunk_index,
    chunk_text,
    array_length(regexp_split_to_array(chunk_text, '\s+'), 1) AS chunk_tokens
FROM (
    SELECT 
        id,
        generate_series(1, (length(content) / 500) + 1) AS chunk_num,
        substring(content FROM ((generate_series(1, (length(content) / 500) + 1) - 1) * 500 + 1) FOR 500) AS chunk_text
    FROM quickstart_documents
    WHERE content IS NOT NULL
) chunks
WHERE chunk_text IS NOT NULL AND length(chunk_text) > 50;

-- Example 2: Sentence-based chunking
-- Split at sentence boundaries (more natural)
INSERT INTO document_chunks (doc_id, chunk_index, chunk_text, chunk_tokens)
SELECT 
    id AS doc_id,
    ROW_NUMBER() OVER (PARTITION BY id ORDER BY sentence_num) - 1 AS chunk_index,
    sentence AS chunk_text,
    array_length(regexp_split_to_array(sentence, '\s+'), 1) AS chunk_tokens
FROM (
    SELECT 
        id,
        ROW_NUMBER() OVER (PARTITION BY id) AS sentence_num,
        regexp_split_to_table(content, '[.!?]+') AS sentence
    FROM quickstart_documents
    WHERE content IS NOT NULL
) sentences
WHERE length(trim(sentence)) > 20;

-- Example 3: Overlapping chunks (sliding window)
-- Creates overlapping chunks for better context preservation
-- Chunk size: 500 chars, Overlap: 100 chars
WITH chunked AS (
    SELECT 
        id AS doc_id,
        content,
        generate_series(1, (length(content) / 400) + 1) AS chunk_num
    FROM quickstart_documents
    WHERE content IS NOT NULL
)
INSERT INTO document_chunks (doc_id, chunk_index, chunk_text, chunk_tokens)
SELECT 
    doc_id,
    chunk_num - 1 AS chunk_index,
    substring(content FROM ((chunk_num - 1) * 400 + 1) FOR 500) AS chunk_text,
    array_length(regexp_split_to_array(
        substring(content FROM ((chunk_num - 1) * 400 + 1) FOR 500), 
        '\s+'
    ), 1) AS chunk_tokens
FROM chunked
WHERE substring(content FROM ((chunk_num - 1) * 400 + 1) FOR 500) IS NOT NULL
  AND length(substring(content FROM ((chunk_num - 1) * 400 + 1) FOR 500)) > 50;

-- Example 4: Token-aware chunking
-- Approximate token counting (roughly 1 token = 4 characters)
INSERT INTO document_chunks (doc_id, chunk_index, chunk_text, chunk_tokens)
SELECT 
    id AS doc_id,
    ROW_NUMBER() OVER (PARTITION BY id ORDER BY chunk_num) - 1 AS chunk_index,
    chunk_text,
    (length(chunk_text) / 4)::integer AS chunk_tokens  -- Rough token estimate
FROM (
    SELECT 
        id,
        generate_series(1, (length(content) / 2000) + 1) AS chunk_num,
        substring(content FROM ((generate_series(1, (length(content) / 2000) + 1) - 1) * 2000 + 1) FOR 2000) AS chunk_text
    FROM quickstart_documents
    WHERE content IS NOT NULL
) chunks
WHERE chunk_text IS NOT NULL AND length(chunk_text) > 100;

-- Example 5: Add embeddings to chunks
-- Generate embeddings for all chunks
ALTER TABLE document_chunks ADD COLUMN IF NOT EXISTS embedding vector(384);

UPDATE document_chunks
SET embedding = embed_text(chunk_text)
WHERE embedding IS NULL
  AND chunk_text IS NOT NULL;

-- Example 6: Verify chunking results
SELECT 
    COUNT(*) AS total_chunks,
    COUNT(DISTINCT doc_id) AS documents_chunked,
    AVG(chunk_tokens) AS avg_tokens_per_chunk,
    MIN(chunk_tokens) AS min_tokens,
    MAX(chunk_tokens) AS max_tokens,
    COUNT(*) FILTER (WHERE embedding IS NOT NULL) AS chunks_with_embeddings
FROM document_chunks;

-- Example 7: View sample chunks
SELECT 
    dc.chunk_id,
    d.title AS document_title,
    dc.chunk_index,
    LEFT(dc.chunk_text, 100) || '...' AS chunk_preview,
    dc.chunk_tokens,
    CASE WHEN dc.embedding IS NOT NULL THEN 'Yes' ELSE 'No' END AS has_embedding
FROM document_chunks dc
JOIN quickstart_documents d ON dc.doc_id = d.id
ORDER BY dc.doc_id, dc.chunk_index
LIMIT 10;



