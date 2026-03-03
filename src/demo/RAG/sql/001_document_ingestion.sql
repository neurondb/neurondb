-- ============================================================================
-- Test 001: Document Ingestion and Chunking
-- ============================================================================
-- Demonstrates: Document storage, text chunking, metadata management
-- ============================================================================

\echo 'Creating document store with metadata...'

-- Create documents table
CREATE TABLE IF NOT EXISTS documents (
    doc_id SERIAL PRIMARY KEY,
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    source TEXT,
    doc_type TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB
);

-- Create chunks table for document segments
CREATE TABLE IF NOT EXISTS document_chunks (
    chunk_id SERIAL PRIMARY KEY,
    doc_id INTEGER REFERENCES documents(doc_id),
    chunk_index INTEGER,
    chunk_text TEXT NOT NULL,
    chunk_tokens INTEGER,
    embedding VECTOR(384),  -- Using 384-dim embeddings
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON document_chunks(doc_id);
-- Note: Vector indexes can be added after embeddings are generated
-- CREATE INDEX IF NOT EXISTS idx_chunks_embedding ON document_chunks USING hnsw (embedding) WITH (m = 16, ef_construction = 64);

\echo 'Ingesting sample documents...'

-- Insert sample technical documents
INSERT INTO documents (title, content, source, doc_type, metadata) VALUES
('PostgreSQL Performance Tuning', 
 'PostgreSQL performance can be significantly improved through proper indexing strategies. B-tree indexes are the default and work well for most queries. GiST indexes are useful for full-text search and geometric data. Hash indexes can be faster for equality comparisons but are not WAL-logged. Partial indexes can reduce index size and improve performance for queries with common WHERE clauses.',
 'https://wiki.postgresql.org/wiki/Performance_Optimization',
 'technical_doc',
 '{"category": "database", "tags": ["postgresql", "performance", "indexing"]}'::jsonb),

('Vector Databases Explained',
 'Vector databases store high-dimensional vector embeddings generated from machine learning models. These embeddings capture semantic meaning of text, images, or other data. Vector similarity search using cosine similarity or Euclidean distance enables semantic search capabilities. HNSW and IVFFlat are popular indexing algorithms that make approximate nearest neighbor search fast even with millions of vectors.',
 'https://example.com/vector-db-guide',
 'technical_doc',
 '{"category": "machine_learning", "tags": ["vectors", "embeddings", "similarity_search"]}'::jsonb),

('Retrieval-Augmented Generation Overview',
 'RAG combines the power of large language models with external knowledge retrieval. The process involves: 1) Converting user queries to embeddings, 2) Retrieving relevant documents using vector similarity, 3) Providing retrieved context to the LLM, 4) Generating accurate responses grounded in factual data. This approach reduces hallucinations and enables LLMs to access up-to-date information.',
 'https://example.com/rag-overview',
 'technical_doc',
 '{"category": "ai", "tags": ["rag", "llm", "retrieval"]}'::jsonb),

('Python Machine Learning Best Practices',
 'When building ML models in Python, always split your data into training, validation, and test sets. Use cross-validation to get robust performance estimates. Feature scaling with StandardScaler or MinMaxScaler often improves model performance. Handle missing data appropriately using imputation or deletion strategies. Use pipelines to ensure consistent preprocessing.',
 'https://example.com/python-ml',
 'technical_doc',
 '{"category": "machine_learning", "tags": ["python", "sklearn", "best_practices"]}'::jsonb),

('Database Sharding Strategies',
 'Sharding distributes data across multiple database instances to improve scalability. Common strategies include: Range-based sharding (e.g., by date), Hash-based sharding (distribute evenly), Directory-based sharding (lookup table), and Geographic sharding (by location). Each approach has trade-offs in terms of query complexity, data distribution, and rebalancing difficulty.',
 'https://example.com/sharding',
 'technical_doc',
 '{"category": "database", "tags": ["sharding", "scalability", "distributed"]}'::jsonb);

\echo 'Chunking documents into smaller segments...'

-- Simple chunking strategy: Split by sentences (approximately)
-- In production, use more sophisticated chunking (sliding window, semantic chunking, etc.)
INSERT INTO document_chunks (doc_id, chunk_index, chunk_text, chunk_tokens)
SELECT 
    doc_id,
    ROW_NUMBER() OVER (PARTITION BY doc_id ORDER BY chunk_num) - 1 AS chunk_index,
    chunk_text,
    array_length(regexp_split_to_array(chunk_text, '\s+'), 1) AS chunk_tokens
FROM (
    SELECT 
        doc_id,
        unnest(regexp_split_to_array(content, '\.\s+')) AS chunk_text,
        generate_series(1, array_length(regexp_split_to_array(content, '\.\s+'), 1)) AS chunk_num
    FROM documents
) chunks
WHERE length(chunk_text) > 20;  -- Filter out very short chunks

\echo 'Document ingestion complete!'
\echo ''

-- Show statistics
SELECT 
    COUNT(*) AS total_documents,
    SUM(length(content)) AS total_content_length,
    AVG(length(content)) AS avg_document_length
FROM documents;

SELECT 
    COUNT(*) AS total_chunks,
    AVG(chunk_tokens) AS avg_tokens_per_chunk,
    MIN(chunk_tokens) AS min_tokens,
    MAX(chunk_tokens) AS max_tokens
FROM document_chunks;

\echo ''
\echo 'Sample chunks:'
SELECT doc_id, chunk_index, left(chunk_text, 100) || '...' AS chunk_preview
FROM document_chunks
LIMIT 5;

