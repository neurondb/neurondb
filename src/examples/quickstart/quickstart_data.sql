-- ============================================================================
-- NeuronDB Quickstart Data Pack
-- ============================================================================
-- This script creates a sample dataset with ~500 documents and pre-built
-- vector embeddings for quick testing and learning.
--
-- Usage:
--   psql "postgresql://neurondb:neurondb@localhost:5433/neurondb" -f quickstart_data.sql
--
-- Or use the wrapper script:
--   ./load_quickstart.sh
-- ============================================================================

\set ON_ERROR_STOP on
\echo '=========================================================================='
\echo '       NeuronDB Quickstart Data Pack'
\echo '=========================================================================='
\echo ''

-- Create extension if not exists
\echo 'Creating NeuronDB extension...'
CREATE EXTENSION IF NOT EXISTS neurondb;
\echo '✓ Extension created'
\echo ''

-- Drop existing quickstart table if it exists (for idempotent reruns)
DROP TABLE IF EXISTS quickstart_documents CASCADE;

-- Create schema
\echo 'Creating quickstart_documents table...'
CREATE TABLE quickstart_documents (
    id SERIAL PRIMARY KEY,
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    embedding vector(384)
);
\echo '✓ Table created'
\echo ''

-- Generate sample data
-- Using generate_series to create ~500 documents with synthetic vectors
\echo 'Inserting sample documents (this may take a moment)...'

-- Create a temporary helper for text arrays
WITH title_list AS (
    SELECT ARRAY[
        'Introduction to Machine Learning',
        'Deep Learning Fundamentals',
        'Neural Networks Explained',
        'Natural Language Processing Basics',
        'Computer Vision Overview',
        'Reinforcement Learning Guide',
        'Transformers Architecture',
        'Convolutional Neural Networks',
        'Recurrent Neural Networks',
        'Attention Mechanisms',
        'Large Language Models',
        'Vector Databases',
        'Embedding Generation',
        'Similarity Search',
        'Semantic Search Techniques',
        'RAG Pipeline Design',
        'Vector Indexing Strategies',
        'Hybrid Search Methods',
        'PostgreSQL Extensions',
        'SQL for AI Applications'
    ] AS titles,
    ARRAY[
        'Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed.',
        'Deep learning uses neural networks with multiple layers to learn complex patterns in data, enabling breakthroughs in image recognition and natural language understanding.',
        'Neural networks are computing systems inspired by biological neural networks, consisting of interconnected nodes that process information.',
        'Natural language processing combines computational linguistics with machine learning to enable computers to understand and generate human language.',
        'Computer vision enables machines to interpret and understand visual information from the world, powering applications like autonomous vehicles and medical imaging.',
        'Reinforcement learning is a type of machine learning where agents learn to make decisions by interacting with an environment and receiving rewards or penalties.',
        'Transformers are a revolutionary neural network architecture that uses attention mechanisms to process sequences, forming the basis of modern LLMs.',
        'Convolutional neural networks are specialized for processing grid-like data such as images, using convolutional layers to detect features.',
        'Recurrent neural networks are designed to process sequences of data, maintaining memory of previous inputs through hidden states.',
        'Attention mechanisms allow neural networks to focus on relevant parts of input data, dramatically improving performance on complex tasks.',
        'Large language models are AI systems trained on vast amounts of text data, capable of understanding context and generating human-like text.',
        'Vector databases are specialized databases designed to store and query high-dimensional vectors efficiently, essential for AI applications.',
        'Embedding generation converts text, images, or other data into dense vector representations that capture semantic meaning.',
        'Similarity search finds items in a dataset that are most similar to a query vector, using distance metrics like cosine or Euclidean distance.',
        'Semantic search goes beyond keyword matching to understand the meaning and intent behind queries, providing more relevant results.',
        'RAG (Retrieval-Augmented Generation) combines information retrieval with language generation to provide accurate, context-aware responses.',
        'Vector indexing strategies like HNSW and IVF enable fast approximate nearest neighbor search even in high-dimensional spaces.',
        'Hybrid search combines vector similarity search with traditional keyword search to leverage the strengths of both approaches.',
        'PostgreSQL extensions add new functionality to the database, enabling advanced features like vector operations and machine learning.',
        'SQL for AI applications involves using database features to store vectors, perform similarity searches, and integrate with ML models.'
    ] AS contents
)
INSERT INTO quickstart_documents (title, content, embedding)
SELECT 
    tl.titles[((i - 1) % array_length(tl.titles, 1)) + 1] AS title,
    tl.contents[((i - 1) % array_length(tl.contents, 1)) + 1] AS content,
    array_agg(random()::real ORDER BY j)::real[]::vector(384) AS embedding
FROM generate_series(1, 500) i,
     generate_series(1, 384) j,
     title_list tl
GROUP BY i, tl.titles, tl.contents;

\echo '✓ 500 documents inserted'
\echo ''

-- Create HNSW index
\echo 'Creating HNSW index (this may take a moment)...'
CREATE INDEX quickstart_documents_embedding_idx
ON quickstart_documents
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

\echo '✓ Index created'
\echo ''

-- Display statistics
\echo '=========================================================================='
\echo 'Quickstart Data Pack Setup Complete!'
\echo '=========================================================================='
\echo ''
SELECT 
    COUNT(*) AS total_documents,
    COUNT(embedding) AS documents_with_embeddings,
    pg_size_pretty(pg_total_relation_size('quickstart_documents')) AS table_size,
    (SELECT indexname FROM pg_indexes WHERE tablename = 'quickstart_documents' LIMIT 1) AS index_name
FROM quickstart_documents;
\echo ''
\echo 'Sample queries you can try:'
\echo ''
\echo '1. View sample documents:'
\echo '   SELECT id, title, LEFT(content, 50) AS preview FROM quickstart_documents LIMIT 5;'
\echo ''
\echo '2. Basic similarity search:'
\echo '   SELECT id, title,'
\echo '          embedding <=> (SELECT embedding FROM quickstart_documents WHERE id = 1) AS distance'
\echo '   FROM quickstart_documents'
\echo '   WHERE id != 1'
\echo '   ORDER BY embedding <=> (SELECT embedding FROM quickstart_documents WHERE id = 1)'
\echo '   LIMIT 10;'
\echo ''
\echo '3. Find similar documents by title:'
\echo '   SELECT id, title'
\echo '   FROM quickstart_documents'
\echo '   WHERE title LIKE ''%Machine Learning%'''
\echo '   LIMIT 5;'
\echo ''
\echo 'For more examples, see: Docs/getting-started/recipes/'
\echo ''

