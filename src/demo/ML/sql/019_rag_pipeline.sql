-- ============================================================================
-- NeuronDB RAG Pipeline Demo
-- Text chunking, embeddings, ranking, and transformations
-- ============================================================================

\set ON_ERROR_STOP on
\set QUIET on

\echo ''
\echo '══════════════================================================================'
\echo '  Demo 19: RAG Pipeline (Retrieval-Augmented Generation)'
\echo '══════════════================================================================'
\echo ''

-- Test 1: neurondb.chunk() - Text chunking
\echo 'Test 1: neurondb.chunk() - Text chunking with overlap'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo ''

WITH long_document AS (
    SELECT 'This is a very long document that needs to be chunked into smaller pieces for processing. ' ||
           'Each chunk will have some overlap with the next chunk to maintain context. ' ||
           'This is important for Retrieval-Augmented Generation pipelines. ' ||
           'The chunks can then be embedded and stored in a vector database. ' ||
           'When a user asks a question, relevant chunks are retrieved and used for generation.' as doc
)
SELECT 
    unnest(neurondb.chunk(doc, 100, 20)) as chunk,
    generate_subscripts(neurondb.chunk(doc, 100, 20), 1) as chunk_number
FROM long_document
LIMIT 5;

\echo ''

-- Test 2: neurondb.embed() - Generate embeddings
\echo 'Test 2: neurondb.embed() - Generate embeddings with GPU support'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo ''

WITH text_samples AS (
    SELECT 'Machine learning in databases is powerful' as text, 1 as id
    UNION ALL
    SELECT 'PostgreSQL extensions enable ML capabilities' as text, 2 as id
    UNION ALL
    SELECT 'Vector search with HNSW indexes is fast' as text, 3 as id
)
SELECT 
    id,
    text,
    vector_dims(neurondb.embed('all-MiniLM-L6-v2', text, true)) as embedding_dims,
    substring(neurondb.embed('all-MiniLM-L6-v2', text, true)::text, 1, 50) || '...' as embedding_preview
FROM text_samples;

\echo ''

-- Test 3: neurondb.rank() - Document reranking
\echo 'Test 3: neurondb.rank() - Rerank documents by relevance'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo ''

WITH documents AS (
    SELECT ARRAY[
        'PostgreSQL is a powerful relational database',
        'Machine learning models can be trained in SQL',
        'Vector search enables semantic similarity',
        'RAG pipelines combine retrieval and generation',
        'NeuronDB extends PostgreSQL with ML capabilities'
    ] as docs
)
SELECT neurondb.rank('machine learning', docs, 'bm25') as ranked_results
FROM documents;

\echo ''

-- Test 4: neurondb.transform() - Data transformations
\echo 'Test 4: neurondb.transform() - Data transformations'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo ''

WITH raw_data AS (
    SELECT ARRAY[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]::float8[] as data
)
SELECT 
    'Original' as transformation,
    data
FROM raw_data

UNION ALL

SELECT 
    'Normalized (L2)' as transformation,
    neurondb.transform('normalize', data)
FROM raw_data

UNION ALL

SELECT 
    'Standardized (Z-score)' as transformation,
    neurondb.transform('standardize', data)
FROM raw_data

UNION ALL

SELECT 
    'Min-Max Scaled' as transformation,
    neurondb.transform('min_max', data)
FROM raw_data;

\echo ''

-- Test 5: Complete RAG workflow
\echo 'Test 5: Complete RAG workflow (chunk + embed + rank)'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo ''

DROP TABLE IF EXISTS rag_documents CASCADE;
CREATE TEMP TABLE rag_documents AS
WITH long_docs AS (
    SELECT 
        i as doc_id,
        'Document about ' || 
        CASE 
            WHEN i % 3 = 0 THEN 'machine learning and artificial intelligence applications'
            WHEN i % 3 = 1 THEN 'database systems and query optimization techniques'
            ELSE 'vector search and similarity matching algorithms'
        END ||
        '. This contains detailed information about the topic.' as content
    FROM generate_series(1, 10) i
)
SELECT 
    doc_id,
    content,
    unnest(neurondb.chunk(content, 50, 10)) as chunk,
    generate_subscripts(neurondb.chunk(content, 50, 10), 1) as chunk_num,
    neurondb.embed('all-MiniLM-L6-v2', unnest(neurondb.chunk(content, 50, 10))) as embedding
FROM long_docs;

SELECT 
    doc_id,
    chunk_num,
    substring(chunk, 1, 60) || '...' as chunk_preview,
    vector_dims(embedding) as embedding_dims
FROM rag_documents
LIMIT 10;

\echo ''
\echo '══════════════================================================================'
\echo '  ✅ RAG Pipeline Demo Complete'
\echo '══════════════================================================================'
\echo ''

