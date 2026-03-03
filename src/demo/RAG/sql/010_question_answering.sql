-- ============================================================================
-- Test 010: Question Answering System
-- ============================================================================
-- Demonstrates: Q&A pipeline, extractive QA, answer ranking
-- ============================================================================

\echo 'Testing Question Answering System...'

-- Create QA pairs table
CREATE TABLE IF NOT EXISTS qa_pairs (
    qa_id SERIAL PRIMARY KEY,
    question TEXT NOT NULL,
    answer TEXT,
    source_chunks INT[],
    confidence FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

\echo ''
\echo 'Question 1: What are vector databases used for?'

WITH question_embedding AS (
    SELECT embed_text('What are vector databases used for?',
        'sentence-transformers/all-MiniLM-L6-v2'::text
    ) AS embedding
),
answer_candidates AS (
    SELECT 
        dc.chunk_id,
        d.title,
        dc.chunk_text,
        1 - (dc.embedding <=> qe.embedding) AS confidence
    FROM document_chunks dc
    JOIN documents d ON dc.doc_id = d.doc_id
    CROSS JOIN question_embedding qe
    ORDER BY dc.embedding <=> qe.embedding
    LIMIT 3
)
INSERT INTO qa_pairs (question, answer, source_chunks, confidence)
SELECT 
    'What are vector databases used for?',
    chunk_text,
    ARRAY[chunk_id],
    confidence
FROM answer_candidates
ORDER BY confidence DESC
LIMIT 1
RETURNING qa_id, question, answer, ROUND(confidence::numeric, 4) AS conf;

\echo ''
\echo 'Question 2: How does RAG reduce hallucinations?'

WITH question_embedding AS (
    SELECT embed_text('How does RAG reduce hallucinations?',
        'sentence-transformers/all-MiniLM-L6-v2'::text
    ) AS embedding
)
SELECT 
    dc.chunk_id,
    d.title AS source,
    dc.chunk_text AS answer,
    ROUND((1 - (dc.embedding <=> qe.embedding))::numeric, 4) AS confidence
FROM document_chunks dc
JOIN documents d ON dc.doc_id = d.doc_id
CROSS JOIN question_embedding qe
ORDER BY dc.embedding <=> qe.embedding
LIMIT 3;

\echo ''
\echo 'Question 3: What are common machine learning best practices?'

WITH question_embedding AS (
    SELECT embed_text('What are common machine learning best practices?',
        'sentence-transformers/all-MiniLM-L6-v2'::text
    ) AS embedding
)
SELECT 
    d.title AS source,
    dc.chunk_text AS answer,
    ROUND((1 - (dc.embedding <=> qe.embedding))::numeric, 4) AS confidence
FROM document_chunks dc
JOIN documents d ON dc.doc_id = d.doc_id
CROSS JOIN question_embedding qe
ORDER BY dc.embedding <=> qe.embedding
LIMIT 3;

\echo ''
\echo 'All QA pairs:'
SELECT * FROM qa_pairs ORDER BY created_at DESC;

\echo ''
\echo 'Question answering complete!'


