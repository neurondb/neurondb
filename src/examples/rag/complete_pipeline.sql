-- ============================================================================
-- Complete RAG Pipeline Example in SQL
-- ============================================================================
-- This example demonstrates:
-- 1. Document ingestion (chunking + embedding)
-- 2. RAG query execution
-- 3. Evaluation metrics
-- ============================================================================

-- Step 1: Create table for documents
CREATE TABLE IF NOT EXISTS documents (
    id SERIAL PRIMARY KEY,
    content TEXT,
    embedding vector(384),
    metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- Step 2: Document Ingestion
-- Ingest a document using rag_ingest_document function
DO $$
DECLARE
    document_text TEXT := 'Machine learning is a subset of artificial intelligence that focuses on 
        the development of algorithms and statistical models that enable computer systems to 
        improve their performance on a specific task through experience. Unlike traditional 
        programming, where explicit instructions are provided, machine learning systems learn 
        from data patterns and make predictions or decisions based on that learning.';
    chunk_record RECORD;
    chunk_count INTEGER := 0;
BEGIN
    RAISE NOTICE '=== Step 1: Document Ingestion ===';
    
    FOR chunk_record IN 
        SELECT * FROM neurondb.rag_ingest_document(
            document_text,
            'documents',
            'content',
            'embedding',
            'default',
            512,
            128,
            '{}'::jsonb
        )
    LOOP
        chunk_count := chunk_count + 1;
        RAISE NOTICE '  Ingested chunk %: %...', 
            chunk_record.chunk_id, 
            LEFT(chunk_record.chunk_text, 50);
    END LOOP;
    
    RAISE NOTICE 'Successfully ingested % chunks', chunk_count;
END $$;

-- Step 3: RAG Query
-- Execute a RAG query using rag_query function
DO $$
DECLARE
    query_text TEXT := 'What is machine learning?';
    result_record RECORD;
    context_count INTEGER := 0;
    contexts TEXT[] := ARRAY[]::TEXT[];
BEGIN
    RAISE NOTICE '';
    RAISE NOTICE '=== Step 2: RAG Query ===';
    
    FOR result_record IN 
        SELECT chunk_text, relevance_score 
        FROM neurondb.rag_query(
            query_text,
            'documents',
            'embedding',
            'content',
            'default',
            3
        )
    LOOP
        context_count := context_count + 1;
        contexts := contexts || result_record.chunk_text;
        RAISE NOTICE '  Retrieved context (relevance: %.3f): %...', 
            result_record.relevance_score,
            LEFT(result_record.chunk_text, 50);
    END LOOP;
    
    RAISE NOTICE 'Retrieved % context chunks', context_count;
END $$;

-- Step 4: RAG Query with Context (includes answer generation)
-- Use rag_query_with_context for complete pipeline with answer
SELECT 
    chunk_text,
    relevance_score,
    answer
FROM neurondb.rag_query_with_context(
    'What is machine learning?',
    'documents',
    'embedding',
    'content',
    'default',
    3,
    '{"system_prompt": "You are a helpful assistant"}'::jsonb
);

-- Step 5: RAG Evaluation
-- Evaluate RAG performance using rag_evaluate function
DO $$
DECLARE
    query_text TEXT := 'What is machine learning?';
    answer_text TEXT := 'Machine learning is a subset of AI that enables systems to learn from data.';
    context_chunks TEXT[] := ARRAY[
        'Machine learning is a subset of artificial intelligence',
        'It focuses on algorithms and statistical models',
        'Systems improve performance through experience'
    ];
    evaluation JSONB;
BEGIN
    RAISE NOTICE '';
    RAISE NOTICE '=== Step 3: RAG Evaluation ===';
    
    SELECT neurondb.rag_evaluate(query_text, answer_text, context_chunks, 'basic')
    INTO evaluation;
    
    RAISE NOTICE '  Relevancy: %.3f', (evaluation->>'relevancy')::FLOAT;
    RAISE NOTICE '  Semantic Similarity: %.3f', (evaluation->>'semantic_similarity')::FLOAT;
    
    IF evaluation->'similarity_stats' IS NOT NULL THEN
        RAISE NOTICE '  Average Similarity: %.3f', 
            (evaluation->'similarity_stats'->>'avg')::FLOAT;
    END IF;
    
    RAISE NOTICE '';
    RAISE NOTICE '=== RAG Pipeline Complete ===';
END $$;

-- Step 6: Conversational RAG
-- Use rag_chat for conversational interface with history
SELECT neurondb.rag_chat(
    'Tell me more about machine learning algorithms',
    'documents',
    'embedding',
    'content',
    'default',
    5,
    '[]'::jsonb,  -- Empty conversation history
    'gpt-3.5-turbo'
) AS chat_response;

-- Step 7: Pipeline Management
-- Create a RAG pipeline configuration
SELECT neurondb.create_rag_pipeline(
    'ml-docs-pipeline',
    'default',
    512,
    128,
    '{"rerank_enabled": true, "hybrid_enabled": false}'::jsonb
) AS pipeline_id;

-- List all pipelines
SELECT 
    pipeline_id,
    pipeline_name,
    embedding_model,
    chunk_size,
    chunk_overlap,
    created_at
FROM neurondb.rag_pipelines
ORDER BY created_at DESC;
