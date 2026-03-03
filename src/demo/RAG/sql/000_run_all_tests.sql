-- ============================================================================
-- NeuronDB RAG (Retrieval-Augmented Generation) Demo - Run All Tests
-- ============================================================================
-- This script runs all RAG demonstration tests sequentially
-- Demonstrates: Document storage, Vector embeddings, Semantic search,
--               Hybrid search, RAG pipelines, Reranking, Q&A
-- ============================================================================

\echo '======================================================================'
\echo 'NeuronDB RAG Comprehensive Demo Suite'
\echo 'Testing Retrieval-Augmented Generation capabilities'
\echo '======================================================================'
\echo ''

-- Set display settings
\timing on
\x auto

-- Test 001: Setup and document ingestion
\echo 'Test 001: Document Ingestion and Chunking'
\i sql/001_document_ingestion.sql
\echo ''

-- Test 002: Generate embeddings
\echo 'Test 002: Vector Embeddings Generation'
\i sql/002_generate_embeddings.sql
\echo ''

-- Test 003: Semantic search
\echo 'Test 003: Semantic Search with Vector Similarity'
\i sql/003_semantic_search.sql
\echo ''

-- Test 004: Hybrid search
\echo 'Test 004: Hybrid Search (Vector + Full-Text)'
\i sql/004_hybrid_search.sql
\echo ''

-- Test 005: RAG pipeline
\echo 'Test 005: Complete RAG Pipeline'
\i sql/005_rag_pipeline.sql
\echo ''

-- Test 006: Reranking strategies
\echo 'Test 006: Document Reranking'
\i sql/006_reranking.sql
\echo ''

-- Test 007: Multi-query retrieval
\echo 'Test 007: Multi-Query Retrieval'
\i sql/007_multi_query.sql
\echo ''

-- Test 008: Contextual retrieval
\echo 'Test 008: Contextual Retrieval with Metadata'
\i sql/008_contextual_retrieval.sql
\echo ''

-- Test 009: Document summarization
\echo 'Test 009: Document Summarization'
\i sql/009_summarization.sql
\echo ''

-- Test 010: Question answering
\echo 'Test 010: Question Answering System'
\i sql/010_question_answering.sql
\echo ''

-- Test 011: Advanced RAG patterns
\echo 'Test 011: Advanced RAG Patterns'
\i sql/011_advanced_rag.sql
\echo ''

-- Test 012: Performance benchmarks
\echo 'Test 012: RAG Performance Benchmarks'
\i sql/012_performance.sql
\echo ''

-- Cleanup
\echo 'Test 999: Cleanup'
\i sql/999_cleanup.sql
\echo ''

\echo '======================================================================'
\echo 'NeuronDB RAG Demo Complete!'
\echo '======================================================================'

