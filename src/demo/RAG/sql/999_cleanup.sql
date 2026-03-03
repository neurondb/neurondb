-- ============================================================================
-- Test 999: Cleanup RAG Demo
-- ============================================================================
-- Cleans up all tables and data created during RAG demo
-- ============================================================================

\echo 'Cleaning up RAG demo tables...'

DROP TABLE IF EXISTS qa_pairs CASCADE;
DROP TABLE IF EXISTS rag_queries CASCADE;
DROP TABLE IF EXISTS document_chunks CASCADE;
DROP TABLE IF EXISTS documents CASCADE;

\echo ''
\echo 'RAG demo cleanup complete!'


