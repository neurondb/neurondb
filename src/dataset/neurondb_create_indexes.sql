-- ============================================================================
-- NeurondB Index Creation Script
-- ============================================================================
-- Creates HNSW indexes on vector columns for fast similarity search.
--
-- Usage:
--   psql -d nurondb_dataset -f neurondb_create_indexes.sql
-- ============================================================================

\timing on
\set ON_ERROR_STOP on

\echo ''
\echo '╔════════════════════════════════════════════════════════════════╗'
\echo '║          NeurondB Index Creation                               ║'
\echo '╚════════════════════════════════════════════════════════════════╝'
\echo ''

-- ============================================================================
-- STEP 1: Create HNSW Index on Documents
-- ============================================================================
\echo '► Step 1: Creating HNSW index on documents.embedding...'

CREATE INDEX IF NOT EXISTS idx_documents_embedding_hnsw 
ON public.documents 
USING hnsw (embedding vector_l2_ops)
WITH (m = 16, ef_construction = 200);

\echo '✅ Documents HNSW index created'
\echo ''

-- ============================================================================
-- STEP 2: Create HNSW Index on Products
-- ============================================================================
\echo '► Step 2: Creating HNSW index on products.embedding...'

CREATE INDEX IF NOT EXISTS idx_products_embedding_hnsw 
ON public.products 
USING hnsw (embedding vector_l2_ops)
WITH (m = 16, ef_construction = 200);

\echo '✅ Products HNSW index created'
\echo ''

-- ============================================================================
-- STEP 3: Create HNSW Indexes on Q&A Pairs
-- ============================================================================
\echo '► Step 3: Creating HNSW indexes on qa_pairs...'

CREATE INDEX IF NOT EXISTS idx_qa_question_embedding_hnsw 
ON public.qa_pairs 
USING hnsw (question_embedding vector_l2_ops)
WITH (m = 16, ef_construction = 200);

CREATE INDEX IF NOT EXISTS idx_qa_answer_embedding_hnsw 
ON public.qa_pairs 
USING hnsw (answer_embedding vector_l2_ops)
WITH (m = 16, ef_construction = 200);

\echo '✅ Q&A HNSW indexes created'
\echo ''

-- ============================================================================
-- STEP 4: Create Full-Text Search Indexes
-- ============================================================================
\echo '► Step 4: Creating FTS indexes for hybrid search...'

CREATE INDEX IF NOT EXISTS idx_documents_content_fts 
ON public.documents 
USING gin(to_tsvector('english', content));

CREATE INDEX IF NOT EXISTS idx_products_description_fts 
ON public.products 
USING gin(to_tsvector('english', name || ' ' || COALESCE(description, '')));

\echo '✅ FTS indexes created'
\echo ''

-- ============================================================================
-- STEP 5: Show Index Information
-- ============================================================================
\echo '► Step 5: Index information...'

SELECT 
    schemaname,
    tablename,
    indexname,
    pg_size_pretty(pg_relation_size(indexname::regclass)) as index_size
FROM pg_indexes
WHERE schemaname = 'public'
  AND tablename IN ('documents', 'products', 'qa_pairs')
ORDER BY tablename, indexname;

\echo ''
\echo '✅ Index creation complete!'
\echo '✅ Ready for Step 4: Test features (04_test_features.sql)'
\echo ''

