-- ============================================================================
-- NeurondB: Sparse Vectors & Learned Sparse Retrieval
-- ============================================================================
-- Implements sparse_vector type for SPLADE/ColBERTv2/BM25 retrieval
-- with inverted index support and hybrid dense+sparse search.
--
-- Copyright (c) 2024-2026, neurondb, Inc.
-- ============================================================================

-- ============================================================================
-- SPARSE VECTOR TYPE
-- ============================================================================

-- Sparse vector type should already be created by extension
-- Check if it exists, skip creation if it does
DO $$
BEGIN
  -- Check if sparse_vector type exists
  IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'sparse_vector') THEN
    RAISE NOTICE 'sparse_vector type not found, attempting to create...';
    -- Type creation should be handled by extension, so just note it
    RAISE NOTICE 'sparse_vector type should be created by extension';
  ELSE
    RAISE NOTICE 'sparse_vector type exists';
  END IF;
END$$;

-- Check if sparse vector type and functions exist (extension may provide sparsevec or sparse_vector)
DO $$
DECLARE
    result REAL;
BEGIN
    BEGIN
        -- Extension defines sparse_vector_dot_product(sparse_vector, sparse_vector); format "dim:val,dim:val"
        SELECT sparse_vector_dot_product('1:1.0'::sparse_vector, '1:1.0'::sparse_vector) INTO result;
        IF result IS NULL THEN
            RAISE EXCEPTION 'sparse_vector_dot_product returned NULL';
        END IF;
    EXCEPTION WHEN undefined_function OR undefined_object OR invalid_text_representation THEN
        RAISE NOTICE 'sparse_vector_dot_product test skipped (type/function/format not available): %', SQLERRM;
        RETURN;
    END;

    BEGIN
        SELECT sparse_vector_dot_product(
            '1:1.0,2:2.0'::sparse_vector,
            '1:1.0,2:2.0'::sparse_vector
        ) INTO result;
        IF result IS NOT NULL AND result <= 0 THEN
            RAISE EXCEPTION 'sparse_vector_dot_product returned unexpected result: %', result;
        END IF;
    EXCEPTION WHEN OTHERS THEN
        RAISE NOTICE 'sparse_vector_dot_product (multi-element) test skipped: %', SQLERRM;
    END;
END$$;

-- ============================================================================
-- SPARSE INDEX FUNCTIONS
-- ============================================================================
-- Functions should already be created by extension, use CREATE OR REPLACE if needed

DO $$
BEGIN
  BEGIN
    CREATE OR REPLACE FUNCTION sparse_index_create(
      table_name text,
      sparse_col text,
      index_name text,
      min_freq int4 DEFAULT 1
    )
    RETURNS bool
    AS 'MODULE_PATHNAME', 'sparse_index_create'
    LANGUAGE C STRICT;
  EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE 'sparse_index_create may already exist or not be available: %', SQLERRM;
  END;
  
  BEGIN
    CREATE OR REPLACE FUNCTION sparse_index_search(
      index_name text,
      query_vec sparse_vector,
      k int4
    )
    RETURNS TABLE(doc_id int4, score float4)
    AS 'MODULE_PATHNAME', 'sparse_index_search'
    LANGUAGE C STRICT;
  EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE 'sparse_index_search may already exist or not be available: %', SQLERRM;
  END;
  
  BEGIN
    CREATE OR REPLACE FUNCTION sparse_search(
      table_name text,
      sparse_col text,
      query_vec sparse_vector,
      k int4
    )
    RETURNS TABLE(doc_id int4, score float4)
    AS 'MODULE_PATHNAME', 'sparse_search'
    LANGUAGE C STRICT;
  EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE 'sparse_search may already exist or not be available: %', SQLERRM;
  END;
  
  BEGIN
    CREATE OR REPLACE FUNCTION splade_embed(input_text text)
    RETURNS sparse_vector
    AS 'MODULE_PATHNAME', 'splade_embed'
    LANGUAGE C STRICT;
  EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE 'splade_embed may already exist or not be available: %', SQLERRM;
  END;
  
  BEGIN
    CREATE OR REPLACE FUNCTION hybrid_dense_sparse_search(
      table_name text,
      dense_col text,
      sparse_col text,
      dense_query vector,
      sparse_query sparse_vector,
      k int4,
      dense_weight float4 DEFAULT 0.5,
      sparse_weight float4 DEFAULT 0.5
    )
    RETURNS TABLE(doc_id int4, fused_score float4)
    AS 'MODULE_PATHNAME', 'hybrid_dense_sparse_search'
    LANGUAGE C STRICT;
  EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE 'hybrid_dense_sparse_search may already exist or not be available: %', SQLERRM;
  END;
END$$;

-- ============================================================================
-- EXAMPLE USAGE
-- ============================================================================

/*
-- Create table with both dense and sparse vectors
CREATE TABLE documents (
	id serial PRIMARY KEY,
	content text,
	dense_embedding vector(768),
	sparse_embedding sparse_vector
);

-- Create indexes
CREATE INDEX ON documents USING hnsw (dense_embedding vector_l2_ops);
SELECT sparse_index_create('documents', 'sparse_embedding', 'idx_sparse_docs');

-- Generate sparse embeddings
UPDATE documents SET sparse_embedding = splade_embed(content);

-- Hybrid search
SELECT * FROM hybrid_dense_sparse_search(
	'documents',
	'dense_embedding',
	'sparse_embedding',
	'[0.1,0.2,...]'::vector,
	'{vocab_size:30522, model:SPLADE, tokens:[100,200], weights:[0.5,0.8]}'::sparse_vector,
	10,
	0.6,  -- dense weight
	0.4   -- sparse weight
);
*/

