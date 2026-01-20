-- ============================================================================
-- NeurondB: Flash Attention 2 Reranking
-- ============================================================================
-- Implements memory-efficient attention mechanism for cross-encoder reranking.
-- Supports long context windows (8K+ tokens) with reduced memory footprint.
--
-- Copyright (c) 2024-2026, neurondb, Inc.
-- ============================================================================

-- ============================================================================
-- FLASH ATTENTION RERANKING
-- ============================================================================
-- Functions should already be created by extension, use CREATE OR REPLACE if needed

DO $$
BEGIN
  BEGIN
    CREATE OR REPLACE FUNCTION rerank_flash(
      query text,
      candidates text[],
      model text DEFAULT NULL,
      top_k int4 DEFAULT 10
    )
    RETURNS TABLE(idx int4, score float4)
    AS 'MODULE_PATHNAME', 'rerank_flash'
    LANGUAGE C STRICT;
  EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE 'rerank_flash may already exist or not be available: %', SQLERRM;
  END;
  
  BEGIN
    CREATE OR REPLACE FUNCTION rerank_long_context(
      query text,
      candidates text[],
      max_tokens int4 DEFAULT 8192,
      top_k int4 DEFAULT 10
    )
    RETURNS TABLE(idx int4, score float4)
    AS 'MODULE_PATHNAME', 'rerank_long_context'
    LANGUAGE C STRICT;
  EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE 'rerank_long_context may already exist or not be available: %', SQLERRM;
  END;
END$$;

