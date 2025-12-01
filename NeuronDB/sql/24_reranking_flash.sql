-- ============================================================================
-- NeurondB: Flash Attention 2 Reranking
-- ============================================================================
-- Implements memory-efficient attention mechanism for cross-encoder reranking.
-- Supports long context windows (8K+ tokens) with reduced memory footprint.
--
-- Copyright (c) 2024-2025, neurondb, Inc.
-- ============================================================================

-- ============================================================================
-- FLASH ATTENTION RERANKING
-- ============================================================================

CREATE FUNCTION rerank_flash(
	query text,
	candidates text[],
	model text DEFAULT NULL,
	top_k int4 DEFAULT 10
)
RETURNS TABLE(idx int4, score float4)
AS 'MODULE_PATHNAME', 'rerank_flash'
LANGUAGE C STRICT;

COMMENT ON FUNCTION rerank_flash(text, text[], text, int4) IS
	'Rerank candidates using Flash Attention 2 for memory-efficient cross-encoder scoring';

CREATE FUNCTION rerank_long_context(
	query text,
	candidates text[],
	max_tokens int4 DEFAULT 8192,
	top_k int4 DEFAULT 10
)
RETURNS TABLE(idx int4, score float4)
AS 'MODULE_PATHNAME', 'rerank_long_context'
LANGUAGE C STRICT;

COMMENT ON FUNCTION rerank_long_context(text, text[], int4, int4) IS
	'Rerank with long context support (8K+ tokens) using Flash Attention';

-- ============================================================================
-- GRANT PERMISSIONS
-- ============================================================================

GRANT EXECUTE ON FUNCTION rerank_flash(text, text[], text, int4) TO PUBLIC;
GRANT EXECUTE ON FUNCTION rerank_long_context(text, text[], int4, int4) TO PUBLIC;

