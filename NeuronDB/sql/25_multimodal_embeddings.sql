-- ============================================================================
-- NeurondB: Multi-Modal Embeddings (CLIP, ImageBind)
-- ============================================================================
-- Implements CLIP and ImageBind model integration for generating embeddings
-- from multiple modalities with cross-modal retrieval support.
--
-- Copyright (c) 2024-2025, neurondb, Inc.
-- ============================================================================

CREATE FUNCTION clip_embed(input text, modality text DEFAULT 'text')
RETURNS vector
AS 'MODULE_PATHNAME', 'clip_embed'
LANGUAGE C STRICT;

COMMENT ON FUNCTION clip_embed(text, text) IS
	'Generate CLIP embedding from text or image';

CREATE FUNCTION imagebind_embed(input text, modality text)
RETURNS vector
AS 'MODULE_PATHNAME', 'imagebind_embed'
LANGUAGE C STRICT;

COMMENT ON FUNCTION imagebind_embed(text, text) IS
	'Generate ImageBind embedding from any modality (text, image, audio, video, depth, thermal)';

CREATE FUNCTION cross_modal_search(
	table_name text,
	embedding_col text,
	query_modality text,
	query_input text,
	target_modality text,
	k int4
)
RETURNS TABLE(doc_id int4, score float4)
AS 'MODULE_PATHNAME', 'cross_modal_search'
LANGUAGE C STRICT;

COMMENT ON FUNCTION cross_modal_search(text, text, text, text, text, int4) IS
	'Search across modalities (e.g., text query, image results)';

GRANT EXECUTE ON FUNCTION clip_embed(text, text) TO PUBLIC;
GRANT EXECUTE ON FUNCTION imagebind_embed(text, text) TO PUBLIC;
GRANT EXECUTE ON FUNCTION cross_modal_search(text, text, text, text, text, int4) TO PUBLIC;

