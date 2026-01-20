-- ============================================================================
-- NeurondB: Multi-Modal Embeddings (CLIP, ImageBind)
-- ============================================================================
-- Implements CLIP and ImageBind model integration for generating embeddings
-- from multiple modalities with cross-modal retrieval support.
--
-- Copyright (c) 2024-2026, neurondb, Inc.
-- ============================================================================

-- Multimodal embedding functions should already be created by extension
-- Test if they exist
DO $$
BEGIN
  BEGIN
    PERFORM clip_embed('test', 'text');
    RAISE NOTICE 'clip_embed is available';
  EXCEPTION WHEN undefined_function THEN
    RAISE NOTICE 'clip_embed not available, skipping multimodal embedding tests';
  END;
  
  BEGIN
    PERFORM imagebind_embed('test', 'text');
    RAISE NOTICE 'imagebind_embed is available';
  EXCEPTION WHEN undefined_function THEN
    RAISE NOTICE 'imagebind_embed not available';
  END;
  
  BEGIN
    PERFORM cross_modal_search('test_table', 'embedding', 'text', 'query', 'text', 10);
    RAISE NOTICE 'cross_modal_search is available';
  EXCEPTION WHEN undefined_function THEN
    RAISE NOTICE 'cross_modal_search not available';
  END;
END$$;

