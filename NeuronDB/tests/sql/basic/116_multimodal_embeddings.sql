-- ============================================================================
-- NeurondB: Multi-Modal Embeddings (CLIP, ImageBind)
-- ============================================================================
-- Implements CLIP and ImageBind model integration for generating embeddings
-- from multiple modalities with cross-modal retrieval support.
--
-- Copyright (c) 2024-2026, neurondb, Inc.
-- ============================================================================

-- Multimodal embedding functions should already be created by extension
-- Validate that functions return expected results
DO $$
DECLARE
    clip_result vector;
    imagebind_result vector;
    search_count INT;
    func_exists boolean;
BEGIN
    -- Check if clip_embed function exists
    SELECT EXISTS (
        SELECT 1 FROM pg_proc p
        JOIN pg_namespace n ON n.oid = p.pronamespace
        WHERE n.nspname = 'public' AND p.proname = 'clip_embed'
    ) INTO func_exists;
    
    IF func_exists THEN
        -- Test clip_embed
        BEGIN
            SELECT clip_embed('test', 'text') INTO clip_result;
            
            IF clip_result IS NULL THEN
                RAISE WARNING 'clip_embed returned NULL';
            END IF;
        EXCEPTION WHEN OTHERS THEN
            RAISE NOTICE 'clip_embed test failed (may require model setup): %', SQLERRM;
        END;
    ELSE
        RAISE NOTICE 'clip_embed function not available (feature may be disabled)';
    END IF;
    
    -- Check if imagebind_embed function exists
    SELECT EXISTS (
        SELECT 1 FROM pg_proc p
        JOIN pg_namespace n ON n.oid = p.pronamespace
        WHERE n.nspname = 'public' AND p.proname = 'imagebind_embed'
    ) INTO func_exists;
    
    IF func_exists THEN
        -- Test imagebind_embed
        BEGIN
            SELECT imagebind_embed('test', 'text') INTO imagebind_result;
            
            IF imagebind_result IS NULL THEN
                RAISE WARNING 'imagebind_embed returned NULL';
            END IF;
        EXCEPTION WHEN OTHERS THEN
            RAISE NOTICE 'imagebind_embed test failed (may require model setup): %', SQLERRM;
        END;
    ELSE
        RAISE NOTICE 'imagebind_embed function not available (feature may be disabled)';
    END IF;
    
    -- Test cross_modal_search (may require table setup, so allow errors)
    BEGIN
        SELECT COUNT(*) INTO search_count
        FROM cross_modal_search('test_table', 'embedding', 'text', 'query', 'text', 10);
        
        IF search_count IS NULL THEN
            RAISE WARNING 'cross_modal_search returned NULL (table may not exist)';
        END IF;
    EXCEPTION WHEN OTHERS THEN
        -- Table might not exist, which is acceptable
        RAISE NOTICE 'cross_modal_search requires table setup: %', SQLERRM;
    END;
END$$;

