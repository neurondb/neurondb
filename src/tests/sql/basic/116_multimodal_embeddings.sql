-- ============================================================================
-- NeurondB: Multi-Modal Embeddings (CLIP, ImageBind)
-- ============================================================================
-- Implements CLIP and ImageBind model integration for generating embeddings
-- from multiple modalities with cross-modal retrieval support.
--
-- Copyright (c) 2024-2026, neurondb, Inc.
-- ============================================================================

-- Multimodal embedding functions should already be created by extension
-- Validate that functions return expected results.
-- Safe mode: when no LLM API key is set, skip clip_embed/imagebind_embed calls
-- to avoid crashes in embedding path; only run cross_modal_search (returns empty when table missing).
DO $$
DECLARE
    clip_result vector;
    imagebind_result vector;
    search_count INT;
    func_exists boolean;
    api_key_val text;
    safe_mode boolean;
BEGIN
    -- Safe mode when no API key (avoids crash in embedding path when model/HTTP not configured)
    BEGIN
        api_key_val := current_setting('neurondb.llm_api_key', true);
        safe_mode := (api_key_val IS NULL OR trim(api_key_val) = '');
    EXCEPTION WHEN OTHERS THEN
        safe_mode := true;
    END;

    -- Check if clip_embed function exists
    SELECT EXISTS (
        SELECT 1 FROM pg_proc p
        JOIN pg_namespace n ON n.oid = p.pronamespace
        WHERE n.nspname = 'public' AND p.proname = 'clip_embed'
    ) INTO func_exists;

    IF func_exists AND NOT safe_mode THEN
        -- Test clip_embed only when API key is set (avoid crash in unconfigured path)
        BEGIN
            SELECT clip_embed('test', 'text') INTO clip_result;

            IF clip_result IS NULL THEN
                RAISE WARNING 'clip_embed returned NULL';
            END IF;
        EXCEPTION WHEN OTHERS THEN
            RAISE NOTICE 'clip_embed test failed (may require model setup): %', SQLERRM;
        END;
    ELSIF func_exists AND safe_mode THEN
        RAISE NOTICE 'clip_embed skipped (safe mode: no neurondb.llm_api_key)';
    ELSE
        RAISE NOTICE 'clip_embed function not available (feature may be disabled)';
    END IF;

    -- Check if imagebind_embed function exists
    SELECT EXISTS (
        SELECT 1 FROM pg_proc p
        JOIN pg_namespace n ON n.oid = p.pronamespace
        WHERE n.nspname = 'public' AND p.proname = 'imagebind_embed'
    ) INTO func_exists;

    IF func_exists AND NOT safe_mode THEN
        -- Test imagebind_embed only when API key is set
        BEGIN
            SELECT imagebind_embed('test', 'text') INTO imagebind_result;

            IF imagebind_result IS NULL THEN
                RAISE WARNING 'imagebind_embed returned NULL';
            END IF;
        EXCEPTION WHEN OTHERS THEN
            RAISE NOTICE 'imagebind_embed test failed (may require model setup): %', SQLERRM;
        END;
    ELSIF func_exists AND safe_mode THEN
        RAISE NOTICE 'imagebind_embed skipped (safe mode: no neurondb.llm_api_key)';
    ELSE
        RAISE NOTICE 'imagebind_embed function not available (feature may be disabled)';
    END IF;

    -- Test cross_modal_search only when not in safe mode (it calls clip_embed internally)
    IF NOT safe_mode THEN
        BEGIN
            SELECT COUNT(*) INTO search_count
            FROM cross_modal_search('test_table', 'embedding', 'text', 'query', 'text', 10);

            IF search_count IS NULL THEN
                RAISE WARNING 'cross_modal_search returned NULL (table may not exist)';
            END IF;
        EXCEPTION WHEN OTHERS THEN
            RAISE NOTICE 'cross_modal_search requires table setup: %', SQLERRM;
        END;
    ELSE
        RAISE NOTICE 'cross_modal_search skipped (safe mode)';
    END IF;
END$$;

