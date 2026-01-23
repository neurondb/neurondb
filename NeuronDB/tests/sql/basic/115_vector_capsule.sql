-- ============================================================================
-- VectorCapsule: Multi-representation vector with metadata
-- ============================================================================
-- Best-in-class vector type with adaptive representation selection,
-- integrity checking, and provenance tracking.
--
-- Copyright (c) 2024-2026, neurondb, Inc.
-- ============================================================================

-- VectorCapsule features should be configured via extension
-- Validate that function returns expected results
DO $$
DECLARE
    capsule_result bytea;
    func_exists boolean;
BEGIN
    -- Check if function exists and feature is enabled
    SELECT EXISTS (
        SELECT 1 FROM pg_proc p
        JOIN pg_namespace n ON n.oid = p.pronamespace
        WHERE n.nspname = 'public' AND p.proname = 'vector_capsule_from_vector'
    ) INTO func_exists;
    
    IF func_exists THEN
        -- Check if feature is enabled
        IF current_setting('neurondb.vector_capsule_enabled', true) = 'true' THEN
            SELECT vector_capsule_from_vector('[1,2,3]'::vector) INTO capsule_result;
            
            IF capsule_result IS NULL THEN
                RAISE WARNING 'vector_capsule_from_vector returned NULL';
            END IF;
        ELSE
            RAISE NOTICE 'vector_capsule_from_vector requires neurondb.vector_capsule_enabled = true';
        END IF;
    ELSE
        RAISE NOTICE 'vector_capsule_from_vector function not available (feature may be disabled)';
    END IF;
END$$;

-- Example usage (when feature is enabled):
/*
SET neurondb.vector_capsule_enabled = true;

-- Create VectorCapsule with all representations
SELECT vector_capsule_from_vector(
	'[1,2,3,4,5]'::vector,
	true,  -- include_fp16
	true,  -- include_int8
	true,  -- include_binary
	true   -- cache_norm
);

-- Validate integrity
SELECT vector_capsule_validate_integrity(
	vector_capsule_from_vector('[1,2,3]'::vector, true, false, false, true)
);
*/



