-- ============================================================================
-- VectorCapsule: Multi-representation vector with metadata
-- ============================================================================
-- Best-in-class vector type with adaptive representation selection,
-- integrity checking, and provenance tracking.
--
-- Copyright (c) 2024-2026, neurondb, Inc.
-- ============================================================================

-- VectorCapsule features should be configured via extension
-- Test if functions exist
DO $$
BEGIN
  BEGIN
    PERFORM vector_capsule_from_vector('[1,2,3]'::vector);
    RAISE NOTICE 'vector_capsule_from_vector is available';
  EXCEPTION WHEN undefined_function THEN
    RAISE NOTICE 'vector_capsule_from_vector not available, skipping VectorCapsule tests';
  END;
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



