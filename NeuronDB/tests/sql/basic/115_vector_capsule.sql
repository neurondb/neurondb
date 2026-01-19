-- ============================================================================
-- VectorCapsule: Multi-representation vector with metadata
-- ============================================================================
-- Best-in-class vector type with adaptive representation selection,
-- integrity checking, and provenance tracking.
--
-- Copyright (c) 2024-2026, neurondb, Inc.
-- ============================================================================

-- GUC for enabling VectorCapsule features
ALTER SYSTEM SET neurondb.vector_capsule_enabled = false;

COMMENT ON VARIABLE neurondb.vector_capsule_enabled IS
	'Enable VectorCapsule features (multi-representation vectors with metadata)';

-- VectorCapsule type (internal for now, can be exposed later)
-- Note: Full type registration would go here when ready

-- Function: Convert standard vector to VectorCapsule
CREATE FUNCTION vector_capsule_from_vector(
	vector,
	bool DEFAULT false,	-- include_fp16
	bool DEFAULT false,	-- include_int8
	bool DEFAULT false,	-- include_binary
	bool DEFAULT false	-- cache_norm
)
RETURNS internal
AS 'MODULE_PATHNAME', 'vector_capsule_from_vector'
LANGUAGE C IMMUTABLE STRICT;

COMMENT ON FUNCTION vector_capsule_from_vector(vector, bool, bool, bool, bool) IS
	'Convert standard vector to VectorCapsule with optional quantized representations';

-- Function: Validate VectorCapsule integrity
CREATE FUNCTION vector_capsule_validate_integrity(internal)
RETURNS bool
AS 'MODULE_PATHNAME', 'vector_capsule_validate_integrity'
LANGUAGE C IMMUTABLE STRICT;

COMMENT ON FUNCTION vector_capsule_validate_integrity(internal) IS
	'Verify integrity checksum of VectorCapsule';

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



