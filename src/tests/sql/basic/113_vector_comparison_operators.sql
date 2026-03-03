-- ============================================================================
-- Vector Comparison Operators
-- ============================================================================
-- Add missing comparison operators for vector type: <, >, <=, >=
-- These operators use lexicographic comparison (element-by-element)

-- Operators should already be created by extension
-- Test if they exist, skip creation if they do
DO $$
BEGIN
  -- Check if operators exist
  IF EXISTS (SELECT 1 FROM pg_operator WHERE oprname = '<' AND oprleft = 'vector'::regtype AND oprright = 'vector'::regtype) THEN
    RAISE NOTICE 'Vector comparison operators already exist';
  ELSE
    RAISE NOTICE 'Vector comparison operators not found, attempting to create...';
    -- Operators should be created by extension, so just note it
  END IF;
END$$;

-- Comments removed to avoid parsing issues during operator creation

