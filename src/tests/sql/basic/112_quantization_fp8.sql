-- ============================================================================
-- NeurondB: FP8 Quantization (INT4, FP8)
-- ============================================================================
-- Implements INT4 (4-bit) and FP8 (8-bit floating point) quantization
-- with GPU acceleration support. FP8 formats: E4M3 and E5M2.
--
-- Copyright (c) 2024-2026, neurondb, Inc.
-- ============================================================================

-- ============================================================================
-- FP8 QUANTIZATION FUNCTIONS
-- ============================================================================
-- Functions should already be created by extension, use CREATE OR REPLACE if needed

DO $$
BEGIN
  BEGIN
    CREATE OR REPLACE FUNCTION quantize_fp8_e4m3(vector)
    RETURNS bytea
    AS 'MODULE_PATHNAME', 'quantize_fp8_e4m3'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;
  EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE 'quantize_fp8_e4m3 may already exist or not be available: %', SQLERRM;
  END;
  
  BEGIN
    CREATE OR REPLACE FUNCTION quantize_fp8_e5m2(vector)
    RETURNS bytea
    AS 'MODULE_PATHNAME', 'quantize_fp8_e5m2'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;
  EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE 'quantize_fp8_e5m2 may already exist or not be available: %', SQLERRM;
  END;
  
  BEGIN
    CREATE OR REPLACE FUNCTION dequantize_fp8(bytea)
    RETURNS vector
    AS 'MODULE_PATHNAME', 'dequantize_fp8'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;
  EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE 'dequantize_fp8 may already exist or not be available: %', SQLERRM;
  END;
  
  BEGIN
    CREATE OR REPLACE FUNCTION auto_quantize(vector, text)
    RETURNS bytea
    AS 'MODULE_PATHNAME', 'auto_quantize'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;
  EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE 'auto_quantize may already exist or not be available: %', SQLERRM;
  END;
END$$;

-- ============================================================================
-- GRANT PERMISSIONS
-- ============================================================================

GRANT EXECUTE ON FUNCTION quantize_fp8_e4m3(vector) TO PUBLIC;
GRANT EXECUTE ON FUNCTION quantize_fp8_e5m2(vector) TO PUBLIC;
GRANT EXECUTE ON FUNCTION dequantize_fp8(bytea) TO PUBLIC;
GRANT EXECUTE ON FUNCTION auto_quantize(vector, text) TO PUBLIC;

