-- ============================================================================
-- NeurondB: FP8 Quantization (INT4, FP8)
-- ============================================================================
-- Implements INT4 (4-bit) and FP8 (8-bit floating point) quantization
-- with GPU acceleration support. FP8 formats: E4M3 and E5M2.
--
-- Copyright (c) 2024-2025, neurondb, Inc.
-- ============================================================================

-- ============================================================================
-- FP8 QUANTIZATION FUNCTIONS
-- ============================================================================

CREATE FUNCTION quantize_fp8_e4m3(vector)
RETURNS bytea
AS 'MODULE_PATHNAME', 'quantize_fp8_e4m3'
LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

COMMENT ON FUNCTION quantize_fp8_e4m3(vector) IS
	'Quantize vector to FP8 E4M3 format (4 exp, 3 mantissa bits)';

CREATE FUNCTION quantize_fp8_e5m2(vector)
RETURNS bytea
AS 'MODULE_PATHNAME', 'quantize_fp8_e5m2'
LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

COMMENT ON FUNCTION quantize_fp8_e5m2(vector) IS
	'Quantize vector to FP8 E5M2 format (5 exp, 2 mantissa bits)';

CREATE FUNCTION dequantize_fp8(bytea)
RETURNS vector
AS 'MODULE_PATHNAME', 'dequantize_fp8'
LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

COMMENT ON FUNCTION dequantize_fp8(bytea) IS
	'Dequantize FP8 vector back to float32';

-- ============================================================================
-- AUTO QUANTIZATION
-- ============================================================================

CREATE FUNCTION auto_quantize(vector, text)
RETURNS bytea
AS 'MODULE_PATHNAME', 'auto_quantize'
LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

COMMENT ON FUNCTION auto_quantize(vector, text) IS
	'Automatically select best quantization based on compression type (int4, fp8_e4m3, fp8_e5m2)';

-- ============================================================================
-- GRANT PERMISSIONS
-- ============================================================================

GRANT EXECUTE ON FUNCTION quantize_fp8_e4m3(vector) TO PUBLIC;
GRANT EXECUTE ON FUNCTION quantize_fp8_e5m2(vector) TO PUBLIC;
GRANT EXECUTE ON FUNCTION dequantize_fp8(bytea) TO PUBLIC;
GRANT EXECUTE ON FUNCTION auto_quantize(vector, text) TO PUBLIC;

