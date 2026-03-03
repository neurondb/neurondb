\timing on
\pset footer off
\pset pager off

\set ON_ERROR_STOP on

\echo '=========================================================================='
\echo 'Test: FP8 Quantization Basic Tests'
\echo '=========================================================================='

-- Test 1: FP8 E4M3 quantization
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Test 1: FP8 E4M3 quantization'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT
	'FP8 E4M3 quantization' AS test_name,
	quantize_fp8_e4m3('[1.0,2.0,3.0,4.0,5.0]'::vector) IS NOT NULL AS created;

-- Test 2: FP8 E5M2 quantization
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Test 2: FP8 E5M2 quantization'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT
	'FP8 E5M2 quantization' AS test_name,
	quantize_fp8_e5m2('[1.0,2.0,3.0,4.0,5.0]'::vector) IS NOT NULL AS created;

-- Test 3: Dequantize FP8
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Test 3: Dequantize FP8'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

WITH quantized AS (
	SELECT quantize_fp8_e4m3('[1.0,2.0,3.0]'::vector) AS q
)
SELECT
	'Dequantize FP8' AS test_name,
	dequantize_fp8(q) IS NOT NULL AS dequantized
FROM quantized;

-- Test 4: Auto quantization
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Test 4: Auto quantization'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT
	'Auto quantization' AS test_name,
	auto_quantize('[1.0,2.0,3.0]'::vector, 'fp8_e4m3') IS NOT NULL AS created;

\echo ''
\echo '✅ Basic FP8 quantization tests completed'

\echo 'Test completed successfully'
