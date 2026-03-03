-- 036_types_negative.sql
-- Negative test cases for types module: error handling

\timing on
\pset footer off
\pset pager off

\set ON_ERROR_STOP off

\echo '=========================================================================='
\echo 'Types Module: Negative Test Cases (Error Handling)'
\echo '=========================================================================='

/*-------------------------------------------------------------------
 * ---- QUANTIZATION ERRORS ----
 *------------------------------------------------------------------*/
\echo ''
\echo 'Quantization Error Tests'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

\echo 'Error Test 1: INT8 quantization with NULL vector'
SELECT vector_quantize_int8(NULL::vector, vector '[1,2,3]'::vector, vector '[4,5,6]'::vector);

\echo 'Error Test 2: INT8 quantization with NULL min vector'
SELECT vector_quantize_int8(vector '[1,2,3]'::vector, NULL::vector, vector '[4,5,6]'::vector);

\echo 'Error Test 3: INT8 quantization with NULL max vector'
SELECT vector_quantize_int8(vector '[1,2,3]'::vector, vector '[1,2,3]'::vector, NULL::vector);

\echo 'Error Test 4: INT8 quantization with dimension mismatch'
SELECT vector_quantize_int8(
	vector '[1,2,3]'::vector,
	vector '[1,2]'::vector,
	vector '[4,5]'::vector
);

\echo 'Error Test 5: FP16 quantization with NULL vector'
SELECT vector_quantize_fp16(NULL::vector);

\echo 'Error Test 6: Binary quantization with NULL vector'
SELECT vector_quantize_binary(NULL::vector);

\echo 'Error Test 7: Dequantization with NULL quantized vector'
SELECT vector_dequantize_int8(NULL, vector '[1,2,3]'::vector, vector '[4,5,6]'::vector);

\echo ''
\echo '=========================================================================='
\echo '✓ Types Module: Negative test cases complete'
\echo '=========================================================================='

\echo 'Test completed successfully'




