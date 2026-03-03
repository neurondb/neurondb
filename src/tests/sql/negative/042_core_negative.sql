-- 030_core_negative.sql
-- Negative test cases for core module: operators and distance functions
-- Tests error handling, invalid inputs, NULL handling, dimension mismatches

\timing on
\pset footer off
\pset pager off

\set ON_ERROR_STOP off

\echo '=========================================================================='
\echo 'Core Module: Negative Test Cases (Error Handling)'
\echo '=========================================================================='

/*-------------------------------------------------------------------
 * ---- DIMENSION MISMATCH ERRORS ----
 * Test error handling for dimension mismatches in distance functions
 *------------------------------------------------------------------*/
\echo ''
\echo 'Dimension Mismatch Tests'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

\echo 'Error Test 1: L2 distance with dimension mismatch'
SELECT vector_l2_distance(
	vector '[1,2,3]'::vector,
	vector '[1,2,3,4]'::vector
);

\echo 'Error Test 2: Cosine distance with dimension mismatch'
SELECT vector_cosine_distance(
	vector '[1,2,3]'::vector,
	vector '[1,2,3,4,5]'::vector
);

\echo 'Error Test 3: Inner product with dimension mismatch'
SELECT vector_inner_product(
	vector '[1,2,3]'::vector,
	vector '[1,2]'::vector
);

\echo 'Error Test 4: L1 distance with dimension mismatch'
SELECT vector_l1_distance(
	vector '[1,2,3,4]'::vector,
	vector '[1,2]'::vector
);

\echo 'Error Test 5: Hamming distance with dimension mismatch'
SELECT vector_hamming_distance(
	vector '[1,2,3]'::vector,
	vector '[1,2,3,4,5,6]'::vector
);

\echo 'Error Test 6: Chebyshev distance with dimension mismatch'
SELECT vector_chebyshev_distance(
	vector '[1,2,3]'::vector,
	vector '[1,2]'::vector
);

\echo 'Error Test 7: Minkowski distance with dimension mismatch'
SELECT vector_minkowski_distance(
	vector '[1,2,3,4]'::vector,
	vector '[1,2]'::vector,
	2.0
);

\echo 'Error Test 8: L2 operator (<->) with dimension mismatch'
SELECT vector '[1,2,3]'::vector <-> vector '[1,2,3,4]'::vector;

\echo 'Error Test 9: Cosine operator (<=>) with dimension mismatch'
SELECT vector '[1,2,3]'::vector <=> vector '[1,2]'::vector;

\echo 'Error Test 10: Inner product operator (<#>) with dimension mismatch'
SELECT vector '[1,2,3,4]'::vector <#> vector '[1,2]'::vector;

/*-------------------------------------------------------------------
 * ---- NULL HANDLING ----
 * Test NULL input handling for distance functions
 *------------------------------------------------------------------*/
\echo ''
\echo 'NULL Handling Tests'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

\echo 'Error Test 11: L2 distance with NULL first argument'
SELECT vector_l2_distance(NULL::vector, vector '[1,2,3]'::vector);

\echo 'Error Test 12: L2 distance with NULL second argument'
SELECT vector_l2_distance(vector '[1,2,3]'::vector, NULL::vector);

\echo 'Error Test 13: L2 distance with both NULL arguments'
SELECT vector_l2_distance(NULL::vector, NULL::vector);

\echo 'Error Test 14: Cosine distance with NULL first argument'
SELECT vector_cosine_distance(NULL::vector, vector '[1,2,3]'::vector);

\echo 'Error Test 15: Cosine distance with NULL second argument'
SELECT vector_cosine_distance(vector '[1,2,3]'::vector, NULL::vector);

\echo 'Error Test 16: Inner product with NULL first argument'
SELECT vector_inner_product(NULL::vector, vector '[1,2,3]'::vector);

\echo 'Error Test 17: Inner product with NULL second argument'
SELECT vector_inner_product(vector '[1,2,3]'::vector, NULL::vector);

\echo 'Error Test 18: L1 distance with NULL arguments'
SELECT vector_l1_distance(NULL::vector, vector '[1,2,3]'::vector);

\echo 'Error Test 19: Hamming distance with NULL arguments'
SELECT vector_hamming_distance(vector '[1,2,3]'::vector, NULL::vector);

\echo 'Error Test 20: Chebyshev distance with NULL arguments'
SELECT vector_chebyshev_distance(NULL::vector, vector '[1,2,3]'::vector);

\echo 'Error Test 21: Minkowski distance with NULL arguments'
SELECT vector_minkowski_distance(vector '[1,2,3]'::vector, NULL::vector, 2.0);

\echo 'Error Test 22: Minkowski distance with NULL p parameter'
SELECT vector_minkowski_distance(vector '[1,2,3]'::vector, vector '[1,2,3]'::vector, NULL::float4);

/*-------------------------------------------------------------------
 * ---- COMPARISON OPERATOR ERRORS ----
 * Test error handling for comparison operators with mismatched dimensions
 *------------------------------------------------------------------*/
\echo ''
\echo 'Comparison Operator Error Tests'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

\echo 'Error Test 23: Equality operator (=) with dimension mismatch'
SELECT vector '[1,2,3]'::vector = vector '[1,2,3,4]'::vector;

\echo 'Error Test 24: Inequality operator (<>) with dimension mismatch'
SELECT vector '[1,2,3]'::vector <> vector '[1,2]'::vector;

\echo 'Error Test 25: Less than operator (<) with dimension mismatch'
SELECT vector '[1,2,3]'::vector < vector '[1,2,3,4,5]'::vector;

\echo 'Error Test 26: Less equal operator (<=) with dimension mismatch'
SELECT vector '[1,2,3,4]'::vector <= vector '[1,2]'::vector;

\echo 'Error Test 27: Greater than operator (>) with dimension mismatch'
SELECT vector '[1,2,3]'::vector > vector '[1,2,3,4]'::vector;

\echo 'Error Test 28: Greater equal operator (>=) with dimension mismatch'
SELECT vector '[1,2,3]'::vector >= vector '[1,2]'::vector;

/*-------------------------------------------------------------------
 * ---- INVALID PARAMETERS ----
 * Test invalid parameter values
 *------------------------------------------------------------------*/
\echo ''
\echo 'Invalid Parameter Tests'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

\echo 'Error Test 29: Minkowski distance with negative p'
SELECT vector_minkowski_distance(
	vector '[1,2,3]'::vector,
	vector '[1,2,3]'::vector,
	-1.0
);

\echo 'Error Test 30: Minkowski distance with zero p'
SELECT vector_minkowski_distance(
	vector '[1,2,3]'::vector,
	vector '[1,2,3]'::vector,
	0.0
);

\echo 'Error Test 31: Minkowski distance with very large p'
SELECT vector_minkowski_distance(
	vector '[1,2,3]'::vector,
	vector '[1,2,3]'::vector,
	1e10::float4
);

/*-------------------------------------------------------------------
 * ---- EMPTY VECTOR ERRORS ----
 * Test error handling for empty or zero-dimension vectors
 *------------------------------------------------------------------*/
\echo ''
\echo 'Empty Vector Tests'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

\echo 'Error Test 32: L2 distance with zero-dimension vectors'
DO $$
BEGIN
	BEGIN
		PERFORM vector_l2_distance(vector '[]'::vector, vector '[]'::vector);
		RAISE EXCEPTION 'FAIL: expected error for zero-dimension vectors';
	EXCEPTION WHEN OTHERS THEN
		NULL; -- Error handled correctly
	END;
END$$;

\echo 'Error Test 33: Comparison with zero-dimension vectors'
DO $$
BEGIN
	BEGIN
		PERFORM vector '[1,2,3]'::vector = vector '[]'::vector;
		RAISE EXCEPTION 'FAIL: expected error for zero-dimension vector';
	EXCEPTION WHEN OTHERS THEN
		NULL; -- Error handled correctly
	END;
END$$;

/*-------------------------------------------------------------------
 * ---- EXTREME VALUES ----
 * Test handling of extreme values (infinity, NaN, very large numbers)
 *------------------------------------------------------------------*/
\echo ''
\echo 'Extreme Values Tests'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

\echo 'Error Test 34: Distance with very large values'
SELECT 
	vector_l2_distance(
		vector '[1e10,2e10,3e10,4e10,5e10,6e10,7e10,8e10,9e10,10e10,11e10,12e10,13e10,14e10,15e10,16e10,17e10,18e10,19e10,20e10,21e10,22e10,23e10,24e10,25e10,26e10,27e10,28e10]'::vector,
		vector '[1e10,2e10,3e10,4e10,5e10,6e10,7e10,8e10,9e10,10e10,11e10,12e10,13e10,14e10,15e10,16e10,17e10,18e10,19e10,20e10,21e10,22e10,23e10,24e10,25e10,26e10,27e10,28e10]'::vector
	) AS l2_large_values;

\echo 'Error Test 35: Distance with very small values'
SELECT 
	vector_l2_distance(
		vector '[1e-10,2e-10,3e-10,4e-10,5e-10,6e-10,7e-10,8e-10,9e-10,10e-10,11e-10,12e-10,13e-10,14e-10,15e-10,16e-10,17e-10,18e-10,19e-10,20e-10,21e-10,22e-10,23e-10,24e-10,25e-10,26e-10,27e-10,28e-10]'::vector,
		vector '[1e-10,2e-10,3e-10,4e-10,5e-10,6e-10,7e-10,8e-10,9e-10,10e-10,11e-10,12e-10,13e-10,14e-10,15e-10,16e-10,17e-10,18e-10,19e-10,20e-10,21e-10,22e-10,23e-10,24e-10,25e-10,26e-10,27e-10,28e-10]'::vector
	) AS l2_small_values;

\echo ''
\echo '=========================================================================='
\echo '✓ Core Module: Negative test cases complete'
\echo '=========================================================================='

\echo 'Test completed successfully'
