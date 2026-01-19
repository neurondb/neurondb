-- 030_core_basic.sql
-- Basic test for core module: operators and distance functions
-- Tests core functionality with simple, straightforward cases

\timing on
\pset footer off
\pset pager off
\pset tuples_only off

\set ON_ERROR_STOP on

\echo '=========================================================================='
\echo 'Core Module: Basic Functionality Tests'
\echo '=========================================================================='

/*-------------------------------------------------------------------
 * ---- BASIC COMPARISON OPERATORS ----
 *------------------------------------------------------------------*/
\echo ''
\echo 'Basic Comparison Operators'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT 
	(vector '[1,2,3]'::vector = vector '[1,2,3]'::vector) AS equal_true,
	(vector '[1,2,3]'::vector = vector '[1,2,4]'::vector) AS equal_false,
	(vector '[1,2,3]'::vector <> vector '[1,2,4]'::vector) AS not_equal_true,
	(vector '[1,2,3]'::vector < vector '[1,2,4]'::vector) AS less_than,
	(vector '[1,2,4]'::vector > vector '[1,2,3]'::vector) AS greater_than,
	(vector '[1,2,3]'::vector <= vector '[1,2,3]'::vector) AS less_equal,
	(vector '[1,2,3]'::vector >= vector '[1,2,3]'::vector) AS greater_equal;

/*-------------------------------------------------------------------
 * ---- BASIC DISTANCE FUNCTIONS ----
 *------------------------------------------------------------------*/
\echo ''
\echo 'Basic Distance Functions'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

WITH test_vectors AS (
	SELECT 
		vector '[1,0,0]'::vector AS v1,
		vector '[0,1,0]'::vector AS v2,
		vector '[0,0,1]'::vector AS v3
)
SELECT 
	vector_l2_distance(v1, v2) AS l2_distance,
	vector_cosine_distance(v1, v2) AS cosine_distance,
	vector_inner_product(v1, v2) AS inner_product,
	vector_l1_distance(v1, v2) AS l1_distance,
	vector_hamming_distance(v1, v2) AS hamming_distance,
	vector_chebyshev_distance(v1, v2) AS chebyshev_distance,
	vector_minkowski_distance(v1, v2, 2.0) AS minkowski_p2
FROM test_vectors;

/*-------------------------------------------------------------------
 * ---- BASIC DISTANCE OPERATORS ----
 *------------------------------------------------------------------*/
\echo ''
\echo 'Basic Distance Operators'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT 
	(vector '[1,2,3]'::vector <-> vector '[4,5,6]'::vector) AS l2_operator,
	(vector '[1,2,3]'::vector <=> vector '[4,5,6]'::vector) AS cosine_operator,
	(vector '[1,2,3]'::vector <#> vector '[4,5,6]'::vector) AS inner_product_operator;

/*-------------------------------------------------------------------
 * ---- IDENTICAL VECTORS ----
 *------------------------------------------------------------------*/
\echo ''
\echo 'Identical Vectors'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT 
	vector_l2_distance(v, v) AS l2_identical,
	vector_cosine_distance(v, v) AS cosine_identical,
	vector_l1_distance(v, v) AS l1_identical
FROM (SELECT vector '[1,2,3,4,5]'::vector AS v) t;

/*-------------------------------------------------------------------
 * ---- ZERO VECTORS ----
 *------------------------------------------------------------------*/
\echo ''
\echo 'Zero Vectors'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT 
	vector_l2_distance(
		vector '[0,0,0]'::vector,
		vector '[1,1,1]'::vector
	) AS l2_zero_to_one,
	vector_cosine_distance(
		vector '[0,0,0]'::vector,
		vector '[0,0,0]'::vector
	) AS cosine_zero_to_zero;

\echo ''
\echo '=========================================================================='
\echo '✓ Core Module: Basic tests complete'
\echo '=========================================================================='

\echo 'Test completed successfully'




