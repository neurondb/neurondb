\timing on
\pset footer off
\pset pager off

\set ON_ERROR_STOP on

\echo '=========================================================================='
\echo '=========================================================================='

-- Test 1: Vector Creation and Basic Properties
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT 
	vector_dims('[1,2,3,4,5]'::vector) AS dims,
	vector_norm('[1,2,3,4,5]'::vector) AS norm,
	vector_normalize('[1,2,3,4,5]'::vector) AS normalized;

-- Test 2: Vector Arithmetic Operations
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT 
	vector_add('[1,2,3]'::vector, '[4,5,6]'::vector) AS addition,
	vector_sub('[4,5,6]'::vector, '[1,2,3]'::vector) AS subtraction,
	vector_mul('[1,2,3]'::vector, 2.0) AS scalar_multiplication;

-- Test 3: Vector Distance Functions
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

WITH vectors AS (
	SELECT '[1,2,3]'::vector AS v1, '[4,5,6]'::vector AS v2
)
SELECT 
	vector_l2_distance(v1, v2) AS l2_distance,
	vector_cosine_distance(v1, v2) AS cosine_distance,
	vector_inner_product(v1, v2) AS inner_product,
	vector_l1_distance(v1, v2) AS l1_distance,
	vector_minkowski_distance(v1, v2, 3.0) AS minkowski_distance
FROM vectors;

-- Test 4: Vector Concatenation
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT 
	vector_concat('[1,2,3]'::vector, '[4,5,6]'::vector) AS concatenated,
	vector_dims(vector_concat('[1,2,3]'::vector, '[4,5,6]'::vector)) AS concat_dims;

-- Test 5: Vector Conversion
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT 
	vector_to_binary('[1,2,3,4,5]'::vector) AS binary_representation,
	pg_column_size(vector_to_binary('[1,2,3,4,5]'::vector)) AS binary_size;

-- Test 6: Advanced Mathematical Operations - Exponential and Logarithmic
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT 
	vector_exp('[0,1,2]'::vector) AS exp_result,
	vector_log('[1,2.71828,7.389]'::vector) AS log_result,
	vector_log10('[1,10,100]'::vector) AS log10_result;

-- Test 7: Trigonometric Functions
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT 
	vector_sin('[0,1.5708,3.14159]'::vector) AS sin_result,
	vector_cos('[0,1.5708,3.14159]'::vector) AS cos_result,
	vector_tan('[0,0.7854,1.5708]'::vector) AS tan_result;

-- Test 8: Inverse Trigonometric Functions
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT 
	vector_asin('[0,0.5,1.0]'::vector) AS asin_result,
	vector_acos('[1.0,0.5,0.0]'::vector) AS acos_result,
	vector_atan('[0,1,2]'::vector) AS atan_result;

-- Test 9: Hyperbolic Functions
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT 
	vector_sinh('[0,1,2]'::vector) AS sinh_result,
	vector_cosh('[0,1,2]'::vector) AS cosh_result,
	vector_tanh('[0,1,2]'::vector) AS tanh_result;

-- Test 10: Error Functions
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT 
	vector_erf('[0,1,2]'::vector) AS erf_result,
	vector_erfc('[0,1,2]'::vector) AS erfc_result;

-- Test 11: Advanced Statistical Functions
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT 
	vector_skewness('[1,2,3,4,5,6,7,8,9,10]'::vector) AS skewness_result,
	vector_kurtosis('[1,2,3,4,5,6,7,8,9,10]'::vector) AS kurtosis_result,
	vector_entropy('[0.1,0.2,0.3,0.4]'::vector) AS entropy_result;

-- Test 12: Geometric Transformations
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT 
	vector_project('[3,4]'::vector, '[1,0]'::vector) AS project_result,
	vector_reject('[3,4]'::vector, '[1,0]'::vector) AS reject_result,
	vector_reflect('[1,1]'::vector, '[0,1]'::vector) AS reflect_result,
	vector_rotate('[1,0]'::vector, 1.5708) AS rotate_result;

-- Test 13: Correlation-Based Distance Metrics
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT 
	vector_pearson_correlation('[1,2,3,4,5]'::vector, '[2,4,6,8,10]'::vector) AS pearson_corr,
	vector_weighted_distance('[1,2,3]'::vector, '[4,5,6]'::vector, '[1,1,1]'::vector) AS weighted_dist;

-- Test 14: Information-Theoretic Distance Metrics
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT 
	vector_kl_divergence('[0.25,0.25,0.25,0.25]'::vector, '[0.5,0.25,0.15,0.1]'::vector) AS kl_div,
	vector_js_divergence('[0.25,0.25,0.25,0.25]'::vector, '[0.5,0.25,0.15,0.1]'::vector) AS js_div;

\echo ''
\echo '=========================================================================='

\echo 'Test completed successfully'
