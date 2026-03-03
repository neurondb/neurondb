-- ============================================================================
-- Test 002: Vector Distance Metrics
-- ============================================================================
-- Demonstrates: L2, Cosine, Inner Product, L1, Hamming, Chebyshev, Minkowski
-- ============================================================================

\echo '=========================================================================='
\echo '|              Vector Distance Metrics - NeuronDB                       |'
\echo '=========================================================================='
\echo ''

-- Test 1: L2 (Euclidean) Distance
\echo 'Test 1: L2 (Euclidean) Distance'
SELECT 
    '[0,0]'::vector AS vec1,
    '[3,4]'::vector AS vec2,
    vector_l2_distance('[0,0]'::vector, '[3,4]'::vector) AS l2_dist_should_be_5,
    '[1,2,3]'::vector AS vec3,
    '[4,5,6]'::vector AS vec4,
    vector_l2_distance('[1,2,3]'::vector, '[4,5,6]'::vector) AS l2_dist;

\echo ''
\echo 'Test 2: Cosine Distance (1 - cosine similarity)'
\echo 'Range: [0, 2], where 0 = identical, 1 = orthogonal, 2 = opposite'
SELECT 
    vector_cosine_distance('[1,0]'::vector, '[1,0]'::vector) AS identical_should_be_0,
    vector_cosine_distance('[1,0]'::vector, '[0,1]'::vector) AS orthogonal_should_be_1,
    vector_cosine_distance('[1,0]'::vector, '[-1,0]'::vector) AS opposite_should_be_2,
    vector_cosine_distance('[1,2,3]'::vector, '[4,5,6]'::vector) AS sample_dist;

\echo ''
\echo 'Test 3: Inner Product Distance (negative inner product)'
SELECT 
    vector_inner_product('[1,2,3]'::vector, '[4,5,6]'::vector) AS inner_prod,
    '[1,2,3]'::vector <#> '[4,5,6]'::vector AS inner_prod_operator,
    1*4 + 2*5 + 3*6 AS manual_calculation;

\echo ''
\echo 'Test 4: L1 (Manhattan) Distance'
SELECT 
    '[0,0]'::vector AS vec1,
    '[3,4]'::vector AS vec2,
    vector_l1_distance('[0,0]'::vector, '[3,4]'::vector) AS l1_dist_should_be_7,
    '[1,2,3]'::vector AS vec3,
    '[4,5,6]'::vector AS vec4,
    vector_l1_distance('[1,2,3]'::vector, '[4,5,6]'::vector) AS l1_dist;

\echo ''
\echo 'Test 5: Hamming Distance (for binary vectors)'
SELECT 
    '[1,0,1,0]'::vector AS vec1,
    '[1,1,0,0]'::vector AS vec2,
    vector_hamming_distance('[1,0,1,0]'::vector, '[1,1,0,0]'::vector) AS hamming_dist;

\echo ''
\echo 'Test 6: Chebyshev Distance (max difference)'
SELECT 
    '[1,2,3]'::vector AS vec1,
    '[4,5,6]'::vector AS vec2,
    vector_chebyshev_distance('[1,2,3]'::vector, '[4,5,6]'::vector) AS chebyshev_should_be_3;

\echo ''
\echo 'Test 7: Minkowski Distance (generalized)'
SELECT 
    '[1,2,3]'::vector AS vec,
    '[4,5,6]'::vector AS vec2,
    vector_minkowski_distance('[1,2,3]'::vector, '[4,5,6]'::vector, 1.0) AS minkowski_p1_equals_l1,
    vector_minkowski_distance('[1,2,3]'::vector, '[4,5,6]'::vector, 2.0) AS minkowski_p2_equals_l2,
    vector_minkowski_distance('[1,2,3]'::vector, '[4,5,6]'::vector, 3.0) AS minkowski_p3;

\echo ''
\echo 'Test 8: Distance comparison table'
CREATE TEMP TABLE distance_comparison AS
SELECT 
    'L2 (Euclidean)' AS metric,
    vector_l2_distance('[1,2,3]'::vector, '[4,5,6]'::vector) AS distance,
    'Most common, geometric distance' AS description
UNION ALL
SELECT 
    'Cosine',
    vector_cosine_distance('[1,2,3]'::vector, '[4,5,6]'::vector),
    'Good for text similarity (direction matters)'
UNION ALL
SELECT 
    'Inner Product',
    vector_inner_product('[1,2,3]'::vector, '[4,5,6]'::vector),
    'Negative dot product, faster than cosine'
UNION ALL
SELECT 
    'L1 (Manhattan)',
    vector_l1_distance('[1,2,3]'::vector, '[4,5,6]'::vector),
    'Sum of absolute differences'
UNION ALL
SELECT 
    'Hamming',
    vector_hamming_distance('[1,0,1]'::vector, '[1,1,0]'::vector)::float,
    'Binary vectors, count of differences'
UNION ALL
SELECT 
    'Chebyshev',
    vector_chebyshev_distance('[1,2,3]'::vector, '[4,5,6]'::vector)::float,
    'Maximum difference across dimensions';

SELECT * FROM distance_comparison ORDER BY distance;

\echo ''
\echo 'Test 9: Similarity vs Distance (Cosine example)'
\echo 'Cosine similarity = 1 - cosine distance'
SELECT 
    '[1,2,3]'::vector AS vec1,
    '[4,5,6]'::vector AS vec2,
    vector_cosine_distance('[1,2,3]'::vector, '[4,5,6]'::vector) AS cosine_distance,
    1 - vector_cosine_distance('[1,2,3]'::vector, '[4,5,6]'::vector) AS cosine_similarity;

\echo ''
\echo '=========================================================================='
\echo 'Distance Metrics Test Complete!'
\echo '  ✅ L2 (Euclidean) - geometric distance'
\echo '  ✅ Cosine - angular distance (text similarity)'
\echo '  ✅ Inner Product - dot product'
\echo '  ✅ L1 (Manhattan) - city block distance'
\echo '  ✅ Hamming - binary vector differences'
\echo '  ✅ Chebyshev - maximum dimension difference'
\echo '  ✅ Minkowski - generalized metric (p-norm)'
\echo '=========================================================================='
\echo ''

