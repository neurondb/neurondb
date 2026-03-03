-- ============================================================================
-- Test 007: Advanced Vector Operations
-- ============================================================================
-- Demonstrates 22 advanced vector manipulation functions
-- ============================================================================

\echo '=========================================================================='
\echo '|    Advanced Vector Operations - Comprehensive Implementation          |'
\echo '=========================================================================='
\echo ''

-- Test 1: Element Access
\echo 'Test 1: Element Access - Get/Set individual elements'
SELECT 
    '[10,20,30,40,50]'::vector AS original,
    vector_get('[10,20,30,40,50]'::vector, 0) AS first_element,
    vector_get('[10,20,30,40,50]'::vector, 2) AS third_element,
    vector_get('[10,20,30,40,50]'::vector, 4) AS last_element;

SELECT 
    '[10,20,30,40,50]'::vector AS original,
    vector_set('[10,20,30,40,50]'::vector, 2, 999.0) AS set_index_2_to_999;

\echo ''
\echo 'Test 2: Vector Slicing'
SELECT 
    '[1,2,3,4,5,6,7,8,9,10]'::vector AS original,
    vector_slice('[1,2,3,4,5,6,7,8,9,10]'::vector, 0, 3) AS first_3_elements,
    vector_slice('[1,2,3,4,5,6,7,8,9,10]'::vector, 5, 10) AS last_5_elements,
    vector_slice('[1,2,3,4,5,6,7,8,9,10]'::vector, 2, 8) AS middle_elements;

\echo ''
\echo 'Test 3: Append/Prepend'
SELECT 
    '[1,2,3]'::vector AS original,
    vector_append('[1,2,3]'::vector, 4.0) AS appended,
    vector_prepend(0.0, '[1,2,3]'::vector) AS prepended;

\echo ''
\echo 'Test 4: Element-wise Operations'
SELECT 
    '[-3,-2,-1,0,1,2,3]'::vector AS original,
    vector_abs('[-3,-2,-1,0,1,2,3]'::vector) AS absolute_values;

SELECT 
    '[1,2,3,4,5]'::vector AS original,
    vector_square('[1,2,3,4,5]'::vector) AS squared,
    vector_sqrt('[1,4,9,16,25]'::vector) AS sqrt_should_be_original;

SELECT 
    '[2,3,4]'::vector AS original,
    vector_pow('[2,3,4]'::vector, 2.0) AS power_of_2,
    vector_pow('[2,3,4]'::vector, 0.5) AS square_root;

\echo ''
\echo 'Test 5: Hadamard Product & Division'
SELECT 
    '[1,2,3,4,5]'::vector AS vec1,
    '[2,2,2,2,2]'::vector AS vec2,
    vector_hadamard('[1,2,3,4,5]'::vector, '[2,2,2,2,2]'::vector) AS hadamard_product,
    vector_divide('[10,20,30,40,50]'::vector, '[2,2,2,2,2]'::vector) AS element_wise_div;

\echo ''
\echo 'Test 6: Statistical Functions'
SELECT 
    '[1,2,3,4,5,6,7,8,9,10]'::vector AS vector,
    vector_mean('[1,2,3,4,5,6,7,8,9,10]'::vector) AS mean_should_be_5_point_5,
    vector_variance('[1,2,3,4,5,6,7,8,9,10]'::vector) AS variance,
    vector_stddev('[1,2,3,4,5,6,7,8,9,10]'::vector) AS stddev,
    vector_min('[5,2,8,1,9]'::vector) AS min_should_be_1,
    vector_max('[5,2,8,1,9]'::vector) AS max_should_be_9,
    vector_sum('[1,2,3,4,5]'::vector) AS sum_should_be_15;

\echo ''
\echo 'Test 7: Vector Comparison'
SELECT 
    '[1,2,3]'::vector = '[1,2,3]'::vector AS should_be_true,
    '[1,2,3]'::vector = '[1,2,4]'::vector AS should_be_false,
    vector_eq('[1.0,2.0,3.0]'::vector, '[1.0,2.0,3.0]'::vector) AS eq_true,
    vector_ne('[1,2,3]'::vector, '[1,2,4]'::vector) AS ne_true;

\echo ''
\echo 'Test 8: Vector Preprocessing'
SELECT 
    '[-10,5,15,25,30]'::vector AS original,
    vector_clip('[-10,5,15,25,30]'::vector, 0.0, 20.0) AS clipped_0_to_20;

SELECT 
    '[100,200,300,400,500]'::vector AS original,
    vector_standardize('[100,200,300,400,500]'::vector) AS standardized,
    vector_mean(vector_standardize('[100,200,300,400,500]'::vector)) AS mean_should_be_near_0,
    vector_stddev(vector_standardize('[100,200,300,400,500]'::vector)) AS stddev_should_be_1;

SELECT 
    '[5,10,15,20,25]'::vector AS original,
    vector_minmax_normalize('[5,10,15,20,25]'::vector) AS normalized_0_to_1,
    vector_min(vector_minmax_normalize('[5,10,15,20,25]'::vector)) AS min_should_be_0,
    vector_max(vector_minmax_normalize('[5,10,15,20,25]'::vector)) AS max_should_be_1;

\echo ''
\echo 'Test 9: Real-world use case - Image feature normalization'
CREATE TEMP TABLE image_features (
    image_id INT,
    raw_features VECTOR(512),
    normalized_features VECTOR(512),
    standardized_features VECTOR(512)
);

INSERT INTO image_features (image_id, raw_features)
SELECT 
    i,
    '[' || string_agg((random() * 255)::int::text, ',') || ']'::vector
FROM generate_series(1, 10) i,
     LATERAL (SELECT string_agg((random() * 255)::int::text, ',') FROM generate_series(1, 512)) dims(val)
GROUP BY i;

UPDATE image_features
SET normalized_features = vector_normalize(raw_features),
    standardized_features = vector_standardize(raw_features);

SELECT 
    image_id,
    vector_mean(raw_features)::numeric(10,2) AS raw_mean,
    vector_mean(normalized_features)::numeric(10,6) AS norm_mean,
    vector_mean(standardized_features)::numeric(10,6) AS std_mean,
    vector_norm(normalized_features)::numeric(10,6) AS norm_magnitude
FROM image_features
LIMIT 5;

\echo ''
\echo '=========================================================================='
\echo 'Advanced Operations Test Complete!'
\echo ''
\echo 'NeuronDB provides 22 advanced vector manipulation functions:'
\echo '  ✅ Element access: vector_get(), vector_set()'
\echo '  ✅ Slicing: vector_slice()'
\echo '  ✅ Append/Prepend: vector_append(), vector_prepend()'
\echo '  ✅ Element-wise: abs(), square(), sqrt(), pow()'
\echo '  ✅ Hadamard product & division'
\echo '  ✅ Statistics: mean(), variance(), stddev(), min(), max(), sum()'
\echo '  ✅ Comparison: eq(), ne()'
\echo '  ✅ Preprocessing: clip(), standardize(), minmax_normalize()'
\echo ''
\echo 'NeuronDB provides: Type, 11 distances, 2 indexes, 22 advanced ops, GPU support'
\echo '=========================================================================='
\echo ''

