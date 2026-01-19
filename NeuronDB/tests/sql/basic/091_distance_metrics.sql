-- Detailed test coverage for all distance metrics and edge cases

-- =============================
-- L2 (Euclidean) Distance
-- =============================

-- Basic and classical case
SELECT vector_l2_distance('[0.0, 0.0]'::vector, '[3.0, 4.0]'::vector) AS l2_3_4_is_5_expected;
SELECT vector_l2_distance('[1.5, -2.5]'::vector, '[-4.5, 7.5]'::vector) AS l2_negative_coords;
SELECT vector_l2_distance_gpu('[0.0, 0.0]'::vector, '[3.0, 4.0]'::vector) AS l2_3_4_gpu;

-- Edge: identical inputs = 0
SELECT vector_l2_distance('[1.0, 2.0]'::vector, '[1.0, 2.0]'::vector) AS l2_identical_should_be_0;

-- Edge: zero vectors
SELECT vector_l2_distance('[0.0, 0.0]'::vector, '[0.0, 0.0]'::vector) AS l2_zero_zero;

-- Edge: negative numbers
SELECT vector_l2_distance('[-1.0, -2.0]'::vector, '[-3.0, -4.0]'::vector) AS l2_negatives;

-- High-dimension
SELECT vector_l2_distance(ARRAY[1,2,3,4,5,6,7,8,9,10]::real[], ARRAY[10,9,8,7,6,5,4,3,2,1]::real[]) AS l2_high_dim;

-- Single-element vectors
SELECT vector_l2_distance('[42]'::vector, '[24]'::vector) AS l2_singleton;

-- =============================
-- L1 (Manhattan/Cityblock) Distance
-- =============================

SELECT vector_l1_distance('[1.0, 2.0]'::vector, '[4.0, 6.0]'::vector) AS l1_basic;
SELECT vector_cityblock_distance('[1.0, 2.0]'::vector, '[4.0, 6.0]'::vector) AS l1_cityblock_synonym;

-- Edge: identical inputs
SELECT vector_l1_distance('[7,8,9]'::vector, '[7,8,9]'::vector) AS l1_identical_is_0;

-- Edge: zeros
SELECT vector_l1_distance('[0,0,0]'::vector, '[0,0,0]'::vector) AS l1_all_zero;

-- Negative values
SELECT vector_l1_distance('[-1,2,-3]'::vector, '[1,-2,3]'::vector) AS l1_negatives;

-- Empty error: (should fail)
DO $$
BEGIN
  BEGIN
    PERFORM vector_l1_distance('[]'::vector, '[]'::vector);
    RAISE WARNING 'ERROR: l1 should not accept empty vectors';
  EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE 'Correctly rejected l1 empty: %', SQLERRM;
  END;
END$$;

-- =============================
-- Cosine Distance & Similarity
-- =============================

SELECT vector_cosine_distance('[1,0]'::vector, '[0,1]'::vector) AS cosine_orthogonal_1;
SELECT vector_cosine_distance_gpu('[1,0]'::vector, '[0,1]'::vector) AS cosine_gpu_orthogonal_1;
SELECT vector_cosine_sim('[1,0]'::vector, '[0,1]'::vector) AS cosine_sim_orthogonal_1;
SELECT vector_cosine_distance('[1,0]'::vector, '[1,0]'::vector) AS cosine_identical_0;
SELECT vector_cosine_distance('[1,0]'::vector, '[-1,0]'::vector) AS cosine_opposite_2;

-- Edge: zeros should error/warn
DO $$
BEGIN
  BEGIN
    PERFORM vector_cosine_distance('[0,0]'::vector, '[1.0, 0.0]'::vector);
    RAISE WARNING 'Cosine distance accepted zero vector!';
  EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE 'Correctly rejected cosine with zero: %', SQLERRM;
  END;
END$$;

-- Edge: nearly identical (floating point)
SELECT vector_cosine_distance('[1.00001,2.0]'::vector, '[1.00002,2.0]'::vector) AS cosine_near_equal;

-- =============================
-- Inner Product
-- =============================

SELECT vector_inner_product('[1,2,3]'::vector, '[4,5,6]'::vector) AS ip_basic;
SELECT vector_inner_product_gpu('[1,2,3]'::vector, '[4,5,6]'::vector) AS ip_gpu;

-- Negative values
SELECT vector_inner_product('[-1,2,-3]'::vector, '[3,-2,1]'::vector) AS ip_neg;

-- Edge: orthogonal (should be 0)
SELECT vector_inner_product('[1,0]'::vector, '[0,1]'::vector) AS ip_orthogonal_zero;

-- =============================
-- Hamming Distance (bitwise difference count)
-- =============================

SELECT vector_hamming_distance('[1,0,1]'::vector, '[1,1,0]'::vector) AS hamming_basic;
SELECT vector_hamming_distance('[0,1,0]'::vector, '[1,1,1]'::vector) AS hamming_two_diff;

-- Edge: identical
SELECT vector_hamming_distance('[1,1,1]'::vector, '[1,1,1]'::vector) AS hamming_identical_zero;

-- Edge: one vector all zero, other all one
SELECT vector_hamming_distance('[0,0,0,0]'::vector, '[1,1,1,1]'::vector) AS hamming_full_diff;

-- =============================
-- Chebyshev (L∞/Max) Distance
-- =============================

SELECT vector_chebyshev_distance('[1.0, 2.0]'::vector, '[4.0, 6.0]'::vector) AS chebyshev_basic;
SELECT vector_chebyshev_distance('[0.0, 100.0]'::vector, '[0.0, -100.0]'::vector) AS chebyshev_large_diff;

-- Edge: identical vectors = 0
SELECT vector_chebyshev_distance('[3,3,3]'::vector, '[3,3,3]'::vector) AS chebyshev_zero;

-- =============================
-- Minkowski Distance (p > 0, various p; generalizes L1, L2, L∞)
-- =============================

SELECT vector_minkowski_distance('[1,2]'::vector, '[4,6]'::vector, 1.0) AS minkowski_p1_l1;
SELECT vector_minkowski_distance('[1,2]'::vector, '[4,6]'::vector, 2.0) AS minkowski_p2_l2;
SELECT vector_minkowski_distance('[1,2]'::vector, '[4,6]'::vector, 3.0) AS minkowski_p3;
SELECT vector_minkowski_distance('[1,2]'::vector, '[4,6]'::vector, 1e6) AS minkowski_largep_chebyshev;

-- Edge: p = 0 (not allowed)
DO $$
BEGIN
  BEGIN
    PERFORM vector_minkowski_distance('[1,2]'::vector, '[4,6]'::vector, 0);
    RAISE WARNING 'Minkowski p=0 not rejected!';
  EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE 'Correctly rejected minkowski p=0: %', SQLERRM;
  END;
END$$;

-- =============================
-- Bray-Curtis Distance
-- =============================

SELECT vector_bray_curtis_distance('[1,2]'::vector, '[3,4]'::vector) AS bray_curtis_basic;
SELECT vector_bray_curtis_distance('[1,2,3,0]'::vector, '[0,2,3,1]'::vector) AS bray_curtis_complex;

-- Edge: all zeros (should error)
DO $$
BEGIN
  BEGIN
    PERFORM vector_bray_curtis_distance('[0,0]'::vector, '[0,0]'::vector);
    RAISE WARNING 'Bray-Curtis zero denominator not rejected!';
  EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE 'Correctly rejected bray-curtis all-zero: %', SQLERRM;
  END;
END$$;

-- =============================
-- Canberra Distance
-- =============================

SELECT vector_canberra_distance('[1,3]'::vector, '[2,0]'::vector) AS canberra_basic;
SELECT vector_canberra_distance('[5,0,-5]'::vector, '[0,0,5]'::vector) AS canberra_zero_in_coords;

-- Edge: all zeros (should error)
DO $$
BEGIN
  BEGIN
    PERFORM vector_canberra_distance('[0,0]'::vector, '[0,0]'::vector);
    RAISE WARNING 'Canberra all zero input NOT rejected!';
  EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE 'Correctly rejected canberra all-zero: %', SQLERRM;
  END;
END$$;

-- =============================
-- Binary/Bernoulli metrics
-- =============================

-- Jaccard
SELECT vector_jaccard_distance('[1,1,0]'::vector, '[1,0,1]'::vector) AS jaccard_basic;
SELECT vector_jaccard_distance('[0,0,0]'::vector, '[0,0,0]'::vector) AS jaccard_both_zero;

-- Sokal-Michener
SELECT vector_sokal_michener_distance('[1,1,0]'::vector, '[1,0,1]'::vector) AS sokal_michener_basic;

-- Rogers-Tanimoto
SELECT vector_rogers_tanimoto_distance('[1,1,0]'::vector, '[1,0,1]'::vector) AS rogers_tanimoto_basic;

-- Dice
SELECT vector_dice_distance('[1,1,0]'::vector, '[1,0,1]'::vector) AS dice_basic;

-- Russell-Rao
SELECT vector_russell_rao_distance('[1,1,0]'::vector, '[1,0,1]'::vector) AS russell_rao_basic;
SELECT vector_russell_rao_distance('[0,0,0]'::vector, '[0,0,0]'::vector) AS russell_rao_all_zero;

-- Matching coefficient
SELECT vector_matching_coefficient('[1,1,0]'::vector, '[1,0,1]'::vector) AS matching_coefficient_basic;

-- =============================
-- Additional Edge Cases & Sanity Checks
-- =============================

-- Dimension mismatch (should error)
DO $$
BEGIN
  BEGIN
    PERFORM vector_l2_distance('[1,2,3]'::vector, '[1,2]'::vector);
    RAISE WARNING 'ERROR: l2 did not reject dim mismatch!';
  EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE 'Correctly rejected l2 dim mismatch: %', SQLERRM;
  END;
END$$;

-- Negative/zero/float edge inputs on all metrics
SELECT vector_l1_distance('[-1,-2]'::vector, '[-3,-4]'::vector) AS l1_negatives;
SELECT vector_cosine_sim('[-1.0, 0.0]'::vector, '[0.0, -1.0]'::vector) AS cosine_negatives;

-- Short binary inputs, all combinations
SELECT vector_hamming_distance('[0,1,0]'::vector, '[1,1,1]'::vector) AS hamming_three;
SELECT vector_hamming_distance('[1,1,1]'::vector, '[1,1,1]'::vector) AS hamming_identical;
SELECT vector_hamming_distance('[0,0,0]'::vector, '[1,1,1]'::vector) AS hamming_all_diff;

-- Zeros in numerator/denominator for susceptible metrics
SELECT vector_bray_curtis_distance('[0,1]'::vector, '[1,0]'::vector) AS bray_curtis_edge;
SELECT vector_canberra_distance('[0,1]'::vector, '[1,0]'::vector) AS canberra_edge;

-- End of all detailed test coverage for distance metrics
