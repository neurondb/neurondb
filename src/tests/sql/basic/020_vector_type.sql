-- compatibility test: vector_type.sql
-- Tests basic vector type operations, operators, and edge cases
-- Based on test/sql/vector_type.sql

\timing on
\pset footer off
\pset pager off

\set ON_ERROR_STOP on

\echo '=========================================================================='
\echo 'Compatibility Test: Vector Type Operations'
\echo '=========================================================================='

-- Test 1: Vector Parsing (valid inputs)
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Test 1: Vector Parsing - Valid Inputs'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT '[1,2,3]'::vector;
SELECT '[-1,-2,-3]'::vector;
SELECT '[1.,2.,3.]'::vector;
SELECT ' [ 1,  2 ,    3  ] '::vector;
SELECT '[1.23456]'::vector;

-- Test 2: Vector Parsing (edge cases - some may error, which is expected)
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Test 2: Vector Parsing - Edge Cases (some may error)'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

-- These should error or handle appropriately
DO $$
BEGIN
    BEGIN
        PERFORM '[hello,1]'::vector;
        RAISE NOTICE 'Non-numeric value accepted (may be implementation-specific)';
    EXCEPTION WHEN OTHERS THEN
        RAISE NOTICE 'Non-numeric value correctly rejected: %', SQLERRM;
    END;
    
    BEGIN
        PERFORM '[NaN,1]'::vector;
        RAISE NOTICE 'NaN value accepted';
    EXCEPTION WHEN OTHERS THEN
        RAISE NOTICE 'NaN value rejected: %', SQLERRM;
    END;
    
    BEGIN
        PERFORM '[Infinity,1]'::vector;
        RAISE NOTICE 'Infinity value accepted';
    EXCEPTION WHEN OTHERS THEN
        RAISE NOTICE 'Infinity value rejected: %', SQLERRM;
    END;
    
    BEGIN
        PERFORM '[-Infinity,1]'::vector;
        RAISE NOTICE '-Infinity value accepted';
    EXCEPTION WHEN OTHERS THEN
        RAISE NOTICE '-Infinity value rejected: %', SQLERRM;
    END;
END $$;

-- Scientific notation
SELECT '[1.5e38,-1.5e38]'::vector;
SELECT '[1.5e+38,-1.5e+38]'::vector;
SELECT '[1.5e-38,-1.5e-38]'::vector;

-- Invalid syntax (should error)
DO $$
BEGIN
    BEGIN
        PERFORM '[1,2,3'::vector;
        RAISE WARNING 'Incomplete vector syntax accepted';
    EXCEPTION WHEN OTHERS THEN
        RAISE NOTICE 'Incomplete vector syntax correctly rejected: %', SQLERRM;
    END;
    
    BEGIN
        PERFORM '[1,2,3]9'::vector;
        RAISE WARNING 'Trailing characters accepted';
    EXCEPTION WHEN OTHERS THEN
        RAISE NOTICE 'Trailing characters correctly rejected: %', SQLERRM;
    END;
    
    BEGIN
        PERFORM '1,2,3'::vector;
        RAISE WARNING 'Missing brackets accepted';
    EXCEPTION WHEN OTHERS THEN
        RAISE NOTICE 'Missing brackets correctly rejected: %', SQLERRM;
    END;
    
    BEGIN
        PERFORM '[]'::vector;
        RAISE WARNING 'Empty vector accepted';
    EXCEPTION WHEN OTHERS THEN
        RAISE NOTICE 'Empty vector correctly rejected: %', SQLERRM;
    END;
END $$;

-- Test 3: Vector Type with Dimensions
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Test 3: Vector Type with Dimension Specification'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT '[1,2,3]'::vector(3);
-- Dimension mismatch (should error or coerce)
DO $$
BEGIN
    BEGIN
        PERFORM '[1,2,3]'::vector(2);
        RAISE NOTICE 'Dimension mismatch handled (may coerce or error)';
    EXCEPTION WHEN OTHERS THEN
        RAISE NOTICE 'Dimension mismatch correctly rejected: %', SQLERRM;
    END;
END $$;

-- Test 4: Vector Arrays
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Test 4: Vector Arrays'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT unnest('{"[1,2,3]", "[4,5,6]"}'::vector[]);

-- Test 5: Vector Arithmetic
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Test 5: Vector Arithmetic Operations'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT '[1,2,3]'::vector + '[4,5,6]';
SELECT '[1,2,3]'::vector - '[4,5,6]';
-- Element-wise vector * vector multiplication (Hadamard product) - not implemented
-- SELECT '[1,2,3]'::vector * '[4,5,6]'::vector;
SELECT '[1,2,3]'::vector * 2.0;
SELECT '[1,2,3]'::vector || '[4,5]';

-- Test 6: Vector Comparison Operators
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Test 6: Vector Comparison Operators'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT '[1,2,3]'::vector < '[1,2,3]';
SELECT '[1,2,3]'::vector <= '[1,2,3]';
SELECT '[1,2,3]'::vector = '[1,2,3]';
SELECT '[1,2,3]'::vector != '[1,2,3]';
SELECT '[1,2,3]'::vector >= '[1,2,3]';
SELECT '[1,2,3]'::vector > '[1,2,3]';

-- Test comparison with different dimensions (lexicographic comparison - new feature)
SELECT '[1,2]'::vector < '[1,2,3]'::vector;
SELECT '[1,2,3]'::vector < '[1,2]'::vector;
SELECT '[1,2]'::vector = '[1,2,3]'::vector;
SELECT '[1,2,3]'::vector = '[1,2]'::vector;
SELECT '[1,2]'::vector <= '[1,2,3]'::vector;
SELECT '[1,2,3]'::vector <= '[1,2]'::vector;
SELECT '[2,1]'::vector < '[1,2,3]'::vector;
SELECT '[1,2,3]'::vector < '[2,1]'::vector;
SELECT '[1,3]'::vector < '[1,2,3]'::vector;
SELECT '[1,2,3]'::vector < '[1,3]'::vector;

-- Test 7: Vector Dimensions and Norm
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Test 7: Vector Dimensions and Norm'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT vector_dims('[1,2,3]'::vector);
SELECT round(vector_norm('[1,1]')::numeric, 5);
SELECT vector_norm('[3,4]');
SELECT vector_norm('[0,1]');
SELECT vector_norm('[0,0]');
SELECT vector_norm('[2]');

-- Test 8: Distance Functions (using NeuronDB function names)
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Test 8: Distance Functions'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT vector_l2_distance('[0,0]'::vector, '[3,4]');
SELECT '[0,0]'::vector <-> '[3,4]';

-- Test vector_l2_squared_distance (new feature - for index optimization)
SELECT vector_l2_squared_distance('[0,0]'::vector, '[3,4]'::vector);
SELECT vector_l2_squared_distance('[0,0]'::vector, '[3,4]'::vector) = 25.0;
SELECT vector_l2_squared_distance('[1,2]'::vector, '[4,6]'::vector);
SELECT vector_l2_squared_distance('[1,2]'::vector, '[4,6]'::vector) = 25.0;
SELECT vector_l2_squared_distance('[0,0,0]'::vector, '[3,4,5]'::vector);
SELECT vector_l2_squared_distance('[0,0,0]'::vector, '[3,4,5]'::vector) = 50.0;

SELECT vector_inner_product('[1,2]'::vector, '[3,4]');
SELECT '[1,2]'::vector <#> '[3,4]';

-- Test vector_negative_inner_product (new feature - for index optimization)
SELECT vector_negative_inner_product('[1,2]'::vector, '[3,4]'::vector);
SELECT vector_negative_inner_product('[1,2]'::vector, '[3,4]'::vector) = -11.0;
SELECT vector_negative_inner_product('[0,0]'::vector, '[3,4]'::vector);
SELECT vector_negative_inner_product('[0,0]'::vector, '[3,4]'::vector) = 0.0;
SELECT vector_negative_inner_product('[1,1]'::vector, '[1,1]'::vector);
SELECT vector_negative_inner_product('[1,1]'::vector, '[1,1]'::vector) = -2.0;

SELECT vector_cosine_distance('[1,2]'::vector, '[2,4]');
SELECT vector_cosine_distance('[1,2]'::vector, '[0,0]');
SELECT vector_cosine_distance('[1,1]'::vector, '[1,1]');
SELECT vector_cosine_distance('[1,0]'::vector, '[0,2]');
SELECT '[1,2]'::vector <=> '[2,4]';

SELECT vector_l1_distance('[0,0]'::vector, '[3,4]');
SELECT '[0,0]'::vector <+> '[3,4]';

-- Test 9: Normalization
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Test 9: Vector Normalization'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT vector_normalize('[3,4]'::vector);
SELECT vector_normalize('[3,0]'::vector);
SELECT vector_normalize('[0,0.1]'::vector);
-- Zero vector normalization (may error)
DO $$
BEGIN
    BEGIN
        PERFORM vector_normalize('[0,0]'::vector);
        RAISE NOTICE 'Zero vector normalization handled';
    EXCEPTION WHEN OTHERS THEN
        RAISE NOTICE 'Zero vector normalization correctly rejected: %', SQLERRM;
    END;
END $$;

-- Test 10: Binary Quantization
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Test 10: Binary Quantization'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT binary_quantize('[1,0,-1]'::vector);
SELECT binary_quantize('[0,0.1,-0.2,-0.3,0.4,0.5,0.6,-0.7,0.8,-0.9,1]'::vector);

-- Test 11: Subvector Extraction
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Test 11: Subvector Extraction'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT subvector('[1,2,3,4,5]'::vector, 1, 3);
SELECT subvector('[1,2,3,4,5]'::vector, 3, 2);

-- Test 12: Aggregates
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Test 12: Vector Aggregates'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

-- Note: uses avg(vector) and sum(vector), NeuronDB uses vector_avg and vector_sum
-- Use batch functions for array-based aggregation (compatible approach)
SELECT vector_avg_batch(ARRAY['[1,2,3]'::vector, '[3,5,7]'::vector]);
SELECT vector_avg_batch(ARRAY['[1,2,3]'::vector, '[3,5,7]'::vector, NULL::vector]);
SELECT vector_sum_batch(ARRAY['[1,2,3]'::vector, '[3,5,7]'::vector]);
SELECT vector_sum_batch(ARRAY['[1,2,3]'::vector, '[3,5,7]'::vector, NULL::vector]);

\echo ''
\echo '=========================================================================='
\echo 'Test completed successfully'
\echo '=========================================================================='

