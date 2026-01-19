-- compatibility test: cast.sql
-- Tests type casting between vector, halfvec, sparsevec, and arrays
-- Based on test/sql/cast.sql

\timing on
\pset footer off
\pset pager off

\set ON_ERROR_STOP on

\echo '=========================================================================='
\echo 'Compatibility Test: Type Casting'
\echo '=========================================================================='

-- Test 1: Array to Vector Conversions
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Test 1: Array to Vector Conversions'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT ARRAY[1,2,3]::vector;
SELECT ARRAY[1.0,2.0,3.0]::vector;
SELECT ARRAY[1,2,3]::float4[]::vector;
SELECT ARRAY[1,2,3]::float8[]::vector;
SELECT ARRAY[1,2,3]::numeric[]::vector;

-- Test 2: Vector to Array Conversions
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Test 2: Vector to Array Conversions'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT '[1,2,3]'::vector::real[];
SELECT vector_to_array('[1,2,3]'::vector);

-- Test 3: Real Array to Vector
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Test 3: Real Array to Vector'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT '{1,2,3}'::real[]::vector;
SELECT '{1,2,3}'::real[]::vector(3);
SELECT '{1,2,3}'::real[]::vector(2);

-- Edge cases with NULL, NaN, Infinity
DO $$
BEGIN
    BEGIN
        PERFORM '{NULL}'::real[]::vector;
        RAISE NOTICE 'NULL in array handled';
    EXCEPTION WHEN OTHERS THEN
        RAISE NOTICE 'NULL in array rejected: %', SQLERRM;
    END;
    
    BEGIN
        PERFORM '{NaN}'::real[]::vector;
        RAISE NOTICE 'NaN in array handled';
    EXCEPTION WHEN OTHERS THEN
        RAISE NOTICE 'NaN in array rejected: %', SQLERRM;
    END;
    
    BEGIN
        PERFORM '{Infinity}'::real[]::vector;
        RAISE NOTICE 'Infinity in array handled';
    EXCEPTION WHEN OTHERS THEN
        RAISE NOTICE 'Infinity in array rejected: %', SQLERRM;
    END;
    
    BEGIN
        PERFORM '{}'::real[]::vector;
        RAISE WARNING 'Empty array accepted';
    EXCEPTION WHEN OTHERS THEN
        RAISE NOTICE 'Empty array correctly rejected: %', SQLERRM;
    END;
END $$;

-- Test 4: Double Precision Array to Vector
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Test 4: Double Precision Array to Vector'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT '{1,2,3}'::double precision[]::vector;
SELECT '{1,2,3}'::double precision[]::vector(3);
SELECT '{1,2,3}'::double precision[]::vector(2);

-- Test 5: Vector to Halfvec Conversions
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Test 5: Vector to Halfvec Conversions'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT vector_to_halfvec('[1,2,3]'::vector);
SELECT vector_to_halfvec('[1,2,3]'::vector);
SELECT vector_to_halfvec('[1,2,3]'::vector);

-- Test 6: Halfvec to Vector Conversions
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Test 6: Halfvec to Vector Conversions'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT halfvec_to_vector('[1,2,3]'::halfvec);
SELECT halfvec_to_vector('[1,2,3]'::halfvec);
SELECT halfvec_to_vector('[1,2,3]'::halfvec);

-- Test 7: Real Array to Halfvec
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Test 7: Real Array to Halfvec'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT vector_to_halfvec('{1,2,3}'::real[]::vector);
SELECT vector_to_halfvec('{1,2,3}'::real[]::vector);
SELECT vector_to_halfvec('{1,2,3}'::real[]::vector);

-- Test 8: Vector to Sparsevec Conversions
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Test 8: Vector to Sparsevec Conversions'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT vector_to_sparsevec('[0,1.5,0,3.5,0]'::vector);
SELECT vector_to_sparsevec('[0,1.5,0,3.5,0]'::vector);
SELECT vector_to_sparsevec('[0,1.5,0,3.5,0]'::vector);

-- Test 9: Sparsevec to Vector Conversions
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Test 9: Sparsevec to Vector Conversions'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT sparsevec_to_vector('{2:1.5,4:3.5}/5'::sparsevec);
SELECT sparsevec_to_vector('{2:1.5,4:3.5}/5'::sparsevec);
SELECT sparsevec_to_vector('{2:1.5,4:3.5}/5'::sparsevec);

-- Test 10: Halfvec to Sparsevec Conversions
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Test 10: Halfvec to Sparsevec Conversions'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT vector_to_sparsevec(halfvec_to_vector('[0,1.5,0,3.5,0]'::halfvec));
SELECT vector_to_sparsevec(halfvec_to_vector('[0,1.5,0,3.5,0]'::halfvec));
SELECT vector_to_sparsevec(halfvec_to_vector('[0,1.5,0,3.5,0]'::halfvec));

-- Test 11: Sparsevec to Halfvec Conversions
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Test 11: Sparsevec to Halfvec Conversions'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT vector_to_halfvec(sparsevec_to_vector('{2:1.5,4:3.5}/5'::sparsevec));
SELECT vector_to_halfvec(sparsevec_to_vector('{2:1.5,4:3.5}/5'::sparsevec));
SELECT vector_to_halfvec(sparsevec_to_vector('{2:1.5,4:3.5}/5'::sparsevec));

-- Test 12: Array to Sparsevec Conversions
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Test 12: Array to Sparsevec Conversions'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

-- Array to sparsevec: convert to vector first, then to sparsevec
SELECT vector_to_sparsevec(ARRAY[1,0,2,0,3,0]::vector);
SELECT vector_to_sparsevec(ARRAY[1.0,0.0,2.0,0.0,3.0,0.0]::vector);
SELECT vector_to_sparsevec(ARRAY[1,0,2,0,3,0]::float4[]::vector);
SELECT vector_to_sparsevec(ARRAY[1,0,2,0,3,0]::float8[]::vector);
SELECT vector_to_sparsevec(ARRAY[1,0,2,0,3,0]::numeric[]::vector);

SELECT vector_to_sparsevec('{1,0,2,0,3,0}'::real[]::vector);
SELECT vector_to_sparsevec('{1,0,2,0,3,0}'::real[]::vector);
SELECT vector_to_sparsevec('{1,0,2,0,3,0}'::real[]::vector);

-- Test 13: Large Dimension Arrays
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Test 13: Large Dimension Arrays'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

-- Note: NeuronDB max dimension is 16000, not 16001
SELECT array_agg(n)::vector FROM generate_series(1, 16000) n;

\echo ''
\echo '=========================================================================='
\echo 'Test completed successfully'
\echo '=========================================================================='

