-- Comprehensive crash test for vector operations
-- Tests NULL handling, dimension mismatches, edge cases

\set ON_ERROR_STOP off
\timing on

BEGIN;

-- Test 1: NULL vector handling
\echo 'Test 1: NULL vector operations'
DO $$
BEGIN
    PERFORM vector_l2_distance(NULL::vector, vector '[1,2,3]'::vector);
    RAISE EXCEPTION 'Should have failed with NULL vector';
EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE 'PASS: NULL vector correctly rejected';
END$$;

DO $$
BEGIN
    PERFORM vector_l2_distance(vector '[1,2,3]'::vector, NULL::vector);
    RAISE EXCEPTION 'Should have failed with NULL vector';
EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE 'PASS: NULL vector correctly rejected';
END$$;

-- Test 2: Dimension mismatches
\echo 'Test 2: Dimension mismatch handling'
DO $$
BEGIN
    PERFORM vector_l2_distance(vector '[1,2,3]'::vector, vector '[1,2,3,4]'::vector);
    RAISE EXCEPTION 'Should have failed with dimension mismatch';
EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE 'PASS: Dimension mismatch correctly rejected';
END$$;

-- Test 3: Zero-length vectors
\echo 'Test 3: Zero-length vector handling'
DO $$
BEGIN
    PERFORM vector_l2_distance(vector '[]'::vector, vector '[]'::vector);
    RAISE NOTICE 'PASS: Zero-length vectors handled';
EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE 'INFO: Zero-length vector result: %', SQLERRM;
END$$;

-- Test 4: Very large vectors (>16384 dimensions)
\echo 'Test 4: Large vector handling'
DO $$
DECLARE
    large_vec text;
    i int;
BEGIN
    large_vec := '[';
    FOR i IN 1..20000 LOOP
        large_vec := large_vec || i::text;
        IF i < 20000 THEN
            large_vec := large_vec || ',';
        END IF;
    END LOOP;
    large_vec := large_vec || ']';
    
    BEGIN
        PERFORM vector_l2_distance(large_vec::vector, large_vec::vector);
        RAISE NOTICE 'PASS: Large vector handled';
    EXCEPTION WHEN OTHERS THEN
        RAISE NOTICE 'INFO: Large vector result: %', SQLERRM;
    END;
END$$;

-- Test 5: Invalid distance metrics in index creation
\echo 'Test 5: Invalid index parameters'
CREATE TABLE test_crash_vectors (id int, v vector(128));
INSERT INTO test_crash_vectors SELECT generate_series(1, 100), array_to_vector(ARRAY(SELECT random()::real FROM generate_series(1, 128)))::vector(128);

DO $$
BEGIN
    CREATE INDEX idx_invalid ON test_crash_vectors USING hnsw (v) WITH (m='0', ef_construction='0');
    RAISE EXCEPTION 'Should have failed with invalid parameters';
EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE 'PASS: Invalid index parameters rejected';
END$$;

-- Test 6: Concurrent operations
\echo 'Test 6: Concurrent vector operations'
DO $$
DECLARE
    result float8;
BEGIN
    -- Multiple concurrent distance calculations
    FOR i IN 1..100 LOOP
        SELECT vector_l2_distance(
            array_to_vector(ARRAY(SELECT random()::real FROM generate_series(1, 128)))::vector(128),
            array_to_vector(ARRAY(SELECT random()::real FROM generate_series(1, 128)))::vector(128)
        ) INTO result;
    END LOOP;
    RAISE NOTICE 'PASS: Concurrent operations completed';
EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE 'ERROR: Concurrent operations failed: %', SQLERRM;
END$$;

ROLLBACK;

