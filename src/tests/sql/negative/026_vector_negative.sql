-- 026_vector_negative.sql
-- Negative test for vector operations
-- All possible negative tests with proper error handling

SET client_min_messages TO WARNING;

\pset null [NULL]
\pset format aligned


DO $$ BEGIN
    PERFORM vector_get(vector '[1,2,3]', -1);
    RAISE EXCEPTION 'Should have failed';
EXCEPTION WHEN OTHERS THEN
END $$;

DO $$ BEGIN
    PERFORM vector_get(vector '[1,2,3]', 100);
    RAISE EXCEPTION 'Should have failed';
EXCEPTION WHEN OTHERS THEN
END $$;

DO $$ BEGIN
    PERFORM vector_set(vector '[1,2,3]', -1, 99.0);
    RAISE EXCEPTION 'Should have failed';
EXCEPTION WHEN OTHERS THEN
END $$;

DO $$ BEGIN
    PERFORM vector_set(vector '[1,2,3]', 100, 99.0);
    RAISE EXCEPTION 'Should have failed';
EXCEPTION WHEN OTHERS THEN
END $$;

DO $$ BEGIN
    PERFORM vector '[1,2,3]' + vector '[1,2]';
    RAISE EXCEPTION 'Should have failed';
EXCEPTION WHEN OTHERS THEN
END $$;

DO $$ BEGIN
    PERFORM vector_inner_product(vector '[1,2,3]', vector '[1,2]');
    RAISE EXCEPTION 'Should have failed';
EXCEPTION WHEN OTHERS THEN
END $$;

DO $$ BEGIN
    PERFORM vector_get(NULL::vector, 0);
    RAISE EXCEPTION 'Should have failed';
EXCEPTION WHEN OTHERS THEN
END $$;

DO $$ BEGIN
    PERFORM vector_dims(NULL::vector);
    RAISE EXCEPTION 'Should have failed';
EXCEPTION WHEN OTHERS THEN
END $$;

DO $$ BEGIN
    PERFORM vector('invalid_syntax'::text);
    RAISE EXCEPTION 'Should have failed';
EXCEPTION WHEN OTHERS THEN
END $$;

DO $$ BEGIN
    PERFORM array_to_vector(ARRAY[NULL, 2, 3]::float8[]);
    RAISE EXCEPTION 'Should have failed';
EXCEPTION WHEN OTHERS THEN
END $$;

DO $$ BEGIN
    PERFORM vector_cast_dimension(vector '[1,2,3]', -1);
    RAISE EXCEPTION 'Should have failed';
EXCEPTION WHEN OTHERS THEN
END $$;

DO $$ BEGIN
    PERFORM vector_cast_dimension(vector '[1,2,3]', 16001);
    RAISE EXCEPTION 'Should have failed';
EXCEPTION WHEN OTHERS THEN
END $$;

DO $$ BEGIN
    PERFORM vector_l2_distance_batch(ARRAY[]::vector[], vector '[1,2,3]');
    RAISE EXCEPTION 'Should have failed';
EXCEPTION WHEN OTHERS THEN
END $$;

DO $$ BEGIN
    PERFORM vector_cross_product(vector '[1,2]', vector '[3,4]');
    RAISE EXCEPTION 'Should have failed';
EXCEPTION WHEN OTHERS THEN
END $$;

DO $$ BEGIN
    PERFORM vector_percentile(vector '[1,2,3]', -0.1);
    RAISE EXCEPTION 'Should have failed';
EXCEPTION WHEN OTHERS THEN
END $$;

DO $$ BEGIN
    PERFORM vector_percentile(vector '[1,2,3]', 1.5);
    RAISE EXCEPTION 'Should have failed';
EXCEPTION WHEN OTHERS THEN
END $$;

DO $$ BEGIN
    PERFORM vector_scale(vector '[1,2,3]', ARRAY[2.0, 3.0]::real[]);
    RAISE EXCEPTION 'Should have failed';
EXCEPTION WHEN OTHERS THEN
END $$;

DO $$ BEGIN
    PERFORM vector_filter(vector '[1,2,3]', ARRAY[true, false]::boolean[]);
    RAISE EXCEPTION 'Should have failed';
EXCEPTION WHEN OTHERS THEN
END $$;

\echo 'Test completed successfully'
