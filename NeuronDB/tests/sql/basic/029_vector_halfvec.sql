-- compatibility test: halfvec.sql
-- Tests halfvec type operations (similar to vector_type.sql)
-- Based on test/sql/halfvec.sql

\timing on
\pset footer off
\pset pager off

\set ON_ERROR_STOP on

\echo '=========================================================================='
\echo 'Compatibility Test: Halfvec Type Operations'
\echo '=========================================================================='

-- Test 1: Halfvec Parsing
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Test 1: Halfvec Parsing'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT '[1,2,3]'::halfvec;
SELECT '[-1,-2,-3]'::halfvec;
SELECT '[1.,2.,3.]'::halfvec;
SELECT ' [ 1,  2 ,    3  ] '::halfvec;

-- Test 2: Halfvec Arithmetic
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Test 2: Halfvec Arithmetic'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

-- Note: Halfvec arithmetic operators are not available in NeuronDB
-- Convert to vector for arithmetic operations, then back to halfvec
SELECT vector_to_halfvec(halfvec_to_vector('[1,2,3]'::halfvec) + halfvec_to_vector('[4,5,6]'::halfvec));
SELECT vector_to_halfvec(halfvec_to_vector('[1,2,3]'::halfvec) - halfvec_to_vector('[4,5,6]'::halfvec));
-- Vector concatenation operator now supported
SELECT vector_to_halfvec(halfvec_to_vector('[1,2,3]'::halfvec) || halfvec_to_vector('[4,5]'::halfvec));

-- Test 3: Halfvec Comparison
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Test 3: Halfvec Comparison'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

-- Comparison operators now implemented for halfvec
SELECT '[1,2,3]'::halfvec < '[1,2,4]'::halfvec;
SELECT '[1,2,3]'::halfvec <= '[1,2,3]'::halfvec;
SELECT '[1,2,3]'::halfvec = '[1,2,3]'::halfvec;
SELECT '[1,2,3]'::halfvec != '[1,2,3]'::halfvec;
SELECT '[1,2,4]'::halfvec >= '[1,2,3]'::halfvec;
SELECT '[1,2,4]'::halfvec > '[1,2,3]'::halfvec;

-- Test 4: Halfvec Distance Functions
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Test 4: Halfvec Distance Functions'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT halfvec_l2_distance('[0,0]'::halfvec, '[3,4]');
SELECT '[0,0]'::halfvec <-> '[3,4]';

SELECT halfvec_inner_product('[1,2]'::halfvec, '[3,4]');
SELECT '[1,2]'::halfvec <#> '[3,4]';

SELECT halfvec_cosine_distance('[1,2]'::halfvec, '[2,4]');
SELECT '[1,2]'::halfvec <=> '[2,4]';

-- L1 distance operator now implemented for halfvec
SELECT '[0,0]'::halfvec <+> '[3,4]'::halfvec;

-- Test 5: Halfvec Normalization
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Test 5: Halfvec Normalization'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

-- Note: NeuronDB may not have l2_norm/l2_normalize for halfvec
-- Convert to vector for normalization if needed
SELECT vector_norm('[3,4]'::halfvec::vector);
SELECT vector_normalize('[3,4]'::halfvec::vector)::halfvec;

-- Test 6: Halfvec Aggregates
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Test 6: Halfvec Aggregates'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

-- SELECT avg(v) FROM unnest(ARRAY['[1,2,3]'::halfvec, '[3,5,7]']) v;
-- SELECT sum(v) FROM unnest(ARRAY['[1,2,3]'::halfvec, '[3,5,7]']) v;

\echo ''
\echo '=========================================================================='
\echo 'Test completed successfully'
\echo '=========================================================================='

