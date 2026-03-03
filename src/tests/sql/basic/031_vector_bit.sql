-- compatibility test: bit.sql
-- Tests bit type operations (Hamming/Jaccard distance)
-- Based on test/sql/bit.sql

\timing on
\pset footer off
\pset pager off

\set ON_ERROR_STOP on

\echo '=========================================================================='
\echo 'Compatibility Test: Bit Type Operations'
\echo '=========================================================================='

-- Test 1: Hamming Distance
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Test 1: Hamming Distance'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT bit_hamming_distance('111'::bit, '111'::bit);
SELECT bit_hamming_distance('111'::bit, '110'::bit);
SELECT bit_hamming_distance('111'::bit, '100'::bit);
SELECT bit_hamming_distance('111'::bit, '000'::bit);
SELECT bit_hamming_distance(''::bit, ''::bit);
SELECT bit_hamming_distance('111'::bit, '00'::bit);

-- Test 2: Jaccard Distance
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Test 2: Jaccard Distance'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT bit_jaccard_distance('1111'::bit, '1111'::bit);
SELECT bit_jaccard_distance('1111'::bit, '1110'::bit);
SELECT bit_jaccard_distance('1111'::bit, '1100'::bit);
SELECT bit_jaccard_distance('1111'::bit, '1000'::bit);
SELECT bit_jaccard_distance('1111'::bit, '0000'::bit);
SELECT bit_jaccard_distance('1100'::bit, '1000'::bit);
SELECT bit_jaccard_distance(''::bit, ''::bit);

-- Test 3: Bit Operators
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Test 3: Bit Distance Operators'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT B'111' <~> B'110';
SELECT B'1111' <%> B'1110';

\echo ''
\echo '=========================================================================='
\echo 'Test completed successfully'
\echo '=========================================================================='

