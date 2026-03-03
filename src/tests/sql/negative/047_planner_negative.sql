-- 034_planner_negative.sql
-- Negative test cases for planner module: error handling

\timing on
\pset footer off
\pset pager off

\set ON_ERROR_STOP off

\echo '=========================================================================='
\echo 'Planner Module: Negative Test Cases (Error Handling)'
\echo '=========================================================================='

\echo ''
\echo 'Planner Error Tests'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

\echo 'Error Test 1: Query plan with invalid table'
EXPLAIN SELECT * FROM nonexistent_planner_table;

\echo 'Error Test 2: Query plan with invalid index'
DROP TABLE IF EXISTS planner_error_test;
CREATE TABLE planner_error_test (
	id SERIAL PRIMARY KEY,
	embedding vector(28)
);

EXPLAIN SELECT * FROM planner_error_test 
WHERE embedding <-> vector '[1,2,3]'::vector < 10.0;

DROP TABLE IF EXISTS planner_error_test;

\echo ''
\echo '=========================================================================='
\echo '✓ Planner Module: Negative test cases complete'
\echo '=========================================================================='

\echo 'Test completed successfully'




