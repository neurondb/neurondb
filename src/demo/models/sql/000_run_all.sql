-- ============================================================================
-- NeuronDB Model Catalog Demo - Run All
-- ============================================================================
\echo '=========================================='
\echo 'NeuronDB Model Catalog Demo'
\echo '=========================================='
\timing on

\echo ''
\echo 'Running 001_model_registration.sql'
\i 001_model_registration.sql

\echo ''
\echo 'Running 002_model_lifecycle.sql'
\i 002_model_lifecycle.sql

\echo ''
\echo 'Running 003_model_audit.sql'
\i 003_model_audit.sql

\echo ''
\echo 'Running 004_model_error_cases.sql'
\i 004_model_error_cases.sql

\echo ''
\echo 'Running 005_cleanup.sql'
\i 005_cleanup.sql

\timing off
\echo ''
\echo '=========================================='
\echo 'Model Catalog Demo Complete'
\echo '=========================================='
