-- ====================================================================
-- NeurondB Test Database Setup
-- ====================================================================
-- Sets up the neurondb_test database with all necessary extensions
-- ====================================================================

\echo '=== Setting up NeurondB Test Database ==='

-- Ensure we're connected to the test database
\c neurondb_test

-- Install NeurondB extension with all dependencies
DROP EXTENSION IF EXISTS neurondb CASCADE;
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE EXTENSION neurondb CASCADE;

-- Verify extension installation
SELECT extname, extversion 
FROM pg_extension 
WHERE extname IN ('neurondb', 'pg_trgm', 'vector');

-- Create schema for datasets if not exists
CREATE SCHEMA IF NOT EXISTS neurondb_datasets;

-- Grant necessary permissions
GRANT USAGE ON SCHEMA neurondb_datasets TO PUBLIC;
GRANT SELECT ON ALL TABLES IN SCHEMA neurondb_datasets TO PUBLIC;
ALTER DEFAULT PRIVILEGES IN SCHEMA neurondb_datasets 
    GRANT SELECT ON TABLES TO PUBLIC;

-- Configure NeurondB settings for testing
SET neurondb.enable_gpu = false;  -- CPU-only for consistent results
SET neurondb.enable_parallel = true;
SET neurondb.work_mem = '256MB';

\echo '=== Test Database Setup Complete ==='

