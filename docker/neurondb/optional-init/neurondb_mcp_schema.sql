-- ============================================================================
-- NeuronMCP Schema Initialization Script
-- ============================================================================
-- This script automatically sets up the NeuronMCP database schema when
-- NeuronDB container is first initialized. It runs after the NeuronDB
-- extension is created (20_create_neurondb.sql).
--
-- The script is idempotent and safe to run multiple times.
-- ============================================================================

\echo '[neurondb_mcp] Starting NeuronMCP schema setup...'

-- Check if NeuronDB extension exists (prerequisite)
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'neurondb') THEN
        RAISE NOTICE '[neurondb_mcp] NeuronDB extension not found, creating it...';
        BEGIN
            CREATE EXTENSION IF NOT EXISTS neurondb;
        EXCEPTION
            WHEN OTHERS THEN
                -- If extension creation fails (e.g., shared_preload_libraries not configured),
                -- log a warning but don't fail the initialization
                RAISE WARNING '[neurondb_mcp] Could not create NeuronDB extension: %. Extension will need to be created manually after PostgreSQL restart with shared_preload_libraries configured.', SQLERRM;
        END;
    ELSE
        RAISE NOTICE '[neurondb_mcp] NeuronDB extension already exists';
    END IF;
END $$;

-- Check if NeuronMCP schema is already set up (for logging purposes)
DO $$
DECLARE
    schema_exists BOOLEAN;
BEGIN
    SELECT EXISTS (
        SELECT 1 
        FROM information_schema.tables 
        WHERE table_schema = 'neurondb' 
        AND table_name = 'llm_providers'
    ) INTO schema_exists;
    
    IF schema_exists THEN
        RAISE NOTICE '[neurondb_mcp] NeuronMCP schema already exists (idempotent setup will skip existing objects)';
    ELSE
        RAISE NOTICE '[neurondb_mcp] NeuronMCP schema not found, running setup...';
    END IF;
END $$;

-- Include the NeuronMCP schema and functions setup files
-- The files are copied to /docker-entrypoint-initdb.d/neurondb_mcp/ by the Dockerfile
-- We use \set ON_ERROR_STOP off to ensure NeuronDB still starts even if NeuronMCP setup fails
-- The SQL files use CREATE IF NOT EXISTS, so they're idempotent and safe to run multiple times

\set ON_ERROR_STOP off

-- Include schema setup
\echo '[neurondb_mcp] Running schema setup...'
\i /docker-entrypoint-initdb.d/neurondb_mcp/setup_neurondb_mcp_schema.sql

-- Include functions setup  
\echo '[neurondb_mcp] Running functions setup...'
\i /docker-entrypoint-initdb.d/neurondb_mcp/neurondb_mcp_functions.sql

\set ON_ERROR_STOP on

-- Verify setup completed successfully
DO $$
DECLARE
    table_count INTEGER;
    function_count INTEGER;
BEGIN
    -- Count key tables
    SELECT COUNT(*) INTO table_count
    FROM information_schema.tables
    WHERE table_schema = 'neurondb'
    AND table_name IN (
        'llm_providers', 'llm_models', 'llm_model_keys',
        'index_configs', 'worker_configs', 'ml_default_configs'
    );
    
    -- Count key functions
    SELECT COUNT(*) INTO function_count
    FROM pg_proc p
    JOIN pg_namespace n ON p.pronamespace = n.oid
    WHERE n.nspname = 'neurondb'
    AND p.proname IN (
        'neurondb_set_model_key',
        'neurondb_get_model_key',
        'neurondb_list_models'
    );
    
    IF table_count >= 3 AND function_count >= 1 THEN
        RAISE NOTICE '[neurondb_mcp] NeuronMCP schema setup completed successfully!';
        RAISE NOTICE '[neurondb_mcp] Found % key tables and % key functions', table_count, function_count;
    ELSE
        RAISE WARNING '[neurondb_mcp] NeuronMCP schema setup may be incomplete';
        RAISE WARNING '[neurondb_mcp] Found % key tables (expected >= 3) and % key functions (expected >= 1)', table_count, function_count;
    END IF;
END $$;

\echo '[neurondb_mcp] NeuronMCP schema initialization complete'

