-- ============================================================================
-- NeuronAgent Schema Initialization Script
-- ============================================================================
-- This script automatically sets up the NeuronAgent database schema when
-- NeuronDB container is first initialized. It runs after the NeuronDB
-- extension is created (20_create_neurondb.sql).
--
-- The script is idempotent and safe to run multiple times.
-- ============================================================================

\echo '[neurondb_agent] Starting NeuronAgent schema setup...'

-- Check if NeuronDB extension exists (prerequisite)
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'neurondb') THEN
        RAISE NOTICE '[neurondb_agent] NeuronDB extension not found, creating it...';
        BEGIN
            CREATE EXTENSION IF NOT EXISTS neurondb;
        EXCEPTION
            WHEN OTHERS THEN
                -- If extension creation fails (e.g., shared_preload_libraries not configured),
                -- log a warning but don't fail the initialization
                RAISE WARNING '[neurondb_agent] Could not create NeuronDB extension: %. Extension will need to be created manually after PostgreSQL restart with shared_preload_libraries configured.', SQLERRM;
        END;
    ELSE
        RAISE NOTICE '[neurondb_agent] NeuronDB extension already exists';
    END IF;
END $$;

-- Check if NeuronAgent schema is already set up (for logging purposes)
DO $$
DECLARE
    schema_exists BOOLEAN;
BEGIN
    SELECT EXISTS (
        SELECT 1 
        FROM information_schema.tables 
        WHERE table_schema = 'neurondb_agent' 
        AND table_name = 'agents'
    ) INTO schema_exists;
    
    IF schema_exists THEN
        RAISE NOTICE '[neurondb_agent] NeuronAgent schema already exists (idempotent setup will skip existing objects)';
    ELSE
        RAISE NOTICE '[neurondb_agent] NeuronAgent schema not found, running migrations...';
    END IF;
END $$;

-- Include the NeuronAgent migration files in order
-- The files are copied to /docker-entrypoint-initdb.d/neurondb_agent/ by the Dockerfile
-- We use \set ON_ERROR_STOP off to ensure NeuronDB still starts even if NeuronAgent setup fails
-- The SQL files use CREATE IF NOT EXISTS, so they're idempotent and safe to run multiple times

\set ON_ERROR_STOP off

-- Run migrations in order
\echo '[neurondb_agent] Running migration 001_initial_schema.sql...'
\i /docker-entrypoint-initdb.d/neurondb_agent/001_initial_schema.sql

\echo '[neurondb_agent] Running migration 002_add_indexes.sql...'
\i /docker-entrypoint-initdb.d/neurondb_agent/002_add_indexes.sql

\echo '[neurondb_agent] Running migration 003_add_triggers.sql...'
\i /docker-entrypoint-initdb.d/neurondb_agent/003_add_triggers.sql

\echo '[neurondb_agent] Running migration 004_advanced_features.sql...'
\i /docker-entrypoint-initdb.d/neurondb_agent/004_advanced_features.sql

\echo '[neurondb_agent] Running migration 005_budget_schema.sql...'
\i /docker-entrypoint-initdb.d/neurondb_agent/005_budget_schema.sql

\echo '[neurondb_agent] Running migration 006_webhooks_schema.sql...'
\i /docker-entrypoint-initdb.d/neurondb_agent/006_webhooks_schema.sql

\echo '[neurondb_agent] Running migration 007_human_in_loop_schema.sql...'
\i /docker-entrypoint-initdb.d/neurondb_agent/007_human_in_loop_schema.sql

\set ON_ERROR_STOP on

-- Verify setup completed successfully
DO $$
DECLARE
    table_count INTEGER;
    index_count INTEGER;
BEGIN
    -- Count key tables
    SELECT COUNT(*) INTO table_count
    FROM information_schema.tables
    WHERE table_schema = 'neurondb_agent'
    AND table_name IN (
        'agents', 'sessions', 'messages', 'memory_chunks', 'tools', 'jobs', 'api_keys'
    );
    
    -- Count indexes on key tables
    SELECT COUNT(*) INTO index_count
    FROM pg_indexes
    WHERE schemaname = 'neurondb_agent'
    AND tablename IN ('sessions', 'messages', 'memory_chunks', 'jobs');
    
    IF table_count >= 5 AND index_count >= 3 THEN
        RAISE NOTICE '[neurondb_agent] NeuronAgent schema setup completed successfully!';
        RAISE NOTICE '[neurondb_agent] Found % key tables and % indexes', table_count, index_count;
    ELSE
        RAISE WARNING '[neurondb_agent] NeuronAgent schema setup may be incomplete';
        RAISE WARNING '[neurondb_agent] Found % key tables (expected >= 5) and % indexes (expected >= 3)', table_count, index_count;
    END IF;
END $$;

\echo '[neurondb_agent] NeuronAgent schema initialization complete'





