-- NeuronDB Extension Initialization Script
-- This script automatically creates the NeuronDB extension in the default database
-- when the container is first initialized.
--
-- Note: During initdb, shared_preload_libraries is configured but PostgreSQL
-- hasn't restarted yet, so extension creation may fail. The custom entrypoint
-- (docker-entrypoint-neurondb.sh) will ensure the extension is created after
-- PostgreSQL starts.

-- Clean up any orphaned neurondb schema that exists without the extension
-- This can happen if a previous initialization partially completed
DO $$
BEGIN
    -- Check if schema exists but extension doesn't
    IF EXISTS (
        SELECT 1 FROM pg_namespace WHERE nspname = 'neurondb'
    ) AND NOT EXISTS (
        SELECT 1 FROM pg_extension WHERE extname = 'neurondb'
    ) THEN
        -- Drop the orphaned schema so the extension can create it properly
        DROP SCHEMA IF EXISTS neurondb CASCADE;
        RAISE NOTICE 'Dropped orphaned neurondb schema (extension will recreate it)';
    END IF;
END
$$;

-- Try to create the extension if it doesn't exist
-- Handle the case where shared_preload_libraries might not be configured yet
DO $$
BEGIN
    CREATE EXTENSION IF NOT EXISTS neurondb;
    RAISE NOTICE 'NeuronDB extension created successfully during initialization';
EXCEPTION
    WHEN OTHERS THEN
        -- If extension creation fails (e.g., shared_preload_libraries not loaded yet),
        -- this is expected during initdb. The custom entrypoint will create it after restart.
        RAISE NOTICE 'NeuronDB extension will be created automatically after PostgreSQL starts (shared_preload_libraries requires restart)';
END
$$;

-- Display extension information (only if extension was created successfully)
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'neurondb') THEN
        PERFORM extname, extversion FROM pg_extension WHERE extname = 'neurondb';
    END IF;
END
$$;
