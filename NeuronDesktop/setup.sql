/*-------------------------------------------------------------------------
 *
 * setup.sql
 *    Initial Setup Script for NeuronDesktop
 *
 * This is the initial setup SQL script for NeuronDesktop. It sets up everything
 * needed for NeuronDesktop to work with NeuronDB, including:
 * - uuid-ossp extension
 * - All tables (profiles, api_keys, request_logs, model_configs)
 * - All indexes for performance
 * - Default profile creation
 * - Permissions configuration
 *
 * This script is idempotent and can be run multiple times safely.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <admin@neurondb.com>
 *
 * IDENTIFICATION
 *    NeuronDesktop/setup.sql
 *
 *-------------------------------------------------------------------------
 *
 * PREREQUISITES
 * =============
 *
 * - PostgreSQL 16 or later
 * - Database user with CREATE privileges
 *
 * USAGE
 * =====
 *
 * To run this initial setup script on a database:
 *
 *   psql -d your_database -f setup.sql
 *
 * Or from within psql:
 *
 *   \i setup.sql
 *
 * This script is idempotent and can be run multiple times safely.
 * It will create all necessary database objects if they don't already exist.
 *
 *-------------------------------------------------------------------------
 */

-- ============================================================================
-- SECTION 1: EXTENSIONS
-- ============================================================================

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ============================================================================
-- SECTION 2: TABLES
-- ============================================================================

-- ============================================================================
-- Profiles Table
-- ============================================================================

CREATE TABLE IF NOT EXISTS profiles (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name TEXT NOT NULL,
    user_id TEXT NOT NULL,
    mcp_config JSONB,
    neurondb_dsn TEXT NOT NULL,
    agent_endpoint TEXT,
    agent_api_key TEXT,
    default_collection TEXT,
    is_default BOOLEAN NOT NULL DEFAULT false,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_profiles_user_id ON profiles(user_id);
CREATE INDEX IF NOT EXISTS idx_profiles_default ON profiles(is_default) WHERE is_default = true;

-- ============================================================================
-- API Keys Table
-- ============================================================================

CREATE TABLE IF NOT EXISTS api_keys (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    key_hash TEXT NOT NULL,
    key_prefix TEXT NOT NULL UNIQUE,
    user_id TEXT NOT NULL,
    profile_id UUID REFERENCES profiles(id) ON DELETE SET NULL,
    rate_limit INTEGER NOT NULL DEFAULT 100,
    last_used_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_api_keys_prefix ON api_keys(key_prefix);
CREATE INDEX IF NOT EXISTS idx_api_keys_user_id ON api_keys(user_id);

-- ============================================================================
-- Request Logs Table
-- ============================================================================

CREATE TABLE IF NOT EXISTS request_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    profile_id UUID REFERENCES profiles(id) ON DELETE SET NULL,
    endpoint TEXT NOT NULL,
    method TEXT NOT NULL,
    request_body JSONB,
    response_body JSONB,
    status_code INTEGER NOT NULL,
    duration_ms INTEGER NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_request_logs_profile_id ON request_logs(profile_id);
CREATE INDEX IF NOT EXISTS idx_request_logs_created_at ON request_logs(created_at DESC);

-- ============================================================================
-- Model Configs Table
-- ============================================================================

CREATE TABLE IF NOT EXISTS model_configs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    profile_id UUID NOT NULL REFERENCES profiles(id) ON DELETE CASCADE,
    model_provider TEXT NOT NULL,
    model_name TEXT NOT NULL,
    api_key TEXT,
    base_url TEXT,
    is_default BOOLEAN NOT NULL DEFAULT false,
    is_free BOOLEAN NOT NULL DEFAULT false,
    metadata JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(profile_id, model_provider, model_name)
);

CREATE INDEX IF NOT EXISTS idx_model_configs_profile_id ON model_configs(profile_id);
CREATE INDEX IF NOT EXISTS idx_model_configs_default ON model_configs(profile_id, is_default) WHERE is_default = true;

-- ============================================================================
-- SECTION 3: DEFAULT DATA
-- ============================================================================

-- Create default profile for current user
-- Note: This will only create a profile if one doesn't exist for the user
DO $$
DECLARE
    current_user_name TEXT;
    default_user_id TEXT := 'default';
    existing_profile_count INTEGER;
BEGIN
    -- Try to get current user, fallback to 'default'
    SELECT current_user INTO current_user_name;
    IF current_user_name IS NOT NULL THEN
        default_user_id := current_user_name;
    END IF;
    
    -- Check if a default profile already exists for this user
    SELECT COUNT(*) INTO existing_profile_count
    FROM profiles
    WHERE user_id = default_user_id AND is_default = true;
    
    -- Only create default profile if none exists
    IF existing_profile_count = 0 THEN
        -- Remove any existing default flags for this user (safety measure)
        UPDATE profiles SET is_default = false WHERE user_id = default_user_id;
        
        -- Insert default profile with basic configuration
        INSERT INTO profiles (
            id,
            name,
            user_id,
            neurondb_dsn,
            is_default,
            created_at,
            updated_at
        ) VALUES (
            uuid_generate_v4(),
            'Default',
            default_user_id,
            format('postgresql://%s@localhost:5432/neurondb', current_user_name),
            true,
            NOW(),
            NOW()
        );
    END IF;
END $$;

-- ============================================================================
-- SECTION 4: PERMISSIONS
-- ============================================================================

-- Grant all privileges to current user
DO $$
BEGIN
    -- Grant table privileges
    GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO CURRENT_USER;
    GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO CURRENT_USER;
    
    -- Set default privileges for future objects
    ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO CURRENT_USER;
    ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON SEQUENCES TO CURRENT_USER;
EXCEPTION
    WHEN OTHERS THEN
        -- Permissions may already be set, ignore errors
        NULL;
END $$;







