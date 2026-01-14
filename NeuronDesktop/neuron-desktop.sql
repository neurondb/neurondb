/*-------------------------------------------------------------------------
 *
 * neuron-desktop.sql
 *    Complete NeuronDesktop Database Setup Script
 *
 * This script sets up everything needed for NeuronDesktop:
 * - Database schema (tables, indexes, views, functions)
 * - All migrations consolidated
 * - Default data and permissions
 *
 * This script is idempotent and can be run multiple times safely.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronDesktop/neuron-desktop.sql
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
 * To run this setup script on a database:
 *
 *   psql -d your_database -f neuron-desktop.sql
 *
 * Or from within psql:
 *
 *   \i neuron-desktop.sql
 *
 *-------------------------------------------------------------------------
 */

-- ============================================================================
-- SECTION 1: EXTENSIONS
-- ============================================================================

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ============================================================================
-- SECTION 2: CORE TABLES
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
-- SECTION 3: MIGRATIONS
-- ============================================================================

-- Migration: Add profile username and password fields
ALTER TABLE profiles 
ADD COLUMN IF NOT EXISTS profile_username TEXT;

ALTER TABLE profiles 
ADD COLUMN IF NOT EXISTS profile_password_hash TEXT;

CREATE INDEX IF NOT EXISTS idx_profiles_username ON profiles(profile_username) WHERE profile_username IS NOT NULL;

COMMENT ON COLUMN profiles.profile_username IS 'Username for this profile. When user logs in with this username/password, they are automatically on this profile.';
COMMENT ON COLUMN profiles.profile_password_hash IS 'Bcrypt hash of the profile password. Never store plain text passwords.';

-- App settings table for storing application-level configuration
CREATE TABLE IF NOT EXISTS app_settings (
    key TEXT PRIMARY KEY,
    value JSONB NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_app_settings_updated_at ON app_settings(updated_at);

-- Users table for authentication
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    username TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);

-- Add is_admin flag to users for role-based access control
ALTER TABLE users
ADD COLUMN IF NOT EXISTS is_admin BOOLEAN NOT NULL DEFAULT FALSE;

CREATE INDEX IF NOT EXISTS idx_users_is_admin ON users(is_admin);

-- Sessions table for session management
CREATE TABLE IF NOT EXISTS sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_seen_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    revoked_at TIMESTAMPTZ,
    user_agent_hash TEXT,
    ip_hash TEXT,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_sessions_revoked_at ON sessions(revoked_at);
CREATE INDEX IF NOT EXISTS idx_sessions_last_seen_at ON sessions(last_seen_at);

-- Refresh tokens table for token rotation
CREATE TABLE IF NOT EXISTS refresh_tokens (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id UUID NOT NULL,
    token_hash TEXT NOT NULL UNIQUE,
    rotated_from UUID,
    expires_at TIMESTAMPTZ NOT NULL,
    revoked_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE,
    FOREIGN KEY (rotated_from) REFERENCES refresh_tokens(id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_refresh_tokens_session_id ON refresh_tokens(session_id);
CREATE INDEX IF NOT EXISTS idx_refresh_tokens_token_hash ON refresh_tokens(token_hash);
CREATE INDEX IF NOT EXISTS idx_refresh_tokens_expires_at ON refresh_tokens(expires_at);
CREATE INDEX IF NOT EXISTS idx_refresh_tokens_revoked_at ON refresh_tokens(revoked_at);

-- OIDC identities table for linking OIDC subjects to users
CREATE TABLE IF NOT EXISTS oidc_identities (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    issuer TEXT NOT NULL,
    subject TEXT NOT NULL,
    user_id UUID NOT NULL,
    email TEXT,
    name TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    UNIQUE(issuer, subject)
);

CREATE INDEX IF NOT EXISTS idx_oidc_identities_user_id ON oidc_identities(user_id);
CREATE INDEX IF NOT EXISTS idx_oidc_identities_issuer_subject ON oidc_identities(issuer, subject);

-- Audit log table for security events
CREATE TABLE IF NOT EXISTS audit_log (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    event_type TEXT NOT NULL,
    user_id UUID,
    ip_hash TEXT,
    user_agent_hash TEXT,
    metadata JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_audit_log_event_type ON audit_log(event_type);
CREATE INDEX IF NOT EXISTS idx_audit_log_user_id ON audit_log(user_id);
CREATE INDEX IF NOT EXISTS idx_audit_log_created_at ON audit_log(created_at);

-- Login attempts table for OIDC state/nonce tracking
CREATE TABLE IF NOT EXISTS login_attempts (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    state TEXT NOT NULL UNIQUE,
    nonce TEXT NOT NULL,
    code_verifier TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    expires_at TIMESTAMPTZ NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_login_attempts_state ON login_attempts(state);
CREATE INDEX IF NOT EXISTS idx_login_attempts_expires_at ON login_attempts(expires_at);

-- Migration: Add MCP Chat Threads and Messages tables
CREATE TABLE IF NOT EXISTS mcp_chat_threads (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    profile_id UUID NOT NULL REFERENCES profiles(id) ON DELETE CASCADE,
    title TEXT NOT NULL DEFAULT 'New chat',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_mcp_threads_profile_id ON mcp_chat_threads(profile_id);
CREATE INDEX IF NOT EXISTS idx_mcp_threads_updated_at ON mcp_chat_threads(updated_at DESC);

CREATE TABLE IF NOT EXISTS mcp_chat_messages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    thread_id UUID NOT NULL REFERENCES mcp_chat_threads(id) ON DELETE CASCADE,
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'system', 'tool')),
    content TEXT NOT NULL,
    tool_name TEXT,
    data JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_mcp_messages_thread_id ON mcp_chat_messages(thread_id);
CREATE INDEX IF NOT EXISTS idx_mcp_messages_created_at ON mcp_chat_messages(created_at);

-- Migration: Add agent templates and workflows tables
CREATE TABLE IF NOT EXISTS agent_templates (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT NOT NULL UNIQUE,
    description TEXT,
    category TEXT,
    configuration JSONB NOT NULL DEFAULT '{}',
    workflow JSONB,
    popularity INTEGER DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_agent_templates_category ON agent_templates(category);
CREATE INDEX IF NOT EXISTS idx_agent_templates_popularity ON agent_templates(popularity DESC);

CREATE TABLE IF NOT EXISTS agent_workflows (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id UUID,
    name TEXT NOT NULL,
    workflow_definition JSONB NOT NULL,
    version INTEGER DEFAULT 1,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_agent_workflows_agent_id ON agent_workflows(agent_id);

CREATE TABLE IF NOT EXISTS user_agent_templates (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID,
    agent_id TEXT NOT NULL,
    template_id UUID REFERENCES agent_templates(id) ON DELETE SET NULL,
    name TEXT NOT NULL,
    custom_config JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_user_agent_templates_user_id ON user_agent_templates(user_id);
CREATE INDEX IF NOT EXISTS idx_user_agent_templates_template_id ON user_agent_templates(template_id);

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_agent_templates_updated_at
    BEFORE UPDATE ON agent_templates
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_agent_workflows_updated_at
    BEFORE UPDATE ON agent_workflows
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- ============================================================================
-- Unified Identity Model Migration
-- ============================================================================

-- Create Organizations Table
CREATE TABLE IF NOT EXISTS organizations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name TEXT NOT NULL,
    slug TEXT NOT NULL UNIQUE,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_organizations_slug ON organizations(slug);

-- Create Service Accounts Table
CREATE TABLE IF NOT EXISTS service_accounts (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name TEXT NOT NULL,
    org_id UUID REFERENCES organizations(id) ON DELETE CASCADE,
    project_id UUID REFERENCES profiles(id) ON DELETE CASCADE,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_service_accounts_org_id ON service_accounts(org_id);
CREATE INDEX IF NOT EXISTS idx_service_accounts_project_id ON service_accounts(project_id);

-- Update Users Table (if not exists, create it)
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'users' AND column_name = 'email'
    ) THEN
        ALTER TABLE users ADD COLUMN email TEXT;
    END IF;
    
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'users' AND column_name = 'metadata'
    ) THEN
        ALTER TABLE users ADD COLUMN metadata JSONB DEFAULT '{}';
    END IF;
END $$;

CREATE INDEX IF NOT EXISTS idx_users_email ON users(email) WHERE email IS NOT NULL;

-- Update Profiles Table to Support Unified Model
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'profiles' AND column_name = 'org_id'
    ) THEN
        ALTER TABLE profiles ADD COLUMN org_id UUID REFERENCES organizations(id) ON DELETE SET NULL;
    END IF;
END $$;

-- Convert user_id from TEXT to UUID if needed
DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'profiles' 
        AND column_name = 'user_id' 
        AND data_type = 'text'
    ) THEN
        ALTER TABLE profiles ADD COLUMN user_id_uuid UUID REFERENCES users(id) ON DELETE CASCADE;
        
        INSERT INTO users (id, username, created_at)
        SELECT DISTINCT 
            COALESCE(
                (SELECT id FROM users WHERE username = profiles.user_id),
                uuid_generate_v4()
            ),
            profiles.user_id,
            NOW()
        FROM profiles
        WHERE profiles.user_id IS NOT NULL
        ON CONFLICT (username) DO NOTHING;
        
        UPDATE profiles 
        SET user_id_uuid = users.id
        FROM users
        WHERE profiles.user_id = users.username;
        
        ALTER TABLE profiles DROP COLUMN user_id;
        ALTER TABLE profiles RENAME COLUMN user_id_uuid TO user_id;
    END IF;
END $$;

CREATE INDEX IF NOT EXISTS idx_profiles_org_id ON profiles(org_id) WHERE org_id IS NOT NULL;

-- Create Principals Table (for unified identity)
CREATE TABLE IF NOT EXISTS principals (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    type TEXT NOT NULL CHECK (type IN ('user', 'org', 'agent', 'tool', 'dataset', 'service_account')),
    name TEXT NOT NULL,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT unique_principal_name_per_type UNIQUE (type, name)
);

CREATE INDEX IF NOT EXISTS idx_principals_type ON principals(type);
CREATE INDEX IF NOT EXISTS idx_principals_name ON principals(name);

-- Update API Keys Table to Support Unified Model
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'api_keys' AND column_name = 'principal_id'
    ) THEN
        ALTER TABLE api_keys ADD COLUMN principal_id UUID REFERENCES principals(id) ON DELETE SET NULL;
    END IF;
    
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'api_keys' AND column_name = 'principal_type'
    ) THEN
        ALTER TABLE api_keys ADD COLUMN principal_type TEXT CHECK (principal_type IN ('user', 'org', 'service_account'));
    END IF;
    
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'api_keys' AND column_name = 'project_id'
    ) THEN
        ALTER TABLE api_keys ADD COLUMN project_id UUID REFERENCES profiles(id) ON DELETE SET NULL;
    END IF;
END $$;

-- Migrate existing user_id to principal_id
DO $$
DECLARE
    user_principal_id UUID;
    user_record RECORD;
BEGIN
    FOR user_record IN SELECT DISTINCT user_id FROM api_keys WHERE user_id IS NOT NULL LOOP
        SELECT id INTO user_principal_id
        FROM principals
        WHERE type = 'user' AND name = user_record.user_id;
        
        IF user_principal_id IS NULL THEN
            INSERT INTO principals (type, name, created_at)
            VALUES ('user', user_record.user_id, NOW())
            RETURNING id INTO user_principal_id;
        END IF;
        
        UPDATE api_keys
        SET principal_id = user_principal_id,
            principal_type = 'user'
        WHERE user_id = user_record.user_id;
    END LOOP;
END $$;

CREATE INDEX IF NOT EXISTS idx_api_keys_principal_id ON api_keys(principal_id) WHERE principal_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_api_keys_principal_type ON api_keys(principal_type);
CREATE INDEX IF NOT EXISTS idx_api_keys_project_id ON api_keys(project_id) WHERE project_id IS NOT NULL;

-- Update Audit Log Table
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'audit_log' AND column_name = 'principal_id'
    ) THEN
        ALTER TABLE audit_log ADD COLUMN principal_id UUID REFERENCES principals(id) ON DELETE SET NULL;
    END IF;
    
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'audit_log' AND column_name = 'project_id'
    ) THEN
        ALTER TABLE audit_log ADD COLUMN project_id UUID REFERENCES profiles(id) ON DELETE SET NULL;
    END IF;
END $$;

-- Migrate existing user_id to principal_id in audit_log
DO $$
DECLARE
    user_principal_id UUID;
    user_record RECORD;
BEGIN
    FOR user_record IN SELECT DISTINCT user_id FROM audit_log WHERE user_id IS NOT NULL LOOP
        SELECT id INTO user_principal_id
        FROM principals
        WHERE type = 'user' AND name = user_record.user_id::TEXT;
        
        IF user_principal_id IS NULL THEN
            INSERT INTO principals (type, name, created_at)
            VALUES ('user', user_record.user_id::TEXT, NOW())
            RETURNING id INTO user_principal_id;
        END IF;
        
        UPDATE audit_log
        SET principal_id = user_principal_id
        WHERE user_id = user_record.user_id;
    END LOOP;
END $$;

CREATE INDEX IF NOT EXISTS idx_audit_log_principal_id ON audit_log(principal_id) WHERE principal_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_audit_log_project_id ON audit_log(project_id) WHERE project_id IS NOT NULL;

-- Create Policies Table (for unified permissions)
CREATE TABLE IF NOT EXISTS policies (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    principal_id UUID NOT NULL REFERENCES principals(id) ON DELETE CASCADE,
    resource_type TEXT NOT NULL,
    resource_id TEXT,
    permissions TEXT[] NOT NULL DEFAULT '{}',
    conditions JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT valid_permissions CHECK (array_length(permissions, 1) > 0)
);

CREATE INDEX IF NOT EXISTS idx_policies_principal_id ON policies(principal_id);
CREATE INDEX IF NOT EXISTS idx_policies_resource ON policies(resource_type, resource_id);

-- Create Views for Convenience
CREATE OR REPLACE VIEW unified_identities AS
SELECT 
    'user' AS identity_type,
    id AS identity_id,
    username AS identity_name,
    email,
    is_admin,
    NULL::UUID AS org_id,
    NULL::UUID AS project_id,
    created_at,
    updated_at
FROM users
UNION ALL
SELECT 
    'org' AS identity_type,
    id AS identity_id,
    name AS identity_name,
    NULL AS email,
    FALSE AS is_admin,
    NULL::UUID AS org_id,
    NULL::UUID AS project_id,
    created_at,
    updated_at
FROM organizations
UNION ALL
SELECT 
    'service_account' AS identity_type,
    id AS identity_id,
    name AS identity_name,
    NULL AS email,
    FALSE AS is_admin,
    org_id,
    project_id,
    created_at,
    updated_at
FROM service_accounts;

CREATE OR REPLACE VIEW api_keys_with_principals AS
SELECT 
    ak.id,
    ak.key_prefix,
    ak.principal_id,
    ak.principal_type,
    ak.project_id,
    ak.rate_limit,
    ak.last_used_at,
    ak.created_at,
    p.name AS principal_name,
    p.metadata AS principal_metadata
FROM api_keys ak
LEFT JOIN principals p ON ak.principal_id = p.id;

-- Create Helper Functions
CREATE OR REPLACE FUNCTION get_or_create_user_principal(user_uuid UUID, user_name TEXT)
RETURNS UUID AS $$
DECLARE
    principal_id UUID;
BEGIN
    SELECT id INTO principal_id
    FROM principals
    WHERE type = 'user' AND name = user_name;
    
    IF principal_id IS NULL THEN
        INSERT INTO principals (type, name, metadata, created_at)
        VALUES ('user', user_name, jsonb_build_object('user_id', user_uuid), NOW())
        RETURNING id INTO principal_id;
    END IF;
    
    RETURN principal_id;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION get_principal_for_api_key(api_key_prefix TEXT)
RETURNS TABLE (
    principal_id UUID,
    principal_type TEXT,
    principal_name TEXT,
    project_id UUID
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        ak.principal_id,
        ak.principal_type,
        p.name,
        ak.project_id
    FROM api_keys ak
    LEFT JOIN principals p ON ak.principal_id = p.id
    WHERE ak.key_prefix = api_key_prefix;
END;
$$ LANGUAGE plpgsql;

-- Audit logs table for tracking sensitive operations
CREATE TABLE IF NOT EXISTS audit_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id TEXT NOT NULL,
    action TEXT NOT NULL,
    resource_type TEXT NOT NULL,
    resource_id TEXT,
    details JSONB,
    ip_address TEXT,
    user_agent TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_audit_logs_user_id ON audit_logs(user_id);
CREATE INDEX IF NOT EXISTS idx_audit_logs_action ON audit_logs(action);
CREATE INDEX IF NOT EXISTS idx_audit_logs_resource_type ON audit_logs(resource_type);
CREATE INDEX IF NOT EXISTS idx_audit_logs_created_at ON audit_logs(created_at DESC);

-- OIDC Hardening Migration
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'login_attempts' AND column_name = 'redirect_uri'
    ) THEN
        ALTER TABLE login_attempts ADD COLUMN redirect_uri TEXT;
    END IF;
END $$;

CREATE INDEX IF NOT EXISTS idx_login_attempts_expires_at_cleanup 
ON login_attempts(expires_at) WHERE expires_at < NOW();

CREATE OR REPLACE FUNCTION cleanup_expired_login_attempts()
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM login_attempts
    WHERE expires_at < NOW() - INTERVAL '1 hour';
    
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION cleanup_expired_login_attempts() IS 
'Cleans up expired login attempts older than 1 hour. Should be run periodically via cron or scheduled job.';

-- Organizations and organization membership
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'organizations' AND column_name = 'description'
    ) THEN
        ALTER TABLE organizations ADD COLUMN description TEXT;
    END IF;
    
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'organizations' AND column_name = 'settings'
    ) THEN
        ALTER TABLE organizations ADD COLUMN settings JSONB DEFAULT '{}'::jsonb;
    END IF;
END $$;

CREATE TABLE IF NOT EXISTS organization_members (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    organization_id UUID NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    role TEXT NOT NULL DEFAULT 'member',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(organization_id, user_id)
);

CREATE INDEX IF NOT EXISTS idx_org_members_org_id ON organization_members(organization_id);
CREATE INDEX IF NOT EXISTS idx_org_members_user_id ON organization_members(user_id);
CREATE INDEX IF NOT EXISTS idx_org_members_role ON organization_members(role);

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'profiles' AND column_name = 'organization_id'
    ) THEN
        ALTER TABLE profiles ADD COLUMN organization_id UUID REFERENCES organizations(id) ON DELETE SET NULL;
    END IF;
END $$;

CREATE INDEX IF NOT EXISTS idx_profiles_organization_id ON profiles(organization_id);

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'api_keys' AND column_name = 'organization_id'
    ) THEN
        ALTER TABLE api_keys ADD COLUMN organization_id UUID REFERENCES organizations(id) ON DELETE SET NULL;
    END IF;
END $$;

CREATE INDEX IF NOT EXISTS idx_api_keys_organization_id ON api_keys(organization_id);

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'audit_logs' AND column_name = 'organization_id'
    ) THEN
        ALTER TABLE audit_logs ADD COLUMN organization_id UUID REFERENCES organizations(id) ON DELETE SET NULL;
    END IF;
END $$;

CREATE INDEX IF NOT EXISTS idx_audit_logs_organization_id ON audit_logs(organization_id);

-- ============================================================================
-- SECTION 4: DEFAULT DATA
-- ============================================================================

-- Create default profile for current user
DO $$
DECLARE
    current_user_name TEXT;
    default_user_id TEXT := 'default';
    existing_profile_count INTEGER;
BEGIN
    SELECT current_user INTO current_user_name;
    IF current_user_name IS NOT NULL THEN
        default_user_id := current_user_name;
    END IF;
    
    SELECT COUNT(*) INTO existing_profile_count
    FROM profiles
    WHERE user_id = default_user_id AND is_default = true;
    
    IF existing_profile_count = 0 THEN
        UPDATE profiles SET is_default = false WHERE user_id = default_user_id;
        
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
-- SECTION 5: PERMISSIONS
-- ============================================================================

-- Grant all privileges to current user
DO $$
BEGIN
    GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO CURRENT_USER;
    GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO CURRENT_USER;
    
    ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO CURRENT_USER;
    ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON SEQUENCES TO CURRENT_USER;
EXCEPTION
    WHEN OTHERS THEN
        NULL;
END $$;
