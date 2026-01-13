-- ============================================================================
-- Unified Identity Model Migration
-- ============================================================================
-- This migration aligns NeuronDesktop with the unified identity model
-- used across NeuronDB ecosystem components.
-- ============================================================================

-- Ensure UUID extension exists
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ============================================================================
-- Step 1: Create Organizations Table
-- ============================================================================
CREATE TABLE IF NOT EXISTS organizations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name TEXT NOT NULL,
    slug TEXT NOT NULL UNIQUE,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_organizations_slug ON organizations(slug);

-- ============================================================================
-- Step 2: Create Service Accounts Table
-- ============================================================================
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

-- ============================================================================
-- Step 3: Update Users Table (if not exists, create it)
-- ============================================================================
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    username TEXT NOT NULL UNIQUE,
    email TEXT,
    password_hash TEXT,
    is_admin BOOLEAN NOT NULL DEFAULT FALSE,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Add is_admin if it doesn't exist
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'users' AND column_name = 'is_admin'
    ) THEN
        ALTER TABLE users ADD COLUMN is_admin BOOLEAN NOT NULL DEFAULT FALSE;
    END IF;
END $$;

CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email) WHERE email IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_users_is_admin ON users(is_admin);

-- ============================================================================
-- Step 4: Update Profiles Table to Support Unified Model
-- ============================================================================
-- Add org_id column if it doesn't exist
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
    -- Check if user_id is TEXT and needs conversion
    IF EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'profiles' 
        AND column_name = 'user_id' 
        AND data_type = 'text'
    ) THEN
        -- Create a temporary UUID column
        ALTER TABLE profiles ADD COLUMN user_id_uuid UUID REFERENCES users(id) ON DELETE CASCADE;
        
        -- Migrate existing user_id values (create users if needed)
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
        
        -- Update user_id_uuid
        UPDATE profiles 
        SET user_id_uuid = users.id
        FROM users
        WHERE profiles.user_id = users.username;
        
        -- Drop old column and rename new one
        ALTER TABLE profiles DROP COLUMN user_id;
        ALTER TABLE profiles RENAME COLUMN user_id_uuid TO user_id;
    END IF;
END $$;

-- Ensure user_id is UUID type
DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'profiles' 
        AND column_name = 'user_id' 
        AND data_type != 'uuid'
    ) THEN
        -- This shouldn't happen after above migration, but handle it
        RAISE NOTICE 'user_id column type needs manual conversion';
    END IF;
END $$;

CREATE INDEX IF NOT EXISTS idx_profiles_org_id ON profiles(org_id) WHERE org_id IS NOT NULL;

-- ============================================================================
-- Step 5: Create Principals Table (for unified identity)
-- ============================================================================
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

-- ============================================================================
-- Step 6: Update API Keys Table to Support Unified Model
-- ============================================================================
-- Add principal_id and principal_type if they don't exist
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
        -- Get or create principal for user
        SELECT id INTO user_principal_id
        FROM principals
        WHERE type = 'user' AND name = user_record.user_id;
        
        IF user_principal_id IS NULL THEN
            -- Create principal
            INSERT INTO principals (type, name, created_at)
            VALUES ('user', user_record.user_id, NOW())
            RETURNING id INTO user_principal_id;
        END IF;
        
        -- Update api_keys
        UPDATE api_keys
        SET principal_id = user_principal_id,
            principal_type = 'user'
        WHERE user_id = user_record.user_id;
    END LOOP;
END $$;

CREATE INDEX IF NOT EXISTS idx_api_keys_principal_id ON api_keys(principal_id) WHERE principal_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_api_keys_principal_type ON api_keys(principal_type);
CREATE INDEX IF NOT EXISTS idx_api_keys_project_id ON api_keys(project_id) WHERE project_id IS NOT NULL;

-- ============================================================================
-- Step 7: Update Audit Log Table
-- ============================================================================
-- Add principal_id, project_id if they don't exist
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
        -- Get or create principal for user
        SELECT id INTO user_principal_id
        FROM principals
        WHERE type = 'user' AND name = user_record.user_id::TEXT;
        
        IF user_principal_id IS NULL THEN
            -- Create principal
            INSERT INTO principals (type, name, created_at)
            VALUES ('user', user_record.user_id::TEXT, NOW())
            RETURNING id INTO user_principal_id;
        END IF;
        
        -- Update audit_log
        UPDATE audit_log
        SET principal_id = user_principal_id
        WHERE user_id = user_record.user_id;
    END LOOP;
END $$;

CREATE INDEX IF NOT EXISTS idx_audit_log_principal_id ON audit_log(principal_id) WHERE principal_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_audit_log_project_id ON audit_log(project_id) WHERE project_id IS NOT NULL;

-- ============================================================================
-- Step 8: Create Policies Table (for unified permissions)
-- ============================================================================
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

-- ============================================================================
-- Step 9: Create Views for Convenience
-- ============================================================================

-- View: Unified identity view
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

-- View: API keys with principal information
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

-- ============================================================================
-- Step 10: Create Helper Functions
-- ============================================================================

-- Function: Get or create principal for user
CREATE OR REPLACE FUNCTION get_or_create_user_principal(user_uuid UUID, user_name TEXT)
RETURNS UUID AS $$
DECLARE
    principal_id UUID;
BEGIN
    -- Try to find existing principal
    SELECT id INTO principal_id
    FROM principals
    WHERE type = 'user' AND name = user_name;
    
    -- If not found, create it
    IF principal_id IS NULL THEN
        INSERT INTO principals (type, name, metadata, created_at)
        VALUES ('user', user_name, jsonb_build_object('user_id', user_uuid), NOW())
        RETURNING id INTO principal_id;
    END IF;
    
    RETURN principal_id;
END;
$$ LANGUAGE plpgsql;

-- Function: Get principal for API key
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







