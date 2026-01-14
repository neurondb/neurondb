/*-------------------------------------------------------------------------
 *
 * neuron-agent.sql
 *    Complete NeuronAgent Database Setup Script
 *
 * This script sets up everything needed for NeuronAgent:
 * - Database schema (tables, indexes, views, triggers)
 * - Management functions
 * - Pre-populated default data
 *
 * This script is idempotent and can be run multiple times safely.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/neuron-agent.sql
 *
 *-------------------------------------------------------------------------
 *
 * PREREQUISITES
 * =============
 *
 * - PostgreSQL 16 or later
 * - NeuronDB extension installed
 * - Database user with CREATE privileges
 *
 * USAGE
 * =====
 *
 * To run this setup script on a database:
 *
 *   psql -d your_database -f neuron-agent.sql
 *
 * Or from within psql:
 *
 *   \i neuron-agent.sql
 *
 *-------------------------------------------------------------------------
 */

-- ============================================================================
-- SECTION 1: EXTENSIONS
-- ============================================================================

-- Ensure required extensions are available
CREATE EXTENSION IF NOT EXISTS neurondb;
CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ============================================================================
-- SECTION 2: SCHEMA CREATION
-- ============================================================================

-- Schema: neurondb_agent
CREATE SCHEMA IF NOT EXISTS neurondb_agent;

-- ============================================================================
-- SECTION 3: CORE TABLES (Initial Schema)
-- ============================================================================

-- Agents table: Agent profiles and configurations
CREATE TABLE IF NOT EXISTS neurondb_agent.agents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT NOT NULL UNIQUE,
    description TEXT,
    system_prompt TEXT NOT NULL,
    model_name TEXT NOT NULL,  -- NeuronDB model identifier
    memory_table TEXT,          -- Optional per-agent memory table name
    enabled_tools TEXT[] DEFAULT '{}',
    config JSONB DEFAULT '{}',  -- temperature, max_tokens, top_p, etc.
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT valid_model_name CHECK (model_name ~ '^[a-zA-Z0-9_-]+$'),
    CONSTRAINT valid_memory_table CHECK (memory_table IS NULL OR memory_table ~ '^[a-z][a-z0-9_]*$')
);
COMMENT ON TABLE neurondb_agent.agents IS 'Agent profiles and configurations';

-- Sessions table: User conversation sessions
CREATE TABLE IF NOT EXISTS neurondb_agent.sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id UUID NOT NULL REFERENCES neurondb_agent.agents(id) ON DELETE CASCADE,
    external_user_id TEXT,  -- Optional external user identifier
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_activity_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT valid_external_user_id CHECK (external_user_id IS NULL OR length(external_user_id) > 0)
);
COMMENT ON TABLE neurondb_agent.sessions IS 'User conversation sessions';

-- Messages table: Conversation history
CREATE TABLE IF NOT EXISTS neurondb_agent.messages (
    id BIGSERIAL PRIMARY KEY,
    session_id UUID NOT NULL REFERENCES neurondb_agent.sessions(id) ON DELETE CASCADE,
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'system', 'tool')),
    content TEXT NOT NULL,
    tool_name TEXT,  -- NULL unless role = 'tool'
    tool_call_id TEXT,  -- For associating tool calls with results
    token_count INT,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT valid_tool_message CHECK (
        (role = 'tool' AND tool_name IS NOT NULL) OR
        (role != 'tool' AND tool_name IS NULL)
    )
);
COMMENT ON TABLE neurondb_agent.messages IS 'Conversation history';

-- Memory chunks table: Vector-embedded long-term memory
CREATE TABLE IF NOT EXISTS neurondb_agent.memory_chunks (
    id BIGSERIAL PRIMARY KEY,
    agent_id UUID NOT NULL REFERENCES neurondb_agent.agents(id) ON DELETE CASCADE,
    session_id UUID REFERENCES neurondb_agent.sessions(id) ON DELETE SET NULL,
    message_id BIGINT REFERENCES neurondb_agent.messages(id) ON DELETE SET NULL,
    content TEXT NOT NULL,
    embedding vector(768),  -- NeuronDB vector type, configurable dimension
    importance_score REAL DEFAULT 0.5 CHECK (importance_score >= 0 AND importance_score <= 1),
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT valid_embedding CHECK (embedding IS NOT NULL)
);
COMMENT ON TABLE neurondb_agent.memory_chunks IS 'Vector-embedded long-term memory';

-- Tools table: Tool registry
CREATE TABLE IF NOT EXISTS neurondb_agent.tools (
    name TEXT PRIMARY KEY,
    description TEXT NOT NULL,
    arg_schema JSONB NOT NULL,  -- JSON Schema for arguments
    handler_type TEXT NOT NULL CHECK (handler_type IN ('sql', 'http', 'code', 'shell', 'queue', 'ml', 'vector', 'rag', 'analytics', 'hybrid_search', 'reranking')),
    handler_config JSONB DEFAULT '{}',
    enabled BOOLEAN DEFAULT true,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT valid_arg_schema CHECK (jsonb_typeof(arg_schema) = 'object')
);
COMMENT ON TABLE neurondb_agent.tools IS 'Tool registry';

-- Jobs table: Background job queue
CREATE TABLE IF NOT EXISTS neurondb_agent.jobs (
    id BIGSERIAL PRIMARY KEY,
    agent_id UUID REFERENCES neurondb_agent.agents(id) ON DELETE SET NULL,
    session_id UUID REFERENCES neurondb_agent.sessions(id) ON DELETE SET NULL,
    type TEXT NOT NULL CHECK (type IN ('http_call', 'sql_task', 'shell_task', 'custom')),
    status TEXT NOT NULL DEFAULT 'queued' CHECK (status IN ('queued', 'running', 'done', 'failed', 'cancelled')),
    priority INT DEFAULT 0,
    payload JSONB NOT NULL,
    result JSONB,
    error_message TEXT,
    retry_count INT DEFAULT 0,
    max_retries INT DEFAULT 3,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ
);
COMMENT ON TABLE neurondb_agent.jobs IS 'Background job queue';

-- API keys table: Authentication
CREATE TABLE IF NOT EXISTS neurondb_agent.api_keys (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    key_hash TEXT NOT NULL UNIQUE,  -- Bcrypt hash of API key
    key_prefix TEXT NOT NULL,  -- First 8 chars for identification
    organization_id TEXT,
    user_id TEXT,
    rate_limit_per_minute INT DEFAULT 60,
    roles TEXT[] DEFAULT '{user}',
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_used_at TIMESTAMPTZ,
    expires_at TIMESTAMPTZ,
    CONSTRAINT valid_roles CHECK (array_length(roles, 1) > 0)
);
COMMENT ON TABLE neurondb_agent.api_keys IS 'API key authentication';

-- ============================================================================
-- SECTION 4: PERFORMANCE INDEXES
-- ============================================================================

-- Performance indexes
CREATE INDEX IF NOT EXISTS idx_sessions_agent_id ON neurondb_agent.sessions(agent_id);
CREATE INDEX IF NOT EXISTS idx_sessions_last_activity ON neurondb_agent.sessions(last_activity_at);
CREATE INDEX IF NOT EXISTS idx_messages_session_id ON neurondb_agent.messages(session_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_messages_session_role ON neurondb_agent.messages(session_id, role);
CREATE INDEX IF NOT EXISTS idx_memory_chunks_agent_id ON neurondb_agent.memory_chunks(agent_id);
CREATE INDEX IF NOT EXISTS idx_memory_chunks_session_id ON neurondb_agent.memory_chunks(session_id);
CREATE INDEX IF NOT EXISTS idx_jobs_status_created ON neurondb_agent.jobs(status, created_at) WHERE status IN ('queued', 'running');
CREATE INDEX IF NOT EXISTS idx_jobs_agent_session ON neurondb_agent.jobs(agent_id, session_id);
CREATE INDEX IF NOT EXISTS idx_api_keys_prefix ON neurondb_agent.api_keys(key_prefix);

-- HNSW index on memory chunks embedding (NeuronDB)
CREATE INDEX IF NOT EXISTS idx_memory_chunks_embedding_hnsw ON neurondb_agent.memory_chunks 
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- ============================================================================
-- SECTION 5: TRIGGERS
-- ============================================================================

-- Triggers for updated_at
CREATE OR REPLACE FUNCTION neurondb_agent.update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER agents_updated_at BEFORE UPDATE ON neurondb_agent.agents
    FOR EACH ROW EXECUTE FUNCTION neurondb_agent.update_updated_at();

CREATE TRIGGER tools_updated_at BEFORE UPDATE ON neurondb_agent.tools
    FOR EACH ROW EXECUTE FUNCTION neurondb_agent.update_updated_at();

CREATE TRIGGER jobs_updated_at BEFORE UPDATE ON neurondb_agent.jobs
    FOR EACH ROW EXECUTE FUNCTION neurondb_agent.update_updated_at();

-- Trigger for session last_activity_at
CREATE OR REPLACE FUNCTION neurondb_agent.update_session_activity()
RETURNS TRIGGER AS $$
BEGIN
    UPDATE neurondb_agent.sessions
    SET last_activity_at = NOW()
    WHERE id = NEW.session_id;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER messages_session_activity AFTER INSERT ON neurondb_agent.messages
    FOR EACH ROW EXECUTE FUNCTION neurondb_agent.update_session_activity();

-- ============================================================================
-- SECTION 6: PRINCIPALS AND PERMISSIONS
-- ============================================================================

-- Principals table: Represents entities that can have permissions (users, orgs, agents, tools, datasets)
CREATE TABLE IF NOT EXISTS neurondb_agent.principals (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    type TEXT NOT NULL CHECK (type IN ('user', 'org', 'agent', 'tool', 'dataset')),
    name TEXT NOT NULL,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT unique_principal_name_per_type UNIQUE (type, name)
);
COMMENT ON TABLE neurondb_agent.principals IS 'Entities that can have permissions (users, orgs, agents, tools, datasets)';

-- Link API keys to principals
ALTER TABLE neurondb_agent.api_keys 
    ADD COLUMN IF NOT EXISTS principal_id UUID REFERENCES neurondb_agent.principals(id) ON DELETE SET NULL;

CREATE INDEX IF NOT EXISTS idx_api_keys_principal_id ON neurondb_agent.api_keys(principal_id);

-- Policies table: Defines permissions for principals
CREATE TABLE IF NOT EXISTS neurondb_agent.policies (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    principal_id UUID NOT NULL REFERENCES neurondb_agent.principals(id) ON DELETE CASCADE,
    resource_type TEXT NOT NULL,  -- e.g., 'agent', 'tool', 'dataset', 'schema', 'table'
    resource_id TEXT,  -- NULL for wildcard policies
    permissions TEXT[] NOT NULL DEFAULT '{}',  -- e.g., ['read', 'write', 'execute']
    conditions JSONB DEFAULT '{}',  -- ABAC conditions
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT valid_permissions CHECK (array_length(permissions, 1) > 0)
);
COMMENT ON TABLE neurondb_agent.policies IS 'Defines permissions for principals';

CREATE INDEX IF NOT EXISTS idx_policies_principal_id ON neurondb_agent.policies(principal_id);
CREATE INDEX IF NOT EXISTS idx_policies_resource ON neurondb_agent.policies(resource_type, resource_id);

-- Tool permissions per agent
CREATE TABLE IF NOT EXISTS neurondb_agent.tool_permissions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id UUID NOT NULL REFERENCES neurondb_agent.agents(id) ON DELETE CASCADE,
    tool_name TEXT NOT NULL REFERENCES neurondb_agent.tools(name) ON DELETE CASCADE,
    allowed BOOLEAN NOT NULL DEFAULT true,
    conditions JSONB DEFAULT '{}',  -- Additional conditions for tool execution
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT unique_agent_tool_permission UNIQUE (agent_id, tool_name)
);
COMMENT ON TABLE neurondb_agent.tool_permissions IS 'Tool permissions per agent';

CREATE INDEX IF NOT EXISTS idx_tool_permissions_agent_id ON neurondb_agent.tool_permissions(agent_id);
CREATE INDEX IF NOT EXISTS idx_tool_permissions_tool_name ON neurondb_agent.tool_permissions(tool_name);

-- Session-scoped tool permissions (overrides agent-level permissions)
CREATE TABLE IF NOT EXISTS neurondb_agent.session_tool_permissions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID NOT NULL REFERENCES neurondb_agent.sessions(id) ON DELETE CASCADE,
    tool_name TEXT NOT NULL REFERENCES neurondb_agent.tools(name) ON DELETE CASCADE,
    allowed BOOLEAN NOT NULL DEFAULT true,
    conditions JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT unique_session_tool_permission UNIQUE (session_id, tool_name)
);
COMMENT ON TABLE neurondb_agent.session_tool_permissions IS 'Session-scoped tool permissions';

CREATE INDEX IF NOT EXISTS idx_session_tool_permissions_session_id ON neurondb_agent.session_tool_permissions(session_id);
CREATE INDEX IF NOT EXISTS idx_session_tool_permissions_tool_name ON neurondb_agent.session_tool_permissions(tool_name);

-- Data permissions: schema, table, row filters, column masks
CREATE TABLE IF NOT EXISTS neurondb_agent.data_permissions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    principal_id UUID NOT NULL REFERENCES neurondb_agent.principals(id) ON DELETE CASCADE,
    schema_name TEXT,
    table_name TEXT,
    row_filter TEXT,  -- SQL WHERE clause for row-level filtering
    column_mask JSONB DEFAULT '{}',  -- Map of column names to masking rules
    permissions TEXT[] NOT NULL DEFAULT '{}',  -- e.g., ['select', 'insert', 'update', 'delete']
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT valid_data_permissions CHECK (array_length(permissions, 1) > 0),
    CONSTRAINT valid_data_permission_target CHECK (
        (schema_name IS NOT NULL) OR (table_name IS NOT NULL)
    )
);
COMMENT ON TABLE neurondb_agent.data_permissions IS 'Data permissions with row filters and column masks';

CREATE INDEX IF NOT EXISTS idx_data_permissions_principal_id ON neurondb_agent.data_permissions(principal_id);
CREATE INDEX IF NOT EXISTS idx_data_permissions_schema_table ON neurondb_agent.data_permissions(schema_name, table_name);

-- Audit log table: Comprehensive audit logging for tool calls and SQL statements
CREATE TABLE IF NOT EXISTS neurondb_agent.audit_log (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    principal_id UUID REFERENCES neurondb_agent.principals(id) ON DELETE SET NULL,
    api_key_id UUID REFERENCES neurondb_agent.api_keys(id) ON DELETE SET NULL,
    agent_id UUID REFERENCES neurondb_agent.agents(id) ON DELETE SET NULL,
    session_id UUID REFERENCES neurondb_agent.sessions(id) ON DELETE SET NULL,
    action TEXT NOT NULL,  -- e.g., 'tool_call', 'sql_execute', 'agent_execute'
    resource_type TEXT NOT NULL,  -- e.g., 'tool', 'sql', 'agent'
    resource_id TEXT,  -- e.g., tool name, SQL query hash, agent ID
    inputs_hash TEXT,  -- SHA-256 hash of inputs
    outputs_hash TEXT,  -- SHA-256 hash of outputs
    inputs JSONB,  -- Optional: actual inputs (may be truncated for privacy)
    outputs JSONB,  -- Optional: actual outputs (may be truncated for privacy)
    metadata JSONB DEFAULT '{}',  -- Additional context
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
COMMENT ON TABLE neurondb_agent.audit_log IS 'Comprehensive audit logging for tool calls and SQL statements';

CREATE INDEX IF NOT EXISTS idx_audit_log_timestamp ON neurondb_agent.audit_log(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_audit_log_principal_id ON neurondb_agent.audit_log(principal_id, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_audit_log_api_key_id ON neurondb_agent.audit_log(api_key_id, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_audit_log_agent_id ON neurondb_agent.audit_log(agent_id, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_audit_log_session_id ON neurondb_agent.audit_log(session_id, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_audit_log_action ON neurondb_agent.audit_log(action, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_audit_log_resource ON neurondb_agent.audit_log(resource_type, resource_id, timestamp DESC);

-- ============================================================================
-- SECTION 7: UNIFIED IDENTITY MODEL
-- ============================================================================

-- Add Organizations Table
CREATE TABLE IF NOT EXISTS neurondb_agent.organizations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT NOT NULL,
    slug TEXT NOT NULL UNIQUE,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
COMMENT ON TABLE neurondb_agent.organizations IS 'Organizations for multi-tenant support';

CREATE INDEX IF NOT EXISTS idx_organizations_slug ON neurondb_agent.organizations(slug);

-- Add Service Accounts Table
CREATE TABLE IF NOT EXISTS neurondb_agent.service_accounts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT NOT NULL,
    org_id UUID REFERENCES neurondb_agent.organizations(id) ON DELETE CASCADE,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
COMMENT ON TABLE neurondb_agent.service_accounts IS 'Service accounts for programmatic access';

CREATE INDEX IF NOT EXISTS idx_service_accounts_org_id ON neurondb_agent.service_accounts(org_id);

-- Update Principals Table to Include Service Account Type
DO $$
BEGIN
    -- Drop existing constraint if it exists
    ALTER TABLE neurondb_agent.principals 
    DROP CONSTRAINT IF EXISTS principals_type_check;
    
    -- Add new constraint with service_account
    ALTER TABLE neurondb_agent.principals 
    ADD CONSTRAINT principals_type_check 
    CHECK (type IN ('user', 'org', 'agent', 'tool', 'dataset', 'service_account'));
END $$;

-- Update API Keys Table to Support Unified Model
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_schema = 'neurondb_agent'
        AND table_name = 'api_keys' 
        AND column_name = 'principal_type'
    ) THEN
        ALTER TABLE neurondb_agent.api_keys 
        ADD COLUMN principal_type TEXT CHECK (principal_type IN ('user', 'org', 'service_account'));
    END IF;
END $$;

-- Add Project Reference to API Keys
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_schema = 'neurondb_agent'
        AND table_name = 'api_keys' 
        AND column_name = 'project_id'
    ) THEN
        ALTER TABLE neurondb_agent.api_keys 
        ADD COLUMN project_id TEXT;
    END IF;
END $$;

CREATE INDEX IF NOT EXISTS idx_api_keys_project_id ON neurondb_agent.api_keys(project_id) WHERE project_id IS NOT NULL;

-- Update Audit Log to Include Project ID
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_schema = 'neurondb_agent'
        AND table_name = 'audit_log' 
        AND column_name = 'project_id'
    ) THEN
        ALTER TABLE neurondb_agent.audit_log 
        ADD COLUMN project_id TEXT;
    END IF;
END $$;

CREATE INDEX IF NOT EXISTS idx_audit_log_project_id ON neurondb_agent.audit_log(project_id) WHERE project_id IS NOT NULL;

-- View: API keys with principal information
CREATE OR REPLACE VIEW neurondb_agent.api_keys_with_principals AS
SELECT 
    ak.id,
    ak.key_prefix,
    ak.principal_id,
    ak.principal_type,
    ak.project_id,
    ak.rate_limit_per_minute,
    ak.roles,
    ak.last_used_at,
    ak.expires_at,
    ak.created_at,
    p.name AS principal_name,
    p.metadata AS principal_metadata
FROM neurondb_agent.api_keys ak
LEFT JOIN neurondb_agent.principals p ON ak.principal_id = p.id;

-- View: Unified principals view
CREATE OR REPLACE VIEW neurondb_agent.unified_principals AS
SELECT 
    'user' AS principal_type,
    id AS principal_id,
    name AS principal_name,
    metadata,
    created_at,
    updated_at
FROM neurondb_agent.principals
WHERE type = 'user'
UNION ALL
SELECT 
    'org' AS principal_type,
    id AS principal_id,
    name AS principal_name,
    metadata,
    created_at,
    updated_at
FROM neurondb_agent.principals
WHERE type = 'org'
UNION ALL
SELECT 
    'service_account' AS principal_type,
    sa.id AS principal_id,
    sa.name AS principal_name,
    sa.metadata,
    sa.created_at,
    sa.updated_at
FROM neurondb_agent.service_accounts sa;

-- Function: Get or create principal for user
CREATE OR REPLACE FUNCTION neurondb_agent.get_or_create_user_principal(user_id TEXT)
RETURNS UUID AS $$
DECLARE
    principal_id UUID;
BEGIN
    SELECT id INTO principal_id
    FROM neurondb_agent.principals
    WHERE type = 'user' AND name = user_id;
    
    IF principal_id IS NULL THEN
        INSERT INTO neurondb_agent.principals (type, name, metadata, created_at)
        VALUES ('user', user_id, jsonb_build_object('user_id', user_id), NOW())
        RETURNING id INTO principal_id;
    END IF;
    
    RETURN principal_id;
END;
$$ LANGUAGE plpgsql;

-- Function: Get principal for API key
CREATE OR REPLACE FUNCTION neurondb_agent.get_principal_for_api_key(api_key_prefix TEXT)
RETURNS TABLE (
    principal_id UUID,
    principal_type TEXT,
    principal_name TEXT,
    project_id TEXT,
    roles TEXT[]
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        ak.principal_id,
        ak.principal_type,
        p.name,
        ak.project_id,
        ak.roles
    FROM neurondb_agent.api_keys ak
    LEFT JOIN neurondb_agent.principals p ON ak.principal_id = p.id
    WHERE ak.key_prefix = api_key_prefix;
END;
$$ LANGUAGE plpgsql;

-- Function: Check if principal has permission
CREATE OR REPLACE FUNCTION neurondb_agent.check_principal_permission(
    p_principal_id UUID,
    p_resource_type TEXT,
    p_resource_id TEXT,
    p_permission TEXT
)
RETURNS BOOLEAN AS $$
DECLARE
    has_permission BOOLEAN;
BEGIN
    SELECT EXISTS (
        SELECT 1
        FROM neurondb_agent.policies
        WHERE principal_id = p_principal_id
        AND resource_type = p_resource_type
        AND (resource_id = p_resource_id OR resource_id IS NULL)
        AND p_permission = ANY(permissions)
    ) INTO has_permission;
    
    RETURN COALESCE(has_permission, FALSE);
END;
$$ LANGUAGE plpgsql;


-- ============================================================================
-- SECTION 8: WORKFLOW ENGINE
-- ============================================================================
CREATE TABLE IF NOT EXISTS neurondb_agent.workflows (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT NOT NULL UNIQUE,
    dag_definition JSONB NOT NULL,  -- DAG structure: nodes, edges, step definitions
    status TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'paused', 'archived')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_workflows_status ON neurondb_agent.workflows(status);
CREATE INDEX IF NOT EXISTS idx_workflows_created_at ON neurondb_agent.workflows(created_at DESC);

-- Workflow steps table: Individual steps in a workflow
CREATE TABLE IF NOT EXISTS neurondb_agent.workflow_steps (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workflow_id UUID NOT NULL REFERENCES neurondb_agent.workflows(id) ON DELETE CASCADE,
    step_name TEXT NOT NULL,
    step_type TEXT NOT NULL CHECK (step_type IN ('agent', 'tool', 'approval', 'http', 'sql', 'custom')),
    inputs JSONB DEFAULT '{}',
    outputs JSONB DEFAULT '{}',
    dependencies TEXT[] DEFAULT '{}',  -- Array of step names this step depends on
    retry_config JSONB DEFAULT '{}',  -- {max_retries, backoff_multiplier, initial_delay, max_delay}
    idempotency_key TEXT,  -- Optional idempotency key for this step
    compensation_step_id UUID REFERENCES neurondb_agent.workflow_steps(id) ON DELETE SET NULL,  -- Rollback step
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT unique_workflow_step_name UNIQUE (workflow_id, step_name)
);

CREATE INDEX IF NOT EXISTS idx_workflow_steps_workflow_id ON neurondb_agent.workflow_steps(workflow_id);
CREATE INDEX IF NOT EXISTS idx_workflow_steps_idempotency_key ON neurondb_agent.workflow_steps(idempotency_key) WHERE idempotency_key IS NOT NULL;

-- Workflow executions table: Execution instances of workflows
CREATE TABLE IF NOT EXISTS neurondb_agent.workflow_executions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workflow_id UUID NOT NULL REFERENCES neurondb_agent.workflows(id) ON DELETE CASCADE,
    status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'running', 'completed', 'failed', 'cancelled')),
    trigger_type TEXT NOT NULL CHECK (trigger_type IN ('manual', 'schedule', 'webhook', 'db_notify', 'queue')),
    trigger_data JSONB DEFAULT '{}',
    inputs JSONB DEFAULT '{}',
    outputs JSONB DEFAULT '{}',
    error_message TEXT,
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_workflow_executions_workflow_id ON neurondb_agent.workflow_executions(workflow_id);
CREATE INDEX IF NOT EXISTS idx_workflow_executions_status ON neurondb_agent.workflow_executions(status);
CREATE INDEX IF NOT EXISTS idx_workflow_executions_created_at ON neurondb_agent.workflow_executions(created_at DESC);

-- Workflow step executions table: Execution instances of individual steps
CREATE TABLE IF NOT EXISTS neurondb_agent.workflow_step_executions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workflow_execution_id UUID NOT NULL REFERENCES neurondb_agent.workflow_executions(id) ON DELETE CASCADE,
    workflow_step_id UUID NOT NULL REFERENCES neurondb_agent.workflow_steps(id) ON DELETE CASCADE,
    status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'running', 'completed', 'failed', 'skipped', 'compensated')),
    inputs JSONB DEFAULT '{}',
    outputs JSONB DEFAULT '{}',
    error_message TEXT,
    retry_count INT DEFAULT 0,
    idempotency_key TEXT,
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT unique_execution_step_idempotency UNIQUE (idempotency_key) WHERE idempotency_key IS NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_workflow_step_executions_execution_id ON neurondb_agent.workflow_step_executions(workflow_execution_id);
CREATE INDEX IF NOT EXISTS idx_workflow_step_executions_step_id ON neurondb_agent.workflow_step_executions(workflow_step_id);
CREATE INDEX IF NOT EXISTS idx_workflow_step_executions_status ON neurondb_agent.workflow_step_executions(status);
CREATE INDEX IF NOT EXISTS idx_workflow_step_executions_idempotency_key ON neurondb_agent.workflow_step_executions(idempotency_key) WHERE idempotency_key IS NOT NULL;

-- Workflow schedules table: Schedule definitions for workflows
CREATE TABLE IF NOT EXISTS neurondb_agent.workflow_schedules (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workflow_id UUID NOT NULL REFERENCES neurondb_agent.workflows(id) ON DELETE CASCADE,
    cron_expression TEXT NOT NULL,  -- Cron expression for scheduling
    timezone TEXT DEFAULT 'UTC',
    enabled BOOLEAN DEFAULT true,
    next_run_at TIMESTAMPTZ,
    last_run_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT unique_workflow_schedule UNIQUE (workflow_id)
);

CREATE INDEX IF NOT EXISTS idx_workflow_schedules_workflow_id ON neurondb_agent.workflow_schedules(workflow_id);
CREATE INDEX IF NOT EXISTS idx_workflow_schedules_next_run_at ON neurondb_agent.workflow_schedules(next_run_at) WHERE enabled = true AND next_run_at IS NOT NULL;



-- ============================================================================
-- SECTION 9: ASYNC TASKS
-- ============================================================================
/* Async tasks table for tracking long-running agent tasks */
CREATE TABLE IF NOT EXISTS neurondb_agent.async_tasks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID NOT NULL REFERENCES neurondb_agent.sessions(id) ON DELETE CASCADE,
    agent_id UUID NOT NULL REFERENCES neurondb_agent.agents(id) ON DELETE CASCADE,
    task_type TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'running', 'completed', 'failed', 'cancelled')),
    priority INT NOT NULL DEFAULT 0,
    input JSONB NOT NULL DEFAULT '{}',
    result JSONB,
    error_message TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    metadata JSONB DEFAULT '{}'
);

/* Indexes for efficient querying */
CREATE INDEX IF NOT EXISTS idx_async_tasks_status ON neurondb_agent.async_tasks(status, priority DESC, created_at);
CREATE INDEX IF NOT EXISTS idx_async_tasks_session ON neurondb_agent.async_tasks(session_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_async_tasks_agent ON neurondb_agent.async_tasks(agent_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_async_tasks_type ON neurondb_agent.async_tasks(task_type, status);

/* Task notifications table for tracking notification delivery */
CREATE TABLE IF NOT EXISTS neurondb_agent.task_notifications (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    task_id UUID NOT NULL REFERENCES neurondb_agent.async_tasks(id) ON DELETE CASCADE,
    notification_type TEXT NOT NULL CHECK (notification_type IN ('completion', 'failure', 'progress', 'milestone')),
    channel TEXT NOT NULL CHECK (channel IN ('email', 'webhook', 'push')),
    recipient TEXT NOT NULL,
    sent_at TIMESTAMPTZ,
    status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'sent', 'failed', 'delivered')),
    error_message TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

/* Indexes for notification queries */
CREATE INDEX IF NOT EXISTS idx_task_notifications_task ON neurondb_agent.task_notifications(task_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_task_notifications_status ON neurondb_agent.task_notifications(status, created_at) WHERE status IN ('pending', 'failed');
CREATE INDEX IF NOT EXISTS idx_task_notifications_type ON neurondb_agent.task_notifications(notification_type, channel);

/* Comments */
COMMENT ON TABLE neurondb_agent.async_tasks IS 'Tracks asynchronous agent tasks with status, results, and metadata. Enables long-running tasks to execute in background with status tracking.';
COMMENT ON COLUMN neurondb_agent.async_tasks.task_type IS 'Type of task (e.g., "agent_execution", "data_processing", "code_execution")';
COMMENT ON COLUMN neurondb_agent.async_tasks.status IS 'Current status: pending, running, completed, failed, or cancelled';
COMMENT ON COLUMN neurondb_agent.async_tasks.priority IS 'Task priority (higher numbers = higher priority)';
COMMENT ON COLUMN neurondb_agent.async_tasks.input IS 'Task input parameters as JSON';
COMMENT ON COLUMN neurondb_agent.async_tasks.result IS 'Task result/output as JSON (null until completion)';
COMMENT ON COLUMN neurondb_agent.async_tasks.error_message IS 'Error message if task failed';

COMMENT ON TABLE neurondb_agent.task_notifications IS 'Tracks notifications sent for task events (completion, failure, progress, milestones)';
COMMENT ON COLUMN neurondb_agent.task_notifications.notification_type IS 'Type of notification: completion, failure, progress, or milestone';
COMMENT ON COLUMN neurondb_agent.task_notifications.channel IS 'Delivery channel: email, webhook, or push';
COMMENT ON COLUMN neurondb_agent.task_notifications.recipient IS 'Recipient identifier (email address, webhook URL, or push token)';


-- ============================================================================
-- SECTION 10: HIERARCHICAL MEMORY
-- ============================================================================
-- Hierarchical Memory System
-- Implements Short-Term, Mid-Term, and Long-Term Personal Memory tiers
-- Provides automatic memory promotion and expiration

-- Short-Term Memory (STM) - conversation-level, 1 hour TTL
CREATE TABLE IF NOT EXISTS neurondb_agent.memory_stm (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id UUID NOT NULL REFERENCES neurondb_agent.agents(id) ON DELETE CASCADE,
    session_id UUID REFERENCES neurondb_agent.sessions(id) ON DELETE CASCADE,
    content TEXT NOT NULL,
    embedding neurondb_vector(768),
    importance_score FLOAT NOT NULL DEFAULT 0.5,
    access_count INT NOT NULL DEFAULT 0,
    last_accessed_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    expires_at TIMESTAMPTZ NOT NULL DEFAULT (NOW() + INTERVAL '1 hour')
);

-- Mid-Term Memory (MTM) - topic summaries, 7 days TTL
CREATE TABLE IF NOT EXISTS neurondb_agent.memory_mtm (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id UUID NOT NULL REFERENCES neurondb_agent.agents(id) ON DELETE CASCADE,
    topic TEXT NOT NULL,
    content TEXT NOT NULL,
    embedding neurondb_vector(768),
    importance_score FLOAT NOT NULL DEFAULT 0.6,
    source_stm_ids UUID[],
    pattern_count INT NOT NULL DEFAULT 1,
    last_reinforced_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    expires_at TIMESTAMPTZ NOT NULL DEFAULT (NOW() + INTERVAL '7 days')
);

-- Long-Term Personal Memory (LPM) - preferences, permanent
CREATE TABLE IF NOT EXISTS neurondb_agent.memory_lpm (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id UUID NOT NULL REFERENCES neurondb_agent.agents(id) ON DELETE CASCADE,
    user_id UUID,
    category TEXT NOT NULL,
    content TEXT NOT NULL,
    embedding neurondb_vector(768),
    importance_score FLOAT NOT NULL DEFAULT 0.8,
    source_mtm_ids UUID[],
    confidence FLOAT NOT NULL DEFAULT 0.7,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Memory transitions tracking
CREATE TABLE IF NOT EXISTS neurondb_agent.memory_transitions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id UUID NOT NULL REFERENCES neurondb_agent.agents(id) ON DELETE CASCADE,
    from_tier TEXT NOT NULL CHECK (from_tier IN ('stm', 'mtm')),
    to_tier TEXT NOT NULL CHECK (to_tier IN ('mtm', 'lpm')),
    source_id UUID NOT NULL,
    target_id UUID NOT NULL,
    reason TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Indexes for STM
CREATE INDEX IF NOT EXISTS idx_memory_stm_agent_id ON neurondb_agent.memory_stm(agent_id);
CREATE INDEX IF NOT EXISTS idx_memory_stm_session_id ON neurondb_agent.memory_stm(session_id);
CREATE INDEX IF NOT EXISTS idx_memory_stm_expires_at ON neurondb_agent.memory_stm(expires_at) WHERE expires_at IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_memory_stm_importance ON neurondb_agent.memory_stm(importance_score DESC);
CREATE INDEX IF NOT EXISTS idx_memory_stm_embedding ON neurondb_agent.memory_stm USING hnsw (embedding neurondb_vector_cosine_ops) WITH (m = 16, ef_construction = 64);

-- Indexes for MTM
CREATE INDEX IF NOT EXISTS idx_memory_mtm_agent_id ON neurondb_agent.memory_mtm(agent_id);
CREATE INDEX IF NOT EXISTS idx_memory_mtm_topic ON neurondb_agent.memory_mtm(topic);
CREATE INDEX IF NOT EXISTS idx_memory_mtm_expires_at ON neurondb_agent.memory_mtm(expires_at) WHERE expires_at IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_memory_mtm_importance ON neurondb_agent.memory_mtm(importance_score DESC);
CREATE INDEX IF NOT EXISTS idx_memory_mtm_embedding ON neurondb_agent.memory_mtm USING hnsw (embedding neurondb_vector_cosine_ops) WITH (m = 16, ef_construction = 64);

-- Indexes for LPM
CREATE INDEX IF NOT EXISTS idx_memory_lpm_agent_id ON neurondb_agent.memory_lpm(agent_id);
CREATE INDEX IF NOT EXISTS idx_memory_lpm_user_id ON neurondb_agent.memory_lpm(user_id) WHERE user_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_memory_lpm_category ON neurondb_agent.memory_lpm(category);
CREATE INDEX IF NOT EXISTS idx_memory_lpm_importance ON neurondb_agent.memory_lpm(importance_score DESC);
CREATE INDEX IF NOT EXISTS idx_memory_lpm_embedding ON neurondb_agent.memory_lpm USING hnsw (embedding neurondb_vector_cosine_ops) WITH (m = 16, ef_construction = 64);

-- Indexes for transitions
CREATE INDEX IF NOT EXISTS idx_memory_transitions_agent_id ON neurondb_agent.memory_transitions(agent_id);
CREATE INDEX IF NOT EXISTS idx_memory_transitions_source_id ON neurondb_agent.memory_transitions(source_id);
CREATE INDEX IF NOT EXISTS idx_memory_transitions_created_at ON neurondb_agent.memory_transitions(created_at DESC);

COMMENT ON TABLE neurondb_agent.memory_stm IS 'Short-Term Memory: Real-time conversation data with 1-hour TTL. Automatically expires and promotes to MTM based on importance and patterns.';
COMMENT ON TABLE neurondb_agent.memory_mtm IS 'Mid-Term Memory: Topic summaries and recurring patterns with 7-day TTL. Promoted from STM when patterns detected.';
COMMENT ON TABLE neurondb_agent.memory_lpm IS 'Long-Term Personal Memory: User preferences and agent knowledge. Permanent storage for high-importance information.';
COMMENT ON TABLE neurondb_agent.memory_transitions IS 'Tracks memory promotions between tiers for analytics and debugging.';

COMMENT ON COLUMN neurondb_agent.memory_stm.expires_at IS 'Automatic expiration timestamp. STM expires after 1 hour unless accessed or promoted.';
COMMENT ON COLUMN neurondb_agent.memory_mtm.pattern_count IS 'Number of times this pattern has been observed. Higher count increases promotion likelihood.';
COMMENT ON COLUMN neurondb_agent.memory_lpm.confidence IS 'Confidence score for this memory (0-1). Higher confidence indicates more reliable information.';


-- ============================================================================
-- SECTION 11: BUDGET MANAGEMENT
-- ============================================================================
-- Budget management schema
CREATE TABLE IF NOT EXISTS neurondb_agent.agent_budgets (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id UUID NOT NULL REFERENCES neurondb_agent.agents(id) ON DELETE CASCADE,
    budget_amount REAL NOT NULL CHECK (budget_amount >= 0),
    period_type TEXT NOT NULL CHECK (period_type IN ('daily', 'weekly', 'monthly', 'yearly', 'total')),
    start_date TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    end_date TIMESTAMPTZ,
    is_active BOOLEAN DEFAULT true,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(agent_id, period_type) WHERE is_active = true
);

CREATE INDEX IF NOT EXISTS idx_agent_budgets_agent ON neurondb_agent.agent_budgets(agent_id, is_active);
CREATE INDEX IF NOT EXISTS idx_agent_budgets_active ON neurondb_agent.agent_budgets(agent_id) WHERE is_active = true;












-- ============================================================================
-- SECTION 12: WEBHOOKS
-- ============================================================================
-- Webhooks schema for event notifications
CREATE TABLE IF NOT EXISTS neurondb_agent.webhooks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    url TEXT NOT NULL,
    events TEXT[] NOT NULL,
    secret TEXT,
    enabled BOOLEAN DEFAULT true,
    timeout_seconds INT DEFAULT 30,
    retry_count INT DEFAULT 3,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_webhooks_enabled ON neurondb_agent.webhooks(enabled) WHERE enabled = true;
CREATE INDEX IF NOT EXISTS idx_webhooks_events ON neurondb_agent.webhooks USING GIN(events);

-- Webhook deliveries for tracking webhook execution
CREATE TABLE IF NOT EXISTS neurondb_agent.webhook_deliveries (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    webhook_id UUID NOT NULL REFERENCES neurondb_agent.webhooks(id) ON DELETE CASCADE,
    event_type TEXT NOT NULL,
    payload JSONB NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('pending', 'success', 'failed', 'retrying')),
    status_code INT,
    response_body TEXT,
    error_message TEXT,
    attempt_count INT DEFAULT 0,
    next_retry_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    delivered_at TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_webhook_deliveries_webhook ON neurondb_agent.webhook_deliveries(webhook_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_webhook_deliveries_status ON neurondb_agent.webhook_deliveries(status, next_retry_at) WHERE status IN ('pending', 'retrying');












-- ============================================================================
-- SECTION 13: HUMAN-IN-THE-LOOP
-- ============================================================================
-- Human-in-the-loop schema
CREATE TABLE IF NOT EXISTS neurondb_agent.approval_requests (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id UUID REFERENCES neurondb_agent.agents(id) ON DELETE CASCADE,
    session_id UUID REFERENCES neurondb_agent.sessions(id) ON DELETE CASCADE,
    request_type TEXT NOT NULL CHECK (request_type IN ('tool_execution', 'agent_action', 'budget_exceeded', 'sensitive_operation')),
    action_description TEXT NOT NULL,
    payload JSONB NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'approved', 'rejected', 'expired')),
    requested_by TEXT,
    approved_by TEXT,
    rejection_reason TEXT,
    expires_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    resolved_at TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_approval_requests_status ON neurondb_agent.approval_requests(status, created_at) WHERE status = 'pending';
CREATE INDEX IF NOT EXISTS idx_approval_requests_agent ON neurondb_agent.approval_requests(agent_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_approval_requests_session ON neurondb_agent.approval_requests(session_id, created_at DESC);

-- User feedback table
CREATE TABLE IF NOT EXISTS neurondb_agent.user_feedback (
    id BIGSERIAL PRIMARY KEY,
    agent_id UUID REFERENCES neurondb_agent.agents(id) ON DELETE SET NULL,
    session_id UUID REFERENCES neurondb_agent.sessions(id) ON DELETE SET NULL,
    message_id BIGINT REFERENCES neurondb_agent.messages(id) ON DELETE SET NULL,
    user_id TEXT,
    feedback_type TEXT NOT NULL CHECK (feedback_type IN ('positive', 'negative', 'neutral', 'correction')),
    rating INT CHECK (rating >= 1 AND rating <= 5),
    comment TEXT,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_user_feedback_agent ON neurondb_agent.user_feedback(agent_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_user_feedback_session ON neurondb_agent.user_feedback(session_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_user_feedback_type ON neurondb_agent.user_feedback(feedback_type, created_at);












-- ============================================================================
-- SECTION 14: COLLABORATION WORKSPACE
-- ============================================================================
-- Real-Time Collaboration Workspace Schema
-- Enables multiple users and agents to collaborate on shared tasks
-- Provides workspace management, participant tracking, and real-time updates

-- Collaboration workspaces
CREATE TABLE IF NOT EXISTS neurondb_agent.collaboration_workspaces (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT NOT NULL,
    owner_id UUID,
    description TEXT,
    shared_context JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Workspace participants
CREATE TABLE IF NOT EXISTS neurondb_agent.workspace_participants (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workspace_id UUID NOT NULL REFERENCES neurondb_agent.collaboration_workspaces(id) ON DELETE CASCADE,
    user_id UUID,
    agent_id UUID REFERENCES neurondb_agent.agents(id) ON DELETE CASCADE,
    role TEXT NOT NULL DEFAULT 'member' CHECK (role IN ('owner', 'admin', 'member', 'viewer')),
    joined_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(workspace_id, COALESCE(user_id::text, ''), COALESCE(agent_id::text, ''))
);

-- Workspace updates for real-time synchronization
CREATE TABLE IF NOT EXISTS neurondb_agent.workspace_updates (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workspace_id UUID NOT NULL REFERENCES neurondb_agent.collaboration_workspaces(id) ON DELETE CASCADE,
    user_id UUID,
    agent_id UUID REFERENCES neurondb_agent.agents(id) ON DELETE SET NULL,
    update_type TEXT NOT NULL CHECK (update_type IN ('message', 'action', 'state_change', 'file_update', 'context_sync')),
    content TEXT,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Workspace sessions linking sessions to workspaces
CREATE TABLE IF NOT EXISTS neurondb_agent.workspace_sessions (
    workspace_id UUID NOT NULL REFERENCES neurondb_agent.collaboration_workspaces(id) ON DELETE CASCADE,
    session_id UUID NOT NULL REFERENCES neurondb_agent.sessions(id) ON DELETE CASCADE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (workspace_id, session_id)
);

-- Indexes for collaboration_workspaces
CREATE INDEX IF NOT EXISTS idx_collaboration_workspaces_owner_id ON neurondb_agent.collaboration_workspaces(owner_id);
CREATE INDEX IF NOT EXISTS idx_collaboration_workspaces_created_at ON neurondb_agent.collaboration_workspaces(created_at DESC);

-- Indexes for workspace_participants
CREATE INDEX IF NOT EXISTS idx_workspace_participants_workspace_id ON neurondb_agent.workspace_participants(workspace_id);
CREATE INDEX IF NOT EXISTS idx_workspace_participants_user_id ON neurondb_agent.workspace_participants(user_id) WHERE user_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_workspace_participants_agent_id ON neurondb_agent.workspace_participants(agent_id) WHERE agent_id IS NOT NULL;

-- Indexes for workspace_updates
CREATE INDEX IF NOT EXISTS idx_workspace_updates_workspace_id ON neurondb_agent.workspace_updates(workspace_id);
CREATE INDEX IF NOT EXISTS idx_workspace_updates_created_at ON neurondb_agent.workspace_updates(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_workspace_updates_update_type ON neurondb_agent.workspace_updates(update_type);

-- Indexes for workspace_sessions
CREATE INDEX IF NOT EXISTS idx_workspace_sessions_workspace_id ON neurondb_agent.workspace_sessions(workspace_id);
CREATE INDEX IF NOT EXISTS idx_workspace_sessions_session_id ON neurondb_agent.workspace_sessions(session_id);

COMMENT ON TABLE neurondb_agent.collaboration_workspaces IS 'Shared workspaces for collaborative agent tasks. Multiple users and agents can work together on shared tasks.';
COMMENT ON TABLE neurondb_agent.workspace_participants IS 'Participants in collaboration workspaces. Tracks users and agents with their roles.';
COMMENT ON TABLE neurondb_agent.workspace_updates IS 'Real-time updates broadcast to all workspace participants. Used for live synchronization.';
COMMENT ON TABLE neurondb_agent.workspace_sessions IS 'Links agent sessions to workspaces for shared context and collaboration.';

COMMENT ON COLUMN neurondb_agent.workspace_participants.role IS 'Participant role: owner (full control), admin (manage participants), member (edit), viewer (read-only).';
COMMENT ON COLUMN neurondb_agent.workspace_updates.update_type IS 'Type of update: message (chat), action (agent action), state_change (workspace state), file_update (file change), context_sync (context update).';


-- ============================================================================
-- SECTION 15: VIRTUAL FILESYSTEM
-- ============================================================================
-- Virtual File System for Agent Scratchpad
-- Provides persistent file storage for agent operations and data externalization
-- Supports both database storage for small files and object storage for large files

-- Virtual files table
CREATE TABLE IF NOT EXISTS neurondb_agent.virtual_files (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id UUID NOT NULL REFERENCES neurondb_agent.agents(id) ON DELETE CASCADE,
    session_id UUID REFERENCES neurondb_agent.sessions(id) ON DELETE CASCADE,
    path TEXT NOT NULL,
    content BYTEA,
    content_s3_key TEXT,
    mime_type TEXT NOT NULL DEFAULT 'text/plain',
    size BIGINT NOT NULL DEFAULT 0,
    compressed BOOLEAN NOT NULL DEFAULT false,
    storage_backend TEXT NOT NULL DEFAULT 'database' CHECK (storage_backend IN ('database', 's3')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(agent_id, path)
);

-- Virtual directories table
CREATE TABLE IF NOT EXISTS neurondb_agent.virtual_directories (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id UUID NOT NULL REFERENCES neurondb_agent.agents(id) ON DELETE CASCADE,
    session_id UUID REFERENCES neurondb_agent.sessions(id) ON DELETE CASCADE,
    path TEXT NOT NULL,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(agent_id, path)
);

-- File access log for audit and analytics
CREATE TABLE IF NOT EXISTS neurondb_agent.file_access_log (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    file_id UUID NOT NULL REFERENCES neurondb_agent.virtual_files(id) ON DELETE CASCADE,
    agent_id UUID NOT NULL REFERENCES neurondb_agent.agents(id) ON DELETE CASCADE,
    operation TEXT NOT NULL CHECK (operation IN ('read', 'write', 'delete', 'create', 'copy', 'move')),
    accessed_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Indexes for virtual_files
CREATE INDEX IF NOT EXISTS idx_virtual_files_agent_id ON neurondb_agent.virtual_files(agent_id);
CREATE INDEX IF NOT EXISTS idx_virtual_files_session_id ON neurondb_agent.virtual_files(session_id);
CREATE INDEX IF NOT EXISTS idx_virtual_files_path ON neurondb_agent.virtual_files(agent_id, path);
CREATE INDEX IF NOT EXISTS idx_virtual_files_storage_backend ON neurondb_agent.virtual_files(storage_backend);
CREATE INDEX IF NOT EXISTS idx_virtual_files_created_at ON neurondb_agent.virtual_files(created_at DESC);

-- Indexes for virtual_directories
CREATE INDEX IF NOT EXISTS idx_virtual_directories_agent_id ON neurondb_agent.virtual_directories(agent_id);
CREATE INDEX IF NOT EXISTS idx_virtual_directories_session_id ON neurondb_agent.virtual_directories(session_id);
CREATE INDEX IF NOT EXISTS idx_virtual_directories_path ON neurondb_agent.virtual_directories(agent_id, path);

-- Indexes for file_access_log
CREATE INDEX IF NOT EXISTS idx_file_access_log_file_id ON neurondb_agent.file_access_log(file_id);
CREATE INDEX IF NOT EXISTS idx_file_access_log_agent_id ON neurondb_agent.file_access_log(agent_id);
CREATE INDEX IF NOT EXISTS idx_file_access_log_accessed_at ON neurondb_agent.file_access_log(accessed_at DESC);

COMMENT ON TABLE neurondb_agent.virtual_files IS 'Virtual file system for agent scratchpad operations. Stores file content in database for small files or S3 for large files.';
COMMENT ON TABLE neurondb_agent.virtual_directories IS 'Virtual directory structure for organizing agent files.';
COMMENT ON TABLE neurondb_agent.file_access_log IS 'Audit log of all file operations for security and analytics.';

COMMENT ON COLUMN neurondb_agent.virtual_files.content IS 'File content stored in database for files < 1MB. NULL if stored in S3.';
COMMENT ON COLUMN neurondb_agent.virtual_files.content_s3_key IS 'S3 object key for files stored in object storage. NULL if stored in database.';
COMMENT ON COLUMN neurondb_agent.virtual_files.storage_backend IS 'Storage backend used: database for small files, s3 for large files.';
COMMENT ON COLUMN neurondb_agent.virtual_files.compressed IS 'Whether file content is compressed using gzip.';


-- ============================================================================
-- SECTION 16: BROWSER SESSIONS
-- ============================================================================
-- Browser sessions schema for web browser tool
-- Stores browser session state for persistent browsing across requests

CREATE TABLE IF NOT EXISTS neurondb_agent.browser_sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id TEXT NOT NULL UNIQUE,
    agent_id UUID REFERENCES neurondb_agent.agents(id) ON DELETE CASCADE,
    current_url TEXT,
    cookies JSONB DEFAULT '[]',
    local_storage JSONB DEFAULT '{}',
    session_storage JSONB DEFAULT '{}',
    user_agent TEXT,
    viewport_width INT DEFAULT 1920,
    viewport_height INT DEFAULT 1080,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    expires_at TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_browser_sessions_session_id ON neurondb_agent.browser_sessions(session_id);
CREATE INDEX IF NOT EXISTS idx_browser_sessions_agent_id ON neurondb_agent.browser_sessions(agent_id);
CREATE INDEX IF NOT EXISTS idx_browser_sessions_expires_at ON neurondb_agent.browser_sessions(expires_at) WHERE expires_at IS NOT NULL;

-- Browser snapshots for screenshot storage
CREATE TABLE IF NOT EXISTS neurondb_agent.browser_snapshots (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id TEXT NOT NULL,
    browser_session_id UUID REFERENCES neurondb_agent.browser_sessions(id) ON DELETE CASCADE,
    url TEXT NOT NULL,
    screenshot_data BYTEA,
    screenshot_b64 TEXT,
    page_title TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_browser_snapshots_session_id ON neurondb_agent.browser_snapshots(session_id);
CREATE INDEX IF NOT EXISTS idx_browser_snapshots_browser_session_id ON neurondb_agent.browser_snapshots(browser_session_id);
CREATE INDEX IF NOT EXISTS idx_browser_snapshots_created_at ON neurondb_agent.browser_snapshots(created_at DESC);

COMMENT ON TABLE neurondb_agent.browser_sessions IS 'Stores browser session state for web browser tool automation';
COMMENT ON TABLE neurondb_agent.browser_snapshots IS 'Stores browser screenshots and page snapshots';


-- ============================================================================
-- SECTION 17: SUB-AGENTS
-- ============================================================================
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/migrations/019_sub_agents.sql
 *
 *-------------------------------------------------------------------------
 */

/* Agent specializations table */
CREATE TABLE IF NOT EXISTS neurondb_agent.agent_specializations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id UUID NOT NULL UNIQUE REFERENCES neurondb_agent.agents(id) ON DELETE CASCADE,
    specialization_type TEXT NOT NULL CHECK (specialization_type IN ('planning', 'research', 'coding', 'execution', 'analysis', 'general')),
    capabilities TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[],
    config JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

/* Indexes */
CREATE INDEX IF NOT EXISTS idx_agent_specializations_type ON neurondb_agent.agent_specializations(specialization_type);
CREATE INDEX IF NOT EXISTS idx_agent_specializations_agent ON neurondb_agent.agent_specializations(agent_id);

/* Comments */
COMMENT ON TABLE neurondb_agent.agent_specializations IS 'Defines agent specializations for automatic task routing. Agents can specialize in planning, research, coding, execution, analysis, or general tasks.';
COMMENT ON COLUMN neurondb_agent.agent_specializations.specialization_type IS 'Type of specialization: planning, research, coding, execution, analysis, or general';
COMMENT ON COLUMN neurondb_agent.agent_specializations.capabilities IS 'Array of specific capabilities this agent has (e.g., ["python", "sql", "web_scraping"])';
COMMENT ON COLUMN neurondb_agent.agent_specializations.config IS 'Specialization-specific configuration as JSON';


-- ============================================================================
-- SECTION 18: TASK ALERTS
-- ============================================================================
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/migrations/020_task_alerts.sql
 *
 *-------------------------------------------------------------------------
 */

/* Task alert preferences table for user notification settings */
CREATE TABLE IF NOT EXISTS neurondb_agent.task_alert_preferences (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID,
    agent_id UUID REFERENCES neurondb_agent.agents(id) ON DELETE CASCADE,
    alert_types TEXT[] NOT NULL DEFAULT ARRAY['completion', 'failure']::TEXT[],
    channels TEXT[] NOT NULL DEFAULT ARRAY['webhook']::TEXT[],
    email_address TEXT,
    webhook_url TEXT,
    enabled BOOLEAN NOT NULL DEFAULT true,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(user_id, agent_id)
);

/* Task alerts history table */
CREATE TABLE IF NOT EXISTS neurondb_agent.task_alerts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    task_id UUID NOT NULL REFERENCES neurondb_agent.async_tasks(id) ON DELETE CASCADE,
    alert_type TEXT NOT NULL CHECK (alert_type IN ('completion', 'failure', 'progress', 'milestone')),
    channel TEXT NOT NULL CHECK (channel IN ('email', 'webhook', 'push')),
    recipient TEXT NOT NULL,
    message TEXT,
    sent_at TIMESTAMPTZ,
    status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'sent', 'delivered', 'failed')),
    error_message TEXT,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

/* Indexes */
CREATE INDEX IF NOT EXISTS idx_task_alert_preferences_user ON neurondb_agent.task_alert_preferences(user_id, enabled) WHERE enabled = true;
CREATE INDEX IF NOT EXISTS idx_task_alert_preferences_agent ON neurondb_agent.task_alert_preferences(agent_id, enabled) WHERE enabled = true;
CREATE INDEX IF NOT EXISTS idx_task_alerts_task ON neurondb_agent.task_alerts(task_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_task_alerts_status ON neurondb_agent.task_alerts(status, created_at) WHERE status IN ('pending', 'failed');
CREATE INDEX IF NOT EXISTS idx_task_alerts_type ON neurondb_agent.task_alerts(alert_type, channel);

/* Comments */
COMMENT ON TABLE neurondb_agent.task_alert_preferences IS 'User preferences for task alerts. Configures which alerts to receive and via which channels.';
COMMENT ON COLUMN neurondb_agent.task_alert_preferences.alert_types IS 'Array of alert types to receive: completion, failure, progress, milestone';
COMMENT ON COLUMN neurondb_agent.task_alert_preferences.channels IS 'Array of delivery channels: email, webhook, push';
COMMENT ON COLUMN neurondb_agent.task_alert_preferences.email_address IS 'Email address for email channel notifications';
COMMENT ON COLUMN neurondb_agent.task_alert_preferences.webhook_url IS 'Webhook URL for webhook channel notifications';

COMMENT ON TABLE neurondb_agent.task_alerts IS 'History of task alerts sent to users. Tracks delivery status and errors.';
COMMENT ON COLUMN neurondb_agent.task_alerts.alert_type IS 'Type of alert: completion, failure, progress, or milestone';
COMMENT ON COLUMN neurondb_agent.task_alerts.channel IS 'Delivery channel used: email, webhook, or push';
COMMENT ON COLUMN neurondb_agent.task_alerts.recipient IS 'Recipient identifier (email, webhook URL, or push token)';
COMMENT ON COLUMN neurondb_agent.task_alerts.status IS 'Delivery status: pending, sent, delivered, or failed';


-- ============================================================================
-- SECTION 19: VERIFICATION AGENT
-- ============================================================================
    output_id UUID,
    output_content TEXT,
    status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'processing', 'completed', 'failed')),
    priority TEXT NOT NULL DEFAULT 'medium' CHECK (priority IN ('low', 'medium', 'high')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    processed_at TIMESTAMPTZ
);

-- Verification results
CREATE TABLE IF NOT EXISTS neurondb_agent.verification_results (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    queue_id UUID NOT NULL REFERENCES neurondb_agent.verification_queue(id) ON DELETE CASCADE,
    verifier_agent_id UUID REFERENCES neurondb_agent.agents(id) ON DELETE SET NULL,
    passed BOOLEAN NOT NULL,
    issues JSONB DEFAULT '[]',
    suggestions JSONB DEFAULT '[]',
    confidence FLOAT NOT NULL DEFAULT 0.0 CHECK (confidence >= 0.0 AND confidence <= 1.0),
    verified_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Verification rules
CREATE TABLE IF NOT EXISTS neurondb_agent.verification_rules (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id UUID NOT NULL REFERENCES neurondb_agent.agents(id) ON DELETE CASCADE,
    rule_type TEXT NOT NULL CHECK (rule_type IN ('output_format', 'data_accuracy', 'logical_consistency', 'completeness')),
    criteria JSONB NOT NULL DEFAULT '{}',
    enabled BOOLEAN NOT NULL DEFAULT true,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Indexes for verification_queue
CREATE INDEX IF NOT EXISTS idx_verification_queue_status ON neurondb_agent.verification_queue(status);
CREATE INDEX IF NOT EXISTS idx_verification_queue_priority ON neurondb_agent.verification_queue(priority DESC);
CREATE INDEX IF NOT EXISTS idx_verification_queue_created_at ON neurondb_agent.verification_queue(created_at ASC);
CREATE INDEX IF NOT EXISTS idx_verification_queue_session_id ON neurondb_agent.verification_queue(session_id);

-- Indexes for verification_results
CREATE INDEX IF NOT EXISTS idx_verification_results_queue_id ON neurondb_agent.verification_results(queue_id);
CREATE INDEX IF NOT EXISTS idx_verification_results_verifier_agent_id ON neurondb_agent.verification_results(verifier_agent_id);
CREATE INDEX IF NOT EXISTS idx_verification_results_passed ON neurondb_agent.verification_results(passed);
CREATE INDEX IF NOT EXISTS idx_verification_results_verified_at ON neurondb_agent.verification_results(verified_at DESC);

-- Indexes for verification_rules
CREATE INDEX IF NOT EXISTS idx_verification_rules_agent_id ON neurondb_agent.verification_rules(agent_id);
CREATE INDEX IF NOT EXISTS idx_verification_rules_rule_type ON neurondb_agent.verification_rules(rule_type);
CREATE INDEX IF NOT EXISTS idx_verification_rules_enabled ON neurondb_agent.verification_rules(enabled) WHERE enabled = true;

COMMENT ON TABLE neurondb_agent.verification_queue IS 'Queue of outputs pending verification. Processes outputs through quality assurance checks.';
COMMENT ON TABLE neurondb_agent.verification_results IS 'Results of verification checks including pass/fail status, issues found, and improvement suggestions.';
COMMENT ON TABLE neurondb_agent.verification_rules IS 'Verification rules defining quality criteria for agent outputs. Rules can be enabled or disabled per agent.';

COMMENT ON COLUMN neurondb_agent.verification_queue.priority IS 'Verification priority: low (background), medium (normal), high (immediate).';
COMMENT ON COLUMN neurondb_agent.verification_results.confidence IS 'Confidence score (0-1) indicating reliability of verification result.';
COMMENT ON COLUMN neurondb_agent.verification_rules.criteria IS 'JSONB object defining rule criteria. Format varies by rule_type.';


-- ============================================================================
-- SECTION 20: EVENT STREAM
-- ============================================================================
    event_type TEXT NOT NULL CHECK (event_type IN ('user_message', 'agent_action', 'tool_execution', 'agent_response', 'error', 'system')),
    actor TEXT NOT NULL,
    content TEXT NOT NULL,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_event_stream_session_id ON neurondb_agent.event_stream(session_id);
CREATE INDEX IF NOT EXISTS idx_event_stream_created_at ON neurondb_agent.event_stream(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_event_stream_event_type ON neurondb_agent.event_stream(event_type);
CREATE INDEX IF NOT EXISTS idx_event_stream_session_time ON neurondb_agent.event_stream(session_id, created_at DESC);

-- Event summaries for compressed event history
CREATE TABLE IF NOT EXISTS neurondb_agent.event_summaries (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID NOT NULL REFERENCES neurondb_agent.sessions(id) ON DELETE CASCADE,
    start_event_id UUID NOT NULL,
    end_event_id UUID NOT NULL,
    event_count INT NOT NULL,
    summary_text TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_event_summaries_session_id ON neurondb_agent.event_summaries(session_id);
CREATE INDEX IF NOT EXISTS idx_event_summaries_created_at ON neurondb_agent.event_summaries(created_at DESC);

COMMENT ON TABLE neurondb_agent.event_stream IS 'Chronological log of all user messages, agent actions, tool executions, and system events. Enables event sourcing, context management, and audit trails.';
COMMENT ON TABLE neurondb_agent.event_summaries IS 'Compressed summaries of event ranges for efficient context window management. Created when event count exceeds threshold.';

COMMENT ON COLUMN neurondb_agent.event_stream.event_type IS 'Type of event: user_message (user input), agent_action (agent decision), tool_execution (tool call), agent_response (agent output), error (error event), system (system event)';
COMMENT ON COLUMN neurondb_agent.event_stream.actor IS 'Entity that triggered the event: user ID, agent ID, tool name, or system';
COMMENT ON COLUMN neurondb_agent.event_stream.content IS 'Event content: message text, action description, tool result, error message';
COMMENT ON COLUMN neurondb_agent.event_stream.metadata IS 'Additional event metadata: tool parameters, error details, timing information';


-- ============================================================================
-- SECTION 21: ADVANCED FEATURES
-- ============================================================================
    relationship_type TEXT NOT NULL CHECK (relationship_type IN ('delegates_to', 'collaborates_with', 'supervises', 'reports_to')),
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT different_agents CHECK (from_agent_id != to_agent_id)
);

CREATE INDEX IF NOT EXISTS idx_agent_relationships_from ON neurondb_agent.agent_relationships(from_agent_id);
CREATE INDEX IF NOT EXISTS idx_agent_relationships_to ON neurondb_agent.agent_relationships(to_agent_id);

-- Tool usage logs for analytics
CREATE TABLE IF NOT EXISTS neurondb_agent.tool_usage_logs (
    id BIGSERIAL PRIMARY KEY,
    agent_id UUID REFERENCES neurondb_agent.agents(id) ON DELETE SET NULL,
    session_id UUID REFERENCES neurondb_agent.sessions(id) ON DELETE SET NULL,
    tool_name TEXT NOT NULL,
    execution_time_ms INTEGER,
    success BOOLEAN DEFAULT true,
    error_message TEXT,
    tokens_used INTEGER DEFAULT 0,
    cost REAL DEFAULT 0.0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_tool_usage_agent ON neurondb_agent.tool_usage_logs(agent_id, created_at);
CREATE INDEX IF NOT EXISTS idx_tool_usage_tool ON neurondb_agent.tool_usage_logs(tool_name, created_at);
CREATE INDEX IF NOT EXISTS idx_tool_usage_session ON neurondb_agent.tool_usage_logs(session_id, created_at);

-- Cost logs for cost tracking
CREATE TABLE IF NOT EXISTS neurondb_agent.cost_logs (
    id BIGSERIAL PRIMARY KEY,
    agent_id UUID REFERENCES neurondb_agent.agents(id) ON DELETE SET NULL,
    session_id UUID REFERENCES neurondb_agent.sessions(id) ON DELETE SET NULL,
    cost_type TEXT NOT NULL CHECK (cost_type IN ('llm', 'embedding', 'tool', 'storage', 'other')),
    tokens_used INTEGER DEFAULT 0,
    cost REAL NOT NULL DEFAULT 0.0,
    model_name TEXT,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_cost_logs_agent ON neurondb_agent.cost_logs(agent_id, created_at);
CREATE INDEX IF NOT EXISTS idx_cost_logs_session ON neurondb_agent.cost_logs(session_id, created_at);
CREATE INDEX IF NOT EXISTS idx_cost_logs_type ON neurondb_agent.cost_logs(cost_type, created_at);

-- Quality scores for response quality tracking
CREATE TABLE IF NOT EXISTS neurondb_agent.quality_scores (
    id BIGSERIAL PRIMARY KEY,
    agent_id UUID REFERENCES neurondb_agent.agents(id) ON DELETE SET NULL,
    session_id UUID REFERENCES neurondb_agent.sessions(id) ON DELETE SET NULL,
    message_id BIGINT REFERENCES neurondb_agent.messages(id) ON DELETE SET NULL,
    overall_score REAL CHECK (overall_score >= 0 AND overall_score <= 1),
    accuracy_score REAL CHECK (accuracy_score >= 0 AND accuracy_score <= 1),
    completeness_score REAL CHECK (completeness_score >= 0 AND completeness_score <= 1),
    clarity_score REAL CHECK (clarity_score >= 0 AND clarity_score <= 1),
    relevance_score REAL CHECK (relevance_score >= 0 AND relevance_score <= 1),
    confidence REAL CHECK (confidence >= 0 AND confidence <= 1),
    issues JSONB DEFAULT '[]',
    user_feedback INTEGER CHECK (user_feedback >= -1 AND user_feedback <= 1), -- -1: negative, 0: neutral, 1: positive
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_quality_scores_agent ON neurondb_agent.quality_scores(agent_id, created_at);
CREATE INDEX IF NOT EXISTS idx_quality_scores_session ON neurondb_agent.quality_scores(session_id, created_at);
CREATE INDEX IF NOT EXISTS idx_quality_scores_overall ON neurondb_agent.quality_scores(overall_score);

-- Agent versions for versioning support
CREATE TABLE IF NOT EXISTS neurondb_agent.agent_versions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id UUID NOT NULL REFERENCES neurondb_agent.agents(id) ON DELETE CASCADE,
    version_number INTEGER NOT NULL,
    name TEXT,
    description TEXT,
    system_prompt TEXT NOT NULL,
    model_name TEXT NOT NULL,
    enabled_tools TEXT[] DEFAULT '{}',
    config JSONB DEFAULT '{}',
    is_active BOOLEAN DEFAULT false,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(agent_id, version_number)
);

CREATE INDEX IF NOT EXISTS idx_agent_versions_agent ON neurondb_agent.agent_versions(agent_id, version_number DESC);
CREATE INDEX IF NOT EXISTS idx_agent_versions_active ON neurondb_agent.agent_versions(agent_id, is_active) WHERE is_active = true;

-- Plans for stored plans
CREATE TABLE IF NOT EXISTS neurondb_agent.plans (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id UUID REFERENCES neurondb_agent.agents(id) ON DELETE SET NULL,
    session_id UUID REFERENCES neurondb_agent.sessions(id) ON DELETE SET NULL,
    task_description TEXT NOT NULL,
    steps JSONB NOT NULL,
    status TEXT NOT NULL DEFAULT 'created' CHECK (status IN ('created', 'executing', 'completed', 'failed', 'cancelled')),
    result JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_plans_agent ON neurondb_agent.plans(agent_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_plans_session ON neurondb_agent.plans(session_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_plans_status ON neurondb_agent.plans(status, created_at);

-- Reflections for reflection logs
CREATE TABLE IF NOT EXISTS neurondb_agent.reflections (
    id BIGSERIAL PRIMARY KEY,
    agent_id UUID REFERENCES neurondb_agent.agents(id) ON DELETE SET NULL,
    session_id UUID REFERENCES neurondb_agent.sessions(id) ON DELETE SET NULL,
    message_id BIGINT REFERENCES neurondb_agent.messages(id) ON DELETE SET NULL,
    user_message TEXT NOT NULL,
    agent_response TEXT NOT NULL,
    quality_score REAL CHECK (quality_score >= 0 AND quality_score <= 1),
    accuracy_score REAL CHECK (accuracy_score >= 0 AND accuracy_score <= 1),
    completeness_score REAL CHECK (completeness_score >= 0 AND completeness_score <= 1),
    clarity_score REAL CHECK (clarity_score >= 0 AND clarity_score <= 1),
    relevance_score REAL CHECK (relevance_score >= 0 AND relevance_score <= 1),
    confidence REAL CHECK (confidence >= 0 AND confidence <= 1),
    issues JSONB DEFAULT '[]',
    suggestions JSONB DEFAULT '[]',
    was_retried BOOLEAN DEFAULT false,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_reflections_agent ON neurondb_agent.reflections(agent_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_reflections_session ON neurondb_agent.reflections(session_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_reflections_quality ON neurondb_agent.reflections(quality_score);

-- Add version column to agents table
ALTER TABLE neurondb_agent.agents ADD COLUMN IF NOT EXISTS version INTEGER DEFAULT 1;
ALTER TABLE neurondb_agent.agents ADD COLUMN IF NOT EXISTS parent_agent_id UUID REFERENCES neurondb_agent.agents(id) ON DELETE SET NULL;

CREATE INDEX IF NOT EXISTS idx_agents_version ON neurondb_agent.agents(version);
CREATE INDEX IF NOT EXISTS idx_agents_parent ON neurondb_agent.agents(parent_agent_id);








-- ============================================================================
-- SECTION 22: EVALUATION FRAMEWORK
-- ============================================================================
CREATE TABLE IF NOT EXISTS neurondb_agent.eval_tasks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    task_type TEXT NOT NULL,  -- e.g., 'tool_sequence', 'sql_side_effect', 'retrieval', 'end_to_end'
    input TEXT NOT NULL,  -- Input for the task
    expected_output TEXT,  -- Expected output
    expected_tool_sequence JSONB,  -- Expected sequence of tool calls
    golden_sql_side_effects JSONB,  -- Expected SQL side effects (table states)
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_eval_tasks_task_type ON neurondb_agent.eval_tasks(task_type);
CREATE INDEX IF NOT EXISTS idx_eval_tasks_created_at ON neurondb_agent.eval_tasks(created_at DESC);

-- Eval runs table: Evaluation run metadata
CREATE TABLE IF NOT EXISTS neurondb_agent.eval_runs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    dataset_version TEXT NOT NULL,  -- Version identifier for the dataset
    agent_id UUID REFERENCES neurondb_agent.agents(id) ON DELETE SET NULL,
    started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at TIMESTAMPTZ,
    score REAL,  -- Overall score (0-1)
    total_tasks INT DEFAULT 0,
    passed_tasks INT DEFAULT 0,
    failed_tasks INT DEFAULT 0,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_eval_runs_dataset_version ON neurondb_agent.eval_runs(dataset_version);
CREATE INDEX IF NOT EXISTS idx_eval_runs_agent_id ON neurondb_agent.eval_runs(agent_id);
CREATE INDEX IF NOT EXISTS idx_eval_runs_started_at ON neurondb_agent.eval_runs(started_at DESC);

-- Eval task results table: Results for individual tasks in a run
CREATE TABLE IF NOT EXISTS neurondb_agent.eval_task_results (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    eval_run_id UUID NOT NULL REFERENCES neurondb_agent.eval_runs(id) ON DELETE CASCADE,
    eval_task_id UUID NOT NULL REFERENCES neurondb_agent.eval_tasks(id) ON DELETE CASCADE,
    session_id UUID REFERENCES neurondb_agent.sessions(id) ON DELETE SET NULL,
    passed BOOLEAN NOT NULL DEFAULT false,
    actual_output TEXT,
    actual_tool_sequence JSONB,
    actual_sql_side_effects JSONB,
    score REAL,  -- Task-specific score (0-1)
    error_message TEXT,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_eval_task_results_eval_run_id ON neurondb_agent.eval_task_results(eval_run_id);
CREATE INDEX IF NOT EXISTS idx_eval_task_results_eval_task_id ON neurondb_agent.eval_task_results(eval_task_id);
CREATE INDEX IF NOT EXISTS idx_eval_task_results_passed ON neurondb_agent.eval_task_results(passed);

-- Retrieval eval results table: Specialized results for retrieval evaluation
CREATE TABLE IF NOT EXISTS neurondb_agent.eval_retrieval_results (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    eval_task_result_id UUID NOT NULL REFERENCES neurondb_agent.eval_task_results(id) ON DELETE CASCADE,
    recall_at_k REAL,  -- Recall@k score
    mrr REAL,  -- Mean Reciprocal Rank
    grounding_passed BOOLEAN,  -- Whether retrieved chunks were properly cited
    retrieved_chunks JSONB,  -- Retrieved chunks
    relevant_chunks JSONB,  -- Ground truth relevant chunks
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_eval_retrieval_results_task_result_id ON neurondb_agent.eval_retrieval_results(eval_task_result_id);



-- ============================================================================
-- SECTION 23: EXECUTION SNAPSHOTS
-- ============================================================================
CREATE TABLE IF NOT EXISTS neurondb_agent.execution_snapshots (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID NOT NULL REFERENCES neurondb_agent.sessions(id) ON DELETE CASCADE,
    agent_id UUID NOT NULL REFERENCES neurondb_agent.agents(id) ON DELETE CASCADE,
    user_message TEXT NOT NULL,
    execution_state JSONB NOT NULL,  -- Complete execution state (inputs, tool calls, outputs, LLM responses)
    deterministic_mode BOOLEAN DEFAULT false,  -- Whether execution was in deterministic mode
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_execution_snapshots_session_id ON neurondb_agent.execution_snapshots(session_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_execution_snapshots_agent_id ON neurondb_agent.execution_snapshots(agent_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_execution_snapshots_deterministic ON neurondb_agent.execution_snapshots(deterministic_mode, created_at DESC);



-- ============================================================================
-- COMPLETION MESSAGE
-- ============================================================================

DO $$
BEGIN
    RAISE NOTICE 'NeuronAgent Database Schema setup completed successfully!';
    RAISE NOTICE 'Created all tables, indexes, views, triggers, and functions.';
    RAISE NOTICE 'Next steps:';
    RAISE NOTICE '1. Verify setup: SELECT COUNT(*) FROM neurondb_agent.agents;';
    RAISE NOTICE '2. Check principals: SELECT * FROM neurondb_agent.principals LIMIT 5;';
    RAISE NOTICE '3. Review audit log: SELECT * FROM neurondb_agent.audit_log LIMIT 5;';
END $$;
