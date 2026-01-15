/*-------------------------------------------------------------------------
 *
 * neuronagent_marketplace_schema.sql
 *    Database schema for marketplace features
 *
 * Creates tables for tool, agent, and workflow marketplace.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/sql/neuronagent_marketplace_schema.sql
 *
 *-------------------------------------------------------------------------
 */

-- Marketplace tools table
CREATE TABLE IF NOT EXISTS neurondb_agent.marketplace_tools (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT NOT NULL,
    description TEXT,
    version TEXT NOT NULL,
    author TEXT NOT NULL,
    rating FLOAT DEFAULT 0 CHECK (rating >= 0 AND rating <= 5),
    downloads INTEGER DEFAULT 0,
    schema JSONB NOT NULL,
    code TEXT,
    tags TEXT[] DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(name, version)
);

CREATE INDEX IF NOT EXISTS idx_marketplace_tools_name ON neurondb_agent.marketplace_tools(name);
CREATE INDEX IF NOT EXISTS idx_marketplace_tools_rating ON neurondb_agent.marketplace_tools(rating);
CREATE INDEX IF NOT EXISTS idx_marketplace_tools_tags ON neurondb_agent.marketplace_tools USING GIN(tags);

COMMENT ON TABLE neurondb_agent.marketplace_tools IS 'Tool marketplace for sharing and discovering tools';

-- Tool ratings table
CREATE TABLE IF NOT EXISTS neurondb_agent.tool_ratings (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tool_id UUID NOT NULL REFERENCES neurondb_agent.marketplace_tools(id) ON DELETE CASCADE,
    user_id TEXT NOT NULL,
    rating FLOAT NOT NULL CHECK (rating >= 0 AND rating <= 5),
    review TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(tool_id, user_id)
);

CREATE INDEX IF NOT EXISTS idx_tool_ratings_tool_id ON neurondb_agent.tool_ratings(tool_id);
CREATE INDEX IF NOT EXISTS idx_tool_ratings_user_id ON neurondb_agent.tool_ratings(user_id);

-- Marketplace agents table
CREATE TABLE IF NOT EXISTS neurondb_agent.marketplace_agents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT NOT NULL,
    description TEXT,
    version TEXT NOT NULL,
    author TEXT NOT NULL,
    rating FLOAT DEFAULT 0 CHECK (rating >= 0 AND rating <= 5),
    downloads INTEGER DEFAULT 0,
    system_prompt TEXT NOT NULL,
    model_name TEXT NOT NULL,
    enabled_tools TEXT[] DEFAULT '{}',
    config JSONB DEFAULT '{}',
    performance JSONB DEFAULT '{}',
    tags TEXT[] DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(name, version)
);

CREATE INDEX IF NOT EXISTS idx_marketplace_agents_name ON neurondb_agent.marketplace_agents(name);
CREATE INDEX IF NOT EXISTS idx_marketplace_agents_rating ON neurondb_agent.marketplace_agents(rating);
CREATE INDEX IF NOT EXISTS idx_marketplace_agents_tags ON neurondb_agent.marketplace_agents USING GIN(tags);

COMMENT ON TABLE neurondb_agent.marketplace_agents IS 'Agent marketplace for sharing and discovering agents';

-- Marketplace workflows table
CREATE TABLE IF NOT EXISTS neurondb_agent.marketplace_workflows (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT NOT NULL,
    description TEXT,
    version TEXT NOT NULL,
    author TEXT NOT NULL,
    rating FLOAT DEFAULT 0 CHECK (rating >= 0 AND rating <= 5),
    downloads INTEGER DEFAULT 0,
    workflow_def JSONB NOT NULL,
    tags TEXT[] DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(name, version)
);

CREATE INDEX IF NOT EXISTS idx_marketplace_workflows_name ON neurondb_agent.marketplace_workflows(name);
CREATE INDEX IF NOT EXISTS idx_marketplace_workflows_rating ON neurondb_agent.marketplace_workflows(rating);

COMMENT ON TABLE neurondb_agent.marketplace_workflows IS 'Workflow marketplace for sharing and discovering workflows';


