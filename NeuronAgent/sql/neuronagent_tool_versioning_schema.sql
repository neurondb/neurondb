/*-------------------------------------------------------------------------
 *
 * neuronagent_tool_versioning_schema.sql
 *    Database schema for tool versioning
 *
 * Creates tables for tool versioning with migration support.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/sql/neuronagent_tool_versioning_schema.sql
 *
 *-------------------------------------------------------------------------
 */

-- Tool versions table
CREATE TABLE IF NOT EXISTS neurondb_agent.tool_versions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tool_name TEXT NOT NULL,
    version TEXT NOT NULL,
    schema JSONB NOT NULL,
    code TEXT,
    is_default BOOLEAN DEFAULT false,
    deprecated BOOLEAN DEFAULT false,
    migration TEXT, -- Migration script for upgrading from previous version
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(tool_name, version)
);

CREATE INDEX IF NOT EXISTS idx_tool_versions_tool_name ON neurondb_agent.tool_versions(tool_name);
CREATE INDEX IF NOT EXISTS idx_tool_versions_is_default ON neurondb_agent.tool_versions(is_default);
CREATE INDEX IF NOT EXISTS idx_tool_versions_deprecated ON neurondb_agent.tool_versions(deprecated);

COMMENT ON TABLE neurondb_agent.tool_versions IS 'Tool versioning with migration support';


