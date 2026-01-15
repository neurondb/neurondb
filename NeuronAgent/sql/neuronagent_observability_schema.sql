/*-------------------------------------------------------------------------
 *
 * neuronagent_observability_schema.sql
 *    Database schema for observability features
 *
 * Creates tables for execution trace, decision trees, and performance profiles.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/sql/neuronagent_observability_schema.sql
 *
 *-------------------------------------------------------------------------
 */

-- Execution trace table: Tracks execution steps for decision tree visualization
CREATE TABLE IF NOT EXISTS neurondb_agent.execution_trace (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    execution_id UUID NOT NULL,
    step_type TEXT NOT NULL CHECK (step_type IN ('llm_call', 'tool_call', 'memory_retrieval', 'decision', 'other')),
    description TEXT,
    input_data JSONB DEFAULT '{}',
    output_data JSONB DEFAULT '{}',
    parent_id UUID REFERENCES neurondb_agent.execution_trace(id) ON DELETE CASCADE,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_execution_trace_execution_id ON neurondb_agent.execution_trace(execution_id);
CREATE INDEX IF NOT EXISTS idx_execution_trace_step_type ON neurondb_agent.execution_trace(step_type);
CREATE INDEX IF NOT EXISTS idx_execution_trace_parent_id ON neurondb_agent.execution_trace(parent_id);
CREATE INDEX IF NOT EXISTS idx_execution_trace_timestamp ON neurondb_agent.execution_trace(timestamp);

COMMENT ON TABLE neurondb_agent.execution_trace IS 'Execution trace for decision tree visualization';

-- Performance profiles table: Stores detailed performance metrics
CREATE TABLE IF NOT EXISTS neurondb_agent.performance_profiles (
    execution_id UUID PRIMARY KEY,
    total_time INTERVAL NOT NULL,
    llm_time INTERVAL,
    tool_time INTERVAL,
    memory_time INTERVAL,
    database_time INTERVAL,
    cpu_time INTERVAL,
    memory_mb FLOAT,
    network_mb FLOAT,
    gpu_time INTERVAL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_performance_profiles_created_at ON neurondb_agent.performance_profiles(created_at);

COMMENT ON TABLE neurondb_agent.performance_profiles IS 'Performance profiles for bottleneck identification';


