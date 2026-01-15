/*-------------------------------------------------------------------------
 *
 * neuronagent_distributed_schema.sql
 *    Database schema for distributed architecture
 *
 * Creates tables for cluster nodes, events, and distributed cache.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/sql/neuronagent_distributed_schema.sql
 *
 *-------------------------------------------------------------------------
 */

-- Cluster nodes table: Tracks nodes in the cluster
CREATE TABLE IF NOT EXISTS neurondb_agent.cluster_nodes (
    node_id TEXT PRIMARY KEY,
    address TEXT NOT NULL,
    port INTEGER NOT NULL,
    status TEXT NOT NULL DEFAULT 'healthy' CHECK (status IN ('healthy', 'unhealthy', 'unknown')),
    last_seen TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    capabilities JSONB DEFAULT '[]',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_cluster_nodes_status ON neurondb_agent.cluster_nodes(status);
CREATE INDEX IF NOT EXISTS idx_cluster_nodes_last_seen ON neurondb_agent.cluster_nodes(last_seen);

COMMENT ON TABLE neurondb_agent.cluster_nodes IS 'Cluster nodes for distributed execution';

-- Events table: Event sourcing for audit and replay
CREATE TABLE IF NOT EXISTS neurondb_agent.events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    type TEXT NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    source TEXT NOT NULL,
    data JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_events_type ON neurondb_agent.events(type);
CREATE INDEX IF NOT EXISTS idx_events_timestamp ON neurondb_agent.events(timestamp);
CREATE INDEX IF NOT EXISTS idx_events_source ON neurondb_agent.events(source);
CREATE INDEX IF NOT EXISTS idx_events_created_at ON neurondb_agent.events(created_at);

COMMENT ON TABLE neurondb_agent.events IS 'Event sourcing for agent actions and system events';

-- Cache table: Distributed cache storage
CREATE TABLE IF NOT EXISTS neurondb_agent.cache (
    key TEXT PRIMARY KEY,
    value JSONB NOT NULL,
    expires_at TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_cache_expires_at ON neurondb_agent.cache(expires_at);

COMMENT ON TABLE neurondb_agent.cache IS 'Distributed cache storage (L3 cache)';

-- Function to clean up expired cache entries
CREATE OR REPLACE FUNCTION neurondb_agent.cleanup_expired_cache()
RETURNS void
LANGUAGE plpgsql
AS $$
BEGIN
    DELETE FROM neurondb_agent.cache WHERE expires_at < NOW();
END;
$$;

-- Function to update cluster node last_seen
CREATE OR REPLACE FUNCTION neurondb_agent.update_cluster_node_seen(p_node_id TEXT)
RETURNS void
LANGUAGE plpgsql
AS $$
BEGIN
    UPDATE neurondb_agent.cluster_nodes
    SET last_seen = NOW(), updated_at = NOW()
    WHERE node_id = p_node_id;
END;
$$;


