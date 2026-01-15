/*-------------------------------------------------------------------------
 *
 * neuronagent_reliability_schema.sql
 *    Database schema for reliability features
 *
 * Creates tables for dead letter queue and failover tracking.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/sql/neuronagent_reliability_schema.sql
 *
 *-------------------------------------------------------------------------
 */

-- Dead letter queue: Stores failed operations for later processing
CREATE TABLE IF NOT EXISTS neurondb_agent.dead_letter_queue (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    operation TEXT NOT NULL,
    payload JSONB NOT NULL,
    error TEXT NOT NULL,
    retry_count INTEGER DEFAULT 0,
    last_retry_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    processed_at TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_dead_letter_queue_operation ON neurondb_agent.dead_letter_queue(operation);
CREATE INDEX IF NOT EXISTS idx_dead_letter_queue_created_at ON neurondb_agent.dead_letter_queue(created_at);
CREATE INDEX IF NOT EXISTS idx_dead_letter_queue_processed ON neurondb_agent.dead_letter_queue(processed_at) WHERE processed_at IS NULL;

COMMENT ON TABLE neurondb_agent.dead_letter_queue IS 'Dead letter queue for failed operations';

-- Add role column to cluster_nodes if it doesn't exist
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_schema = 'neurondb_agent' 
        AND table_name = 'cluster_nodes' 
        AND column_name = 'role'
    ) THEN
        ALTER TABLE neurondb_agent.cluster_nodes 
        ADD COLUMN role TEXT DEFAULT 'replica' CHECK (role IN ('primary', 'replica'));
    END IF;
END $$;

CREATE INDEX IF NOT EXISTS idx_cluster_nodes_role ON neurondb_agent.cluster_nodes(role);


