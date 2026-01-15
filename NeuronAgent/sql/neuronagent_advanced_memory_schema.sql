/*-------------------------------------------------------------------------
 *
 * neuronagent_advanced_memory_schema.sql
 *    Database schema for advanced memory systems
 *
 * Creates tables for episodic memory, semantic memory, and memory indexing.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/sql/neuronagent_advanced_memory_schema.sql
 *
 *-------------------------------------------------------------------------
 */

-- Episodic memory table: Stores specific events with temporal context
CREATE TABLE IF NOT EXISTS neurondb_agent.episodic_memory (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id UUID NOT NULL REFERENCES neurondb_agent.agents(id) ON DELETE CASCADE,
    session_id UUID REFERENCES neurondb_agent.sessions(id) ON DELETE SET NULL,
    event TEXT NOT NULL,
    context TEXT,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    importance FLOAT NOT NULL DEFAULT 0.5 CHECK (importance >= 0 AND importance <= 1),
    emotional_valence FLOAT DEFAULT 0 CHECK (emotional_valence >= -1 AND emotional_valence <= 1),
    embedding VECTOR(1536), -- Adjust dimension based on embedding model
    access_count INTEGER DEFAULT 0,
    last_accessed TIMESTAMPTZ,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_episodic_memory_agent_id ON neurondb_agent.episodic_memory(agent_id);
CREATE INDEX IF NOT EXISTS idx_episodic_memory_timestamp ON neurondb_agent.episodic_memory(timestamp);
CREATE INDEX IF NOT EXISTS idx_episodic_memory_importance ON neurondb_agent.episodic_memory(importance);
CREATE INDEX IF NOT EXISTS idx_episodic_memory_embedding ON neurondb_agent.episodic_memory USING hnsw (embedding vector_cosine_ops);

COMMENT ON TABLE neurondb_agent.episodic_memory IS 'Episodic memory for storing specific events with temporal context';

-- Semantic memory table: Stores factual knowledge and concepts
CREATE TABLE IF NOT EXISTS neurondb_agent.semantic_memory (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id UUID NOT NULL REFERENCES neurondb_agent.agents(id) ON DELETE CASCADE,
    concept TEXT NOT NULL,
    fact TEXT NOT NULL,
    category TEXT,
    confidence FLOAT NOT NULL DEFAULT 0.5 CHECK (confidence >= 0 AND confidence <= 1),
    embedding VECTOR(1536), -- Adjust dimension based on embedding model
    relations TEXT[] DEFAULT '{}',
    access_count INTEGER DEFAULT 0,
    last_accessed TIMESTAMPTZ,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_semantic_memory_agent_id ON neurondb_agent.semantic_memory(agent_id);
CREATE INDEX IF NOT EXISTS idx_semantic_memory_category ON neurondb_agent.semantic_memory(category);
CREATE INDEX IF NOT EXISTS idx_semantic_memory_confidence ON neurondb_agent.semantic_memory(confidence);
CREATE INDEX IF NOT EXISTS idx_semantic_memory_embedding ON neurondb_agent.semantic_memory USING hnsw (embedding vector_cosine_ops);

COMMENT ON TABLE neurondb_agent.semantic_memory IS 'Semantic memory for storing factual knowledge and concepts';

-- Add memory_type and category columns to memory_chunks if they don't exist
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_schema = 'neurondb_agent' 
        AND table_name = 'memory_chunks' 
        AND column_name = 'memory_type'
    ) THEN
        ALTER TABLE neurondb_agent.memory_chunks 
        ADD COLUMN memory_type TEXT DEFAULT 'working' CHECK (memory_type IN ('working', 'episodic', 'semantic'));
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_schema = 'neurondb_agent' 
        AND table_name = 'memory_chunks' 
        AND column_name = 'category'
    ) THEN
        ALTER TABLE neurondb_agent.memory_chunks 
        ADD COLUMN category TEXT;
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_schema = 'neurondb_agent' 
        AND table_name = 'memory_chunks' 
        AND column_name = 'last_accessed'
    ) THEN
        ALTER TABLE neurondb_agent.memory_chunks 
        ADD COLUMN last_accessed TIMESTAMPTZ;
    END IF;
END $$;

CREATE INDEX IF NOT EXISTS idx_memory_chunks_memory_type ON neurondb_agent.memory_chunks(memory_type);
CREATE INDEX IF NOT EXISTS idx_memory_chunks_category ON neurondb_agent.memory_chunks(category);
CREATE INDEX IF NOT EXISTS idx_memory_chunks_last_accessed ON neurondb_agent.memory_chunks(last_accessed);

-- Function to update last_accessed timestamp
CREATE OR REPLACE FUNCTION neurondb_agent.update_memory_access(p_memory_id UUID)
RETURNS void
LANGUAGE plpgsql
AS $$
BEGIN
    UPDATE neurondb_agent.memory_chunks
    SET last_accessed = NOW(), access_count = COALESCE(access_count, 0) + 1
    WHERE id = p_memory_id;
END;
$$;

-- Function to update episodic memory access
CREATE OR REPLACE FUNCTION neurondb_agent.update_episodic_access(p_memory_id UUID)
RETURNS void
LANGUAGE plpgsql
AS $$
BEGIN
    UPDATE neurondb_agent.episodic_memory
    SET last_accessed = NOW(), access_count = COALESCE(access_count, 0) + 1
    WHERE id = p_memory_id;
END;
$$;

-- Function to update semantic memory access
CREATE OR REPLACE FUNCTION neurondb_agent.update_semantic_access(p_memory_id UUID)
RETURNS void
LANGUAGE plpgsql
AS $$
BEGIN
    UPDATE neurondb_agent.semantic_memory
    SET last_accessed = NOW(), access_count = COALESCE(access_count, 0) + 1
    WHERE id = p_memory_id;
END;
$$;


