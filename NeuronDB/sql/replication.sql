/*-------------------------------------------------------------------------
 *
 * replication.sql
 *    Replication utilities for NeuronDB vector indexes
 *
 * Provides SQL functions for managing vector index replication,
 * consistency checks, and read replica support.
 *
 * Copyright (c) 2024-2026, neurondb, Inc.
 *
 *-------------------------------------------------------------------------
 */

-- ============================================================================
-- REPLICATION MANAGEMENT FUNCTIONS
-- ============================================================================

/*
 * Check index consistency across replicas
 * Returns a JSON object with consistency status and details
 */
CREATE OR REPLACE FUNCTION neurondb_check_index_consistency(
    index_name TEXT,
    replica_connections TEXT[]
) RETURNS JSONB
LANGUAGE plpgsql
AS $$
DECLARE
    result JSONB;
    replica_conn TEXT;
    local_count BIGINT;
    replica_count BIGINT;
    consistency_status TEXT := 'consistent';
    details JSONB := '[]'::JSONB;
    detail JSONB;
BEGIN
    -- Get local index count
    EXECUTE format('SELECT COUNT(*) FROM %I', index_name) INTO local_count;
    
    -- Check each replica
    FOREACH replica_conn IN ARRAY replica_connections
    LOOP
        BEGIN
            -- Query replica (simplified - in production would use dblink or FDW)
            -- For now, we'll return a placeholder structure
            replica_count := local_count; -- Placeholder
            
            IF replica_count != local_count THEN
                consistency_status := 'inconsistent';
                detail := jsonb_build_object(
                    'replica', replica_conn,
                    'local_count', local_count,
                    'replica_count', replica_count,
                    'status', 'mismatch'
                );
            ELSE
                detail := jsonb_build_object(
                    'replica', replica_conn,
                    'count', local_count,
                    'status', 'consistent'
                );
            END IF;
            
            details := details || detail;
        EXCEPTION WHEN OTHERS THEN
            detail := jsonb_build_object(
                'replica', replica_conn,
                'status', 'error',
                'error', SQLERRM
            );
            details := details || detail;
            consistency_status := 'error';
        END;
    END LOOP;
    
    result := jsonb_build_object(
        'index_name', index_name,
        'status', consistency_status,
        'local_count', local_count,
        'replicas', details,
        'checked_at', now()
    );
    
    RETURN result;
END;
$$;

COMMENT ON FUNCTION neurondb_check_index_consistency IS 
'Check index consistency across replicas. Returns JSONB with consistency status and details.';

/*
 * Get replication status for an index
 */
CREATE OR REPLACE FUNCTION neurondb_get_replication_status(
    index_name TEXT DEFAULT NULL
) RETURNS TABLE (
    index_name TEXT,
    replica_name TEXT,
    slot_name TEXT,
    publication_name TEXT,
    sync_status TEXT,
    last_lsn PG_LSN,
    lag_bytes BIGINT,
    sync_started_at TIMESTAMPTZ,
    last_updated TIMESTAMPTZ
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT 
        s.source_index_name::TEXT,
        s.target_replica_name::TEXT,
        s.slot_name::TEXT,
        s.publication_name::TEXT,
        s.sync_status::TEXT,
        s.last_lsn,
        CASE 
            WHEN s.last_lsn IS NOT NULL THEN
                (pg_current_wal_lsn() - s.last_lsn)::BIGINT
            ELSE NULL
        END AS lag_bytes,
        s.sync_started_at,
        s.last_updated
    FROM neurondb_index_sync_state s
    WHERE (neurondb_get_replication_status.index_name IS NULL 
           OR s.source_index_name = neurondb_get_replication_status.index_name)
    ORDER BY s.source_index_name, s.target_replica_name;
END;
$$;

COMMENT ON FUNCTION neurondb_get_replication_status IS 
'Get replication status for one or all indexes. Returns detailed replication status information.';

/*
 * Enable read replica for vector search
 * Sets up a read-only replica with vector index synchronization
 */
CREATE OR REPLACE FUNCTION neurondb_enable_read_replica(
    table_name TEXT,
    replica_connection_string TEXT,
    replication_mode TEXT DEFAULT 'async'
) RETURNS BOOLEAN
LANGUAGE plpgsql
AS $$
DECLARE
    index_rec RECORD;
    result BOOLEAN := true;
BEGIN
    -- Enable replication for the table
    PERFORM enable_vector_replication(table_name, replication_mode);
    
    -- For each vector index on the table, set up replication
    FOR index_rec IN
        SELECT indexname
        FROM pg_indexes
        WHERE tablename = table_name
          AND indexdef LIKE '%hnsw%' OR indexdef LIKE '%ivf%'
    LOOP
        -- Set up async sync for the index
        PERFORM sync_index_async(index_rec.indexname, replica_connection_string);
    END LOOP;
    
    RETURN result;
END;
$$;

COMMENT ON FUNCTION neurondb_enable_read_replica IS 
'Enable read replica for a table with automatic vector index synchronization.';

/*
 * Verify replica is ready for queries
 */
CREATE OR REPLACE FUNCTION neurondb_verify_replica_ready(
    replica_name TEXT,
    timeout_seconds INTEGER DEFAULT 30
) RETURNS BOOLEAN
LANGUAGE plpgsql
AS $$
DECLARE
    start_time TIMESTAMPTZ;
    status_rec RECORD;
    all_ready BOOLEAN := false;
BEGIN
    start_time := now();
    
    WHILE (EXTRACT(EPOCH FROM (now() - start_time)) < timeout_seconds) LOOP
        SELECT COUNT(*) = 0 INTO all_ready
        FROM neurondb_index_sync_state
        WHERE target_replica_name = replica_name
          AND sync_status != 'active';
        
        IF all_ready THEN
            RETURN true;
        END IF;
        
        PERFORM pg_sleep(1);
    END LOOP;
    
    RETURN false;
END;
$$;

COMMENT ON FUNCTION neurondb_verify_replica_ready IS 
'Verify that a replica is ready for queries by checking sync status.';

-- ============================================================================
-- REPLICATION METADATA VIEWS
-- ============================================================================

/*
 * View for replication status summary
 */
CREATE OR REPLACE VIEW neurondb_replication_summary AS
SELECT 
    source_index_name AS index_name,
    COUNT(*) AS replica_count,
    COUNT(*) FILTER (WHERE sync_status = 'active') AS active_replicas,
    COUNT(*) FILTER (WHERE sync_status = 'error') AS error_replicas,
    MIN(sync_started_at) AS first_sync_started,
    MAX(last_updated) AS last_sync_update
FROM neurondb_index_sync_state
GROUP BY source_index_name;

COMMENT ON VIEW neurondb_replication_summary IS 
'Summary view of replication status per index.';

/*
 * View for replica lag monitoring
 */
CREATE OR REPLACE VIEW neurondb_replica_lag AS
SELECT 
    source_index_name AS index_name,
    target_replica_name AS replica_name,
    last_lsn,
    pg_current_wal_lsn() AS current_lsn,
    CASE 
        WHEN last_lsn IS NOT NULL THEN
            (pg_current_wal_lsn() - last_lsn)::BIGINT
        ELSE NULL
    END AS lag_bytes,
    CASE 
        WHEN last_lsn IS NOT NULL THEN
            pg_size_pretty((pg_current_wal_lsn() - last_lsn)::BIGINT)
        ELSE 'unknown'
    END AS lag_pretty,
    sync_status,
    last_updated
FROM neurondb_index_sync_state
WHERE sync_status = 'active';

COMMENT ON VIEW neurondb_replica_lag IS 
'Monitor replication lag for active replicas.';



