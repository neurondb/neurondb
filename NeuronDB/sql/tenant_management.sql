/*-------------------------------------------------------------------------
 *
 * tenant_management.sql
 *    Tenant management functions for NeuronDB
 *
 * Provides SQL functions for tenant creation, management, resource quotas,
 * and monitoring in a multi-tenant environment.
 *
 * Copyright (c) 2024-2026, neurondb, Inc.
 *
 *-------------------------------------------------------------------------
 */

-- ============================================================================
-- TENANT MANAGEMENT FUNCTIONS
-- ============================================================================

/*
 * Create a new tenant
 */
CREATE OR REPLACE FUNCTION neurondb_create_tenant(
    tenant_name TEXT,
    max_vectors BIGINT DEFAULT 1000000,
    max_storage_mb BIGINT DEFAULT 10240,
    max_qps INTEGER DEFAULT 1000
) RETURNS JSONB
LANGUAGE plpgsql
AS $$
DECLARE
    tenant_id INTEGER;
    result JSONB;
BEGIN
    -- Generate tenant ID
    tenant_id := nextval('neurondb_tenant_id_seq');
    
    -- Insert tenant record
    INSERT INTO neurondb_tenants (
        tenant_id,
        tenant_name,
        max_vectors,
        max_storage_mb,
        max_qps,
        created_at,
        status
    ) VALUES (
        tenant_id,
        tenant_name,
        max_vectors,
        max_storage_mb,
        max_qps,
        now(),
        'active'
    );
    
    result := jsonb_build_object(
        'tenant_id', tenant_id,
        'tenant_name', tenant_name,
        'max_vectors', max_vectors,
        'max_storage_mb', max_storage_mb,
        'max_qps', max_qps,
        'created_at', now(),
        'status', 'active'
    );
    
    RETURN result;
END;
$$;

COMMENT ON FUNCTION neurondb_create_tenant IS 
'Create a new tenant with resource quotas. Returns tenant metadata as JSONB.';

/*
 * Get tenant information
 */
CREATE OR REPLACE FUNCTION neurondb_get_tenant(
    tenant_id INTEGER
) RETURNS JSONB
LANGUAGE plpgsql
AS $$
DECLARE
    result JSONB;
    tenant_rec RECORD;
BEGIN
    SELECT * INTO tenant_rec
    FROM neurondb_tenants
    WHERE tenant_id = neurondb_get_tenant.tenant_id;
    
    IF NOT FOUND THEN
        RETURN jsonb_build_object('error', 'Tenant not found');
    END IF;
    
    result := jsonb_build_object(
        'tenant_id', tenant_rec.tenant_id,
        'tenant_name', tenant_rec.tenant_name,
        'max_vectors', tenant_rec.max_vectors,
        'max_storage_mb', tenant_rec.max_storage_mb,
        'max_qps', tenant_rec.max_qps,
        'current_vectors', COALESCE(tenant_rec.current_vectors, 0),
        'current_storage_mb', COALESCE(tenant_rec.current_storage_mb, 0),
        'current_qps', COALESCE(tenant_rec.current_qps, 0),
        'created_at', tenant_rec.created_at,
        'status', tenant_rec.status
    );
    
    RETURN result;
END;
$$;

COMMENT ON FUNCTION neurondb_get_tenant IS 
'Get tenant information and current resource usage.';

/*
 * Update tenant quotas
 */
CREATE OR REPLACE FUNCTION neurondb_update_tenant_quotas(
    tenant_id INTEGER,
    max_vectors BIGINT DEFAULT NULL,
    max_storage_mb BIGINT DEFAULT NULL,
    max_qps INTEGER DEFAULT NULL
) RETURNS JSONB
LANGUAGE plpgsql
AS $$
DECLARE
    result JSONB;
BEGIN
    UPDATE neurondb_tenants
    SET 
        max_vectors = COALESCE(neurondb_update_tenant_quotas.max_vectors, max_vectors),
        max_storage_mb = COALESCE(neurondb_update_tenant_quotas.max_storage_mb, max_storage_mb),
        max_qps = COALESCE(neurondb_update_tenant_quotas.max_qps, max_qps),
        updated_at = now()
    WHERE tenant_id = neurondb_update_tenant_quotas.tenant_id
    RETURNING * INTO result;
    
    IF NOT FOUND THEN
        RETURN jsonb_build_object('error', 'Tenant not found');
    END IF;
    
    RETURN result;
END;
$$;

COMMENT ON FUNCTION neurondb_update_tenant_quotas IS 
'Update tenant resource quotas.';

/*
 * Check tenant quota usage
 */
CREATE OR REPLACE FUNCTION neurondb_check_tenant_quota(
    tenant_id INTEGER
) RETURNS JSONB
LANGUAGE plpgsql
AS $$
DECLARE
    result JSONB;
    tenant_rec RECORD;
    vectors_usage FLOAT;
    storage_usage FLOAT;
    qps_usage FLOAT;
BEGIN
    SELECT * INTO tenant_rec
    FROM neurondb_tenants
    WHERE tenant_id = neurondb_check_tenant_quota.tenant_id;
    
    IF NOT FOUND THEN
        RETURN jsonb_build_object('error', 'Tenant not found');
    END IF;
    
    vectors_usage := CASE 
        WHEN tenant_rec.max_vectors > 0 THEN
            (tenant_rec.current_vectors::FLOAT / tenant_rec.max_vectors::FLOAT) * 100
        ELSE 0
    END;
    
    storage_usage := CASE 
        WHEN tenant_rec.max_storage_mb > 0 THEN
            (tenant_rec.current_storage_mb::FLOAT / tenant_rec.max_storage_mb::FLOAT) * 100
        ELSE 0
    END;
    
    qps_usage := CASE 
        WHEN tenant_rec.max_qps > 0 THEN
            (tenant_rec.current_qps::FLOAT / tenant_rec.max_qps::FLOAT) * 100
        ELSE 0
    END;
    
    result := jsonb_build_object(
        'tenant_id', tenant_id,
        'vectors', jsonb_build_object(
            'current', tenant_rec.current_vectors,
            'max', tenant_rec.max_vectors,
            'usage_percent', vectors_usage,
            'exceeded', tenant_rec.current_vectors > tenant_rec.max_vectors
        ),
        'storage', jsonb_build_object(
            'current_mb', tenant_rec.current_storage_mb,
            'max_mb', tenant_rec.max_storage_mb,
            'usage_percent', storage_usage,
            'exceeded', tenant_rec.current_storage_mb > tenant_rec.max_storage_mb
        ),
        'qps', jsonb_build_object(
            'current', tenant_rec.current_qps,
            'max', tenant_rec.max_qps,
            'usage_percent', qps_usage,
            'exceeded', tenant_rec.current_qps > tenant_rec.max_qps
        )
    );
    
    RETURN result;
END;
$$;

COMMENT ON FUNCTION neurondb_check_tenant_quota IS 
'Check tenant quota usage and return detailed usage statistics.';

-- ============================================================================
-- TENANT SCHEMA
-- ============================================================================

/*
 * Create tenant management tables
 */
CREATE SEQUENCE IF NOT EXISTS neurondb_tenant_id_seq;

CREATE TABLE IF NOT EXISTS neurondb_tenants (
    tenant_id INTEGER PRIMARY KEY DEFAULT nextval('neurondb_tenant_id_seq'),
    tenant_name TEXT NOT NULL UNIQUE,
    max_vectors BIGINT NOT NULL DEFAULT 1000000,
    max_storage_mb BIGINT NOT NULL DEFAULT 10240,
    max_qps INTEGER NOT NULL DEFAULT 1000,
    current_vectors BIGINT DEFAULT 0,
    current_storage_mb BIGINT DEFAULT 0,
    current_qps INTEGER DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now(),
    status TEXT DEFAULT 'active',
    metadata JSONB
);

CREATE INDEX IF NOT EXISTS idx_tenants_name ON neurondb_tenants(tenant_name);
CREATE INDEX IF NOT EXISTS idx_tenants_status ON neurondb_tenants(status);

COMMENT ON TABLE neurondb_tenants IS 
'Tenant management table with resource quotas and usage tracking.';

-- ============================================================================
-- TENANT QUOTA ENFORCEMENT VIEWS
-- ============================================================================

/*
 * View for tenant quota usage summary
 */
CREATE OR REPLACE VIEW neurondb_tenant_quota_summary AS
SELECT 
    tenant_id,
    tenant_name,
    max_vectors,
    current_vectors,
    ROUND((current_vectors::FLOAT / NULLIF(max_vectors, 0)::FLOAT) * 100, 2) AS vectors_usage_percent,
    max_storage_mb,
    current_storage_mb,
    ROUND((current_storage_mb::FLOAT / NULLIF(max_storage_mb, 0)::FLOAT) * 100, 2) AS storage_usage_percent,
    max_qps,
    current_qps,
    ROUND((current_qps::FLOAT / NULLIF(max_qps, 0)::FLOAT) * 100, 2) AS qps_usage_percent,
    status,
    updated_at
FROM neurondb_tenants
ORDER BY tenant_id;

COMMENT ON VIEW neurondb_tenant_quota_summary IS 
'Summary view of tenant quota usage across all tenants.';



