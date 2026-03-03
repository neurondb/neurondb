-- Example: Row-Level Security (RLS) for multi-tenant ML catalog
-- Run this in a test schema; adapt table names to your schema.

-- 1. Add tenant_id to your tables (if not already present)
-- ALTER TABLE neurondb.ml_models ADD COLUMN IF NOT EXISTS tenant_id uuid DEFAULT current_setting('app.tenant_id', true)::uuid;

-- 2. Enable RLS on the table
-- ALTER TABLE neurondb.ml_models ENABLE ROW LEVEL SECURITY;

-- 3. Policy: users see only their tenant's rows
-- CREATE POLICY tenant_isolation ON neurondb.ml_models
--   USING (tenant_id = current_setting('app.tenant_id', true)::uuid);

-- 4. Policy: inserts must use current tenant
-- CREATE POLICY tenant_insert ON neurondb.ml_models
--   FOR INSERT WITH CHECK (tenant_id = current_setting('app.tenant_id', true)::uuid);

-- 5. In your application, set the tenant at session start (e.g. from JWT):
-- SET app.tenant_id = 'your-tenant-uuid';
