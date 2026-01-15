/*-------------------------------------------------------------------------
 *
 * neuronagent_compliance_schema.sql
 *    Database schema for compliance features
 *
 * Creates tables for enhanced audit logging and compliance reporting.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/sql/neuronagent_compliance_schema.sql
 *
 *-------------------------------------------------------------------------
 */

-- Enhanced audit logs table for compliance
CREATE TABLE IF NOT EXISTS neurondb_agent.audit_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    event_type TEXT NOT NULL,
    user_id TEXT,
    resource_type TEXT,
    resource_id UUID,
    action TEXT NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    metadata JSONB DEFAULT '{}',
    ip_address INET,
    user_agent TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_audit_logs_event_type ON neurondb_agent.audit_logs(event_type);
CREATE INDEX IF NOT EXISTS idx_audit_logs_user_id ON neurondb_agent.audit_logs(user_id);
CREATE INDEX IF NOT EXISTS idx_audit_logs_timestamp ON neurondb_agent.audit_logs(timestamp);
CREATE INDEX IF NOT EXISTS idx_audit_logs_resource ON neurondb_agent.audit_logs(resource_type, resource_id);

COMMENT ON TABLE neurondb_agent.audit_logs IS 'Enhanced audit logs for SOC2, ISO27001, and GDPR compliance';

-- Compliance reports table
CREATE TABLE IF NOT EXISTS neurondb_agent.compliance_reports (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    report_type TEXT NOT NULL CHECK (report_type IN ('soc2', 'iso27001', 'gdpr')),
    start_date TIMESTAMPTZ NOT NULL,
    end_date TIMESTAMPTZ NOT NULL,
    metrics JSONB NOT NULL,
    generated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    generated_by TEXT
);

CREATE INDEX IF NOT EXISTS idx_compliance_reports_type ON neurondb_agent.compliance_reports(report_type);
CREATE INDEX IF NOT EXISTS idx_compliance_reports_generated_at ON neurondb_agent.compliance_reports(generated_at);

COMMENT ON TABLE neurondb_agent.compliance_reports IS 'Compliance reports for SOC2, ISO27001, and GDPR';

