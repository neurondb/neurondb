# Audit Logging for ML and RAG Operations

## Overview

NeuronDB provides comprehensive audit logging for ML inference and RAG operations, enabling compliance monitoring, security analysis, and usage tracking.

## Features

- Audit logging for ML model inference calls
- Audit logging for RAG retrieve/generate operations
- Configurable retention periods
- Query interface for audit logs
- Input/output hashing for integrity verification

## Configuration

Enable audit logging:

```sql
-- Enable ML inference audit logging
SET neurondb.audit_ml_enabled = true;

-- Enable RAG operation audit logging
SET neurondb.audit_rag_enabled = true;

-- Set retention period (days)
SET neurondb.audit_retention_days = 365;
```

## ML Inference Audit Logging

### Automatic Logging

When `neurondb.audit_ml_enabled` is enabled, ML inference operations are automatically logged.

### Manual Logging

```sql
-- Log ML inference operation
SELECT log_ml_inference(
    model_id := 1,
    operation_type := 'predict',
    input_hash := encode(digest(input_data::text, 'sha256'), 'hex'),
    output_hash := encode(digest(output_data::text, 'sha256'), 'hex'),
    metadata := '{"batch_size": 100, "latency_ms": 45}'::jsonb
);
```

### Querying ML Audit Logs

```sql
-- Query ML inference audit logs
SELECT * FROM query_audit_log(
    'ml_inference',
    start_time := '2024-01-01'::timestamptz,
    end_time := '2024-12-31'::timestamptz,
    user_id := 'admin',
    operation_type := 'predict'
);
```

## RAG Operation Audit Logging

### Automatic Logging

When `neurondb.audit_rag_enabled` is enabled, RAG operations are automatically logged.

### Manual Logging

```sql
-- Log RAG operation
SELECT log_rag_operation(
    pipeline_name := 'documents_rag',
    operation_type := 'retrieve',
    query_hash := encode(digest('What is machine learning?'::text, 'sha256'), 'hex'),
    result_count := 5,
    metadata := '{"k": 5, "rerank": true}'::jsonb
);
```

### Querying RAG Audit Logs

```sql
-- Query RAG operation audit logs
SELECT * FROM query_audit_log(
    'rag_operation',
    start_time := '2024-01-01'::timestamptz,
    end_time := '2024-12-31'::timestamptz,
    user_id := 'user123',
    operation_type := 'generate'
);
```

## Audit Log Schema

### ML Inference Audit Log

```sql
CREATE TABLE neurondb.ml_inference_audit_log (
    audit_id BIGSERIAL PRIMARY KEY,
    model_id INTEGER,
    operation_type TEXT NOT NULL,
    user_id TEXT DEFAULT CURRENT_USER,
    input_hash TEXT,
    output_hash TEXT,
    metadata JSONB,
    timestamp TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);
```

### RAG Operation Audit Log

```sql
CREATE TABLE neurondb.rag_operation_audit_log (
    audit_id BIGSERIAL PRIMARY KEY,
    pipeline_name TEXT NOT NULL,
    operation_type TEXT NOT NULL,
    user_id TEXT DEFAULT CURRENT_USER,
    query_hash TEXT,
    result_count INTEGER DEFAULT 0,
    metadata JSONB,
    timestamp TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);
```

## Compliance Considerations

### GDPR

- User IDs are logged for data access tracking
- Input/output hashes enable integrity verification without storing sensitive data
- Retention periods can be configured per compliance requirements

### HIPAA

- Audit logs track all access to PHI-related ML models
- Query hashes enable compliance reporting
- Timestamps enable audit trail reconstruction

### SOC 2

- Comprehensive logging of all ML/RAG operations
- User attribution for all operations
- Configurable retention for audit requirements

## Best Practices

1. **Retention Management**: Regularly archive or delete old audit logs based on retention policies.

2. **Indexing**: Audit tables are indexed for efficient querying. Monitor table sizes and partition if needed.

3. **Performance**: Audit logging is asynchronous where possible to minimize impact on operations.

4. **Monitoring**: Set up alerts for unusual patterns in audit logs (e.g., excessive access, failed operations).

## Log Rotation

Audit logs should be periodically rotated or archived based on retention policies:

```sql
-- Delete logs older than retention period
DELETE FROM neurondb.ml_inference_audit_log
WHERE timestamp < CURRENT_TIMESTAMP - (neurondb.audit_retention_days || ' days')::interval;

DELETE FROM neurondb.rag_operation_audit_log
WHERE timestamp < CURRENT_TIMESTAMP - (neurondb.audit_retention_days || ' days')::interval;
```

## Related Topics

- [Security Overview](overview.md)
- [RLS for Embeddings](rls-embeddings.md)
- [Field-Level Encryption](field-encryption.md)



