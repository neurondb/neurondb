# Security

Comprehensive security and governance features including Row-Level Security (RLS) for embeddings, field-level encryption, and audit logging for ML inference and RAG operations.

## Overview

NeuronDB provides enterprise-grade security features for protecting vector data, ensuring multi-tenant isolation, and maintaining compliance through comprehensive audit logging.

## Row-Level Security (RLS) for Embeddings

Enhanced RLS support for embedding-specific access patterns with integration into vector index scans.

```sql
-- Enable RLS on table
ALTER TABLE documents ENABLE ROW LEVEL SECURITY;

-- Create RLS policy
CREATE POLICY documents_policy ON documents
FOR SELECT USING (user_id = current_user_id());

-- Enable embedding RLS
SET neurondb.rls_embeddings_enabled = true;
SELECT enable_embedding_rls('documents');
```

**Learn more:** [RLS for Embeddings](rls-embeddings.md)

## Field-Level Encryption

AES-256-GCM encryption for sensitive vector data and metadata at rest.

```sql
-- Encrypt vector data
INSERT INTO documents (content, encrypted_embedding)
VALUES (
    'Sensitive content',
    encrypt_vector(embedding, 'encryption_key')
);

-- Decrypt for querying
SELECT decrypt_vector(encrypted_embedding, 'encryption_key') AS embedding
FROM documents;
```

**Learn more:** [Field-Level Encryption](field-encryption.md)

## Audit Logging

Comprehensive audit logging for ML inference and RAG operations for compliance and security.

```sql
-- Enable audit logging
SET neurondb.audit_ml_enabled = true;
SET neurondb.audit_rag_enabled = true;

-- Query audit logs
SELECT * FROM query_audit_log(
    'ml_inference',
    start_time := '2024-01-01'::timestamptz,
    user_id := 'admin'
);
```

**Learn more:** [Audit Logging](audit-logging.md)

## Differential Privacy

Add noise to protect individual records:

```sql
-- Apply differential privacy
SELECT differential_privacy_add_noise(
    embedding,
    0.1  -- epsilon (privacy parameter)
) AS private_embedding
FROM documents;
```

## Configuration

All security features are opt-in and configurable via GUC parameters:

- `neurondb.rls_embeddings_enabled` - Enable RLS for embeddings (default: false)
- `neurondb.encryption_enabled` - Enable field encryption (default: false)
- `neurondb.audit_ml_enabled` - Enable ML audit logging (default: false)
- `neurondb.audit_rag_enabled` - Enable RAG audit logging (default: false)
- `neurondb.audit_retention_days` - Audit log retention period (default: 365)

## Compliance

NeuronDB security features support compliance with:

- **GDPR**: User tracking, data access logging, retention policies
- **HIPAA**: Comprehensive audit trails, access control
- **SOC 2**: Audit logging, access control, encryption

## Best Practices

1. **Access Control**: Use RLS policies for multi-tenant isolation
2. **Encryption**: Encrypt sensitive vectors at rest
3. **Auditing**: Enable audit logging for compliance requirements
4. **Key Management**: Secure encryption keys using key management systems
5. **Monitoring**: Regularly review audit logs for security incidents

## Related Topics

- [RLS for Embeddings](rls-embeddings.md) - Row-level security for vectors
- [Field-Level Encryption](field-encryption.md) - Encryption for sensitive data
- [Audit Logging](audit-logging.md) - Compliance and security logging
- [Configuration](../configuration.md) - Security configuration
- [Multi-Tenancy](../configuration.md) - Tenant isolation

