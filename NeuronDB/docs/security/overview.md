# Security

Encryption, differential privacy, and Row-Level Security (RLS) integration.

## Encryption

NeuronDB supports encryption for sensitive data:

```sql
-- Encrypt vector data
SELECT encrypt_vector(embedding, 'encryption_key') AS encrypted
FROM documents;
```

## Row-Level Security (RLS)

Integration with PostgreSQL RLS:

```sql
-- Enable RLS on table
ALTER TABLE documents ENABLE ROW LEVEL SECURITY;

-- Create RLS policy
CREATE POLICY documents_policy ON documents
FOR SELECT USING (user_id = current_user_id());
```

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

## Learn More

For detailed documentation on security features, encryption, privacy protection, and access control, visit:

**[Security Documentation](https://neurondb.ai/docs/security/overview/)**

## Related Topics

- [Configuration](../configuration.md) - Security configuration
- [Multi-Tenancy](../configuration.md) - Tenant isolation

