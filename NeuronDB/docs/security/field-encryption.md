# Field-Level Encryption for Vectors

## Overview

NeuronDB provides field-level encryption for sensitive vector data and metadata using AES-256-GCM encryption. This ensures data confidentiality at rest while maintaining query capabilities.

## Features

- AES-256-GCM encryption for vectors at rest
- Transparent encryption/decryption using SQL functions
- Support for per-column encryption keys
- Key rotation capabilities

## Prerequisites

Field-level encryption requires OpenSSL support in the PostgreSQL build. The encryption functions use OpenSSL's EVP API for AES-256-GCM.

## Configuration

Enable encryption:

```sql
SET neurondb.encryption_enabled = true;
```

## Usage

### Encrypting Vectors

```sql
-- Encrypt a vector column
CREATE TABLE sensitive_documents (
    id SERIAL PRIMARY KEY,
    content TEXT,
    embedding vector(384),
    encrypted_embedding bytea
);

-- Insert with encryption
INSERT INTO sensitive_documents (content, encrypted_embedding)
VALUES (
    'Sensitive content',
    encrypt_vector(
        '[0.1, 0.2, 0.3, ...]'::vector(384),
        'my-encryption-key-12345'
    )
);
```

### Decrypting Vectors

```sql
-- Decrypt for use in queries
SELECT 
    id,
    content,
    decrypt_vector(encrypted_embedding, 'my-encryption-key-12345') AS embedding
FROM sensitive_documents;
```

### Key Rotation

```sql
-- Rotate encryption key for a column
SELECT rotate_encryption_key(
    'sensitive_documents',
    'encrypted_embedding',
    'old-encryption-key',
    'new-encryption-key'
);
```

## Storage Format

Encrypted vectors are stored as BYTEA containing the `EncryptedVector` structure:
- Encryption IV (12 bytes)
- Authentication tag (16 bytes)
- Dimension information
- Encrypted ciphertext

## Security Considerations

1. **Key Management**: Store encryption keys securely. Consider using PostgreSQL's key management extensions or external key management systems.

2. **Performance**: Encryption/decryption adds latency. Use encryption selectively for sensitive data only.

3. **Key Rotation**: Regularly rotate encryption keys for enhanced security.

4. **Backup**: Encrypted data in backups remains encrypted. Ensure backup keys are also securely managed.

## Limitations

- Encrypted vectors cannot be used directly in vector similarity searches. Decrypt before querying.
- Key management is the responsibility of the database administrator.
- Encryption functions require OpenSSL support.

## Related Topics

- [Security Overview](overview.md)
- [RLS for Embeddings](rls-embeddings.md)
- [Audit Logging](audit-logging.md)


