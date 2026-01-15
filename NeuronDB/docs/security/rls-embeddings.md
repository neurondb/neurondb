# Row-Level Security for Embeddings

## Overview

NeuronDB extends PostgreSQL's Row-Level Security (RLS) to support embedding-specific access patterns and integration with vector index scans. This ensures multi-tenant isolation and fine-grained access control for vector data.

## Features

- Embedding-aware RLS policies (filter by metadata, tenant_id, classification labels)
- RLS integration with vector index scans (HNSW, IVF)
- Performance optimization for RLS checks during ANN searches
- SQL helper functions for policy management

## Configuration

Enable RLS for embeddings:

```sql
SET neurondb.rls_embeddings_enabled = true;
```

## Usage

### Enable RLS on Table

```sql
-- Enable RLS on documents table
ALTER TABLE documents ENABLE ROW LEVEL SECURITY;

-- Create RLS policy
CREATE POLICY documents_policy ON documents
FOR SELECT USING (user_id = current_user_id());
```

### Using Helper Functions

```sql
-- Enable embedding RLS with tenant isolation
SELECT enable_embedding_rls('documents');

-- Create custom embedding RLS policy
SELECT create_embedding_rls_policy(
    'documents',
    'tenant_isolation_policy',
    'tenant_id = current_setting(''app.tenant_id'')::integer'
);
```

### Example: Multi-Tenant Isolation

```sql
-- Create documents table with tenant_id
CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
    tenant_id INTEGER NOT NULL,
    content TEXT,
    embedding vector(384)
);

-- Enable RLS
ALTER TABLE documents ENABLE ROW LEVEL SECURITY;

-- Create policy for tenant isolation
CREATE POLICY tenant_isolation ON documents
FOR SELECT USING (tenant_id = current_setting('app.tenant_id', true)::integer);

-- Set tenant context
SET app.tenant_id = 123;

-- Query - only returns documents for tenant 123
SELECT * FROM documents 
ORDER BY embedding <=> query_vector
LIMIT 10;
```

## Best Practices

1. **Performance**: RLS checks add overhead to vector searches. Use indexed columns in policy expressions.

2. **Tenant Isolation**: Use session variables or role-based policies for multi-tenant scenarios.

3. **Index Integration**: RLS policies are enforced during index scans, ensuring efficient filtering.

4. **Policy Design**: Keep policy expressions simple and use indexed columns when possible.

## Related Topics

- [Security Overview](overview.md)
- [Field-Level Encryption](field-encryption.md)
- [Audit Logging](audit-logging.md)



