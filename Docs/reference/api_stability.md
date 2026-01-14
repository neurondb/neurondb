# API Stability Policy

NeuronDB API versioning, stability guarantees, deprecation policy, and compatibility matrix.

## API Stability Levels

NeuronDB functions and operators are classified into three stability levels:

### Stable APIs

**Guarantee:** Backward-compatible across minor and patch versions. Breaking changes only in major versions.

**Includes:**
- Core vector operators: `<->`, `<=>`, `<#>`
- Standard distance functions
- Basic embedding functions: `embed_text`, `embed_text_batch`
- Core indexing: `CREATE INDEX ... USING hnsw/ivf`
- Essential configuration functions

**Examples:**
```sql
-- Stable: Vector distance operators
SELECT embedding <=> query_vector;

-- Stable: Basic embedding
SELECT embed_text('text', 'model');

-- Stable: Index creation
CREATE INDEX idx ON table USING hnsw (vector vector_cosine_ops);
```

### Experimental APIs

**Guarantee:** May change in minor versions. Use with caution in production.

**Includes:**
- Advanced ML algorithms
- GPU-specific functions
- RAG pipeline functions (some)
- Hybrid search functions
- New features in recent releases

**Examples:**
```sql
-- Experimental: GPU functions
SELECT vector_l2_distance_gpu(a, b);

-- Experimental: Advanced RAG
SELECT * FROM neurondb.rag_evaluate(...);
```

### Deprecated APIs

**Guarantee:** Will be removed in next major version. Migration path provided.

**Includes:**
- Functions replaced by better alternatives
- Functions with security issues
- Functions with performance problems

**Current deprecated functions:** None in v1.0.0

## Version Compatibility

### Semantic Versioning

NeuronDB follows [Semantic Versioning](https://semver.org/):
- **MAJOR** (X.0.0): Breaking changes
- **MINOR** (X.Y.0): New features, backward-compatible
- **PATCH** (X.Y.Z): Bug fixes, backward-compatible

### Compatibility Matrix

| From Version | To Version | Compatibility | Notes |
|--------------|------------|---------------|-------|
| 1.0.0 | 1.0.1 | ✅ Full | Patch release, bug fixes only |
| 1.0.x | 1.1.0 | ✅ Full | Minor release, new features added |
| 1.x.x | 2.0.0 | ⚠️ Breaking | Major release, may require migration |

### PostgreSQL Version Compatibility

| NeuronDB Version | PostgreSQL 16 | PostgreSQL 17 | PostgreSQL 18 |
|------------------|---------------|---------------|---------------|
| 1.0.x | ✅ Supported | ✅ Supported | ✅ Supported |
| 1.1.x | ✅ Supported | ✅ Supported | ✅ Supported |
| 2.0.x | ⚠️ TBD | ✅ Supported | ✅ Supported |

*Note: Future major versions may drop support for older PostgreSQL versions.*

## Deprecation Policy

### Deprecation Process

1. **Announcement:** Deprecated in minor release with notice
2. **Warning period:** At least one minor version with warnings
3. **Removal:** Removed in next major version
4. **Migration guide:** Provided for all deprecated features

### Deprecation Timeline

**Example:**
- **v1.1.0:** Function `old_function()` marked as deprecated
- **v1.2.0:** Function still works, but logs warnings
- **v1.3.0:** Function still works, but logs warnings
- **v2.0.0:** Function removed, use `new_function()` instead

### Migration Support

Deprecated functions include:
- Clear deprecation notices in documentation
- Migration examples
- Compatibility shims (when possible)
- Support during transition period

## Breaking Changes

### What Constitutes Breaking Changes

**Breaking changes (major version only):**
- Removing functions or operators
- Changing function signatures (parameters, return types)
- Changing operator behavior
- Removing configuration options
- Changing default behavior significantly

**Not breaking changes (minor/patch versions):**
- Adding new functions
- Adding optional parameters
- Performance improvements
- Bug fixes that change behavior (to correct behavior)
- New configuration options

### Breaking Changes in v2.0.0 (Planned)

**None currently planned.** This section will be updated as plans develop.

## API Surface

### Function Count by Category

| Category | Stable | Experimental | Total |
|----------|--------|--------------|-------|
| Vector Operations | 25 | 15 | 40 |
| Distance Functions | 8 | 7 | 15 |
| Embedding Functions | 5 | 5 | 10 |
| Index Functions | 10 | 10 | 20 |
| ML Functions | 50 | 150 | 200 |
| RAG Functions | 10 | 20 | 30 |
| Configuration | 8 | 2 | 10 |
| **Total** | **116** | **209** | **325** |

*Note: Additional functions exist but are internal or administrative.*

### Stable API List

**Vector Operators:**
- `<->` (L2 distance)
- `<=>` (Cosine distance)
- `<#>` (Inner product)
- `<~>` (Manhattan distance)
- `<@>` (Hamming distance)
- `<%>` (Jaccard distance)

**Core Functions:**
- `embed_text(text, text) → vector`
- `embed_text_batch(text[], text) → vector[]`
- `neurondb.version() → jsonb`
- `neurondb_gpu_info() → table`

**Index Creation:**
- `CREATE INDEX ... USING hnsw`
- `CREATE INDEX ... USING ivf`

**Configuration:**
- `set_llm_config(provider, api_key, endpoint)`
- `get_llm_config()`

*See [Top 20 Functions](top_functions.md) for commonly used stable APIs.*

## Extension Upgrade Path

### Minor/Patch Upgrades

**Process:**
1. Backup database
2. Install new extension version
3. Run: `ALTER EXTENSION neurondb UPDATE;`
4. Verify functionality

**Compatibility:** All stable APIs remain compatible.

### Major Upgrades

**Process:**
1. Review migration guide
2. Backup database
3. Test upgrade on staging
4. Install new extension version
5. Run migration scripts (if provided)
6. Update application code (if needed)
7. Verify functionality

**Compatibility:** May require code changes for deprecated/removed APIs.

## Best Practices

### Using Stable APIs

**Recommended for:**
- Production applications
- Long-term projects
- Applications requiring stability

```sql
-- Use stable APIs
SELECT embedding <=> query_vector;  -- Stable
SELECT embed_text('text', 'model');  -- Stable
```

### Using Experimental APIs

**Recommended for:**
- Prototyping
- Testing new features
- Applications that can adapt to changes

```sql
-- Use experimental APIs with caution
SELECT vector_l2_distance_gpu(a, b);  -- Experimental
-- Monitor for changes in minor versions
```

### Handling Deprecations

**When a function is deprecated:**
1. Review migration guide
2. Update code to use replacement
3. Test thoroughly
4. Deploy before next major version

## Version History

### v1.0.0 (Current)

- **Stable APIs:** 116 functions
- **Experimental APIs:** 209 functions
- **Deprecated APIs:** 0 functions
- **Breaking changes:** None (initial release)

### Future Versions

**v1.1.0 (Planned):**
- New experimental features
- No breaking changes
- No deprecations planned

**v2.0.0 (Future):**
- Potential breaking changes
- Deprecations removed
- Migration guide provided

## Support and Compatibility

### Supported Versions

- **Current:** v1.0.x (full support)
- **Previous:** None (first release)
- **Security:** All versions receive security updates

### Compatibility Guarantees

- **Stable APIs:** Compatible across minor/patch versions
- **Experimental APIs:** May change in minor versions
- **Data compatibility:** Vector types compatible across versions
- **Index compatibility:** Indexes compatible within major version

## Related Documentation

- [Top 20 Functions](top_functions.md) - Most commonly used functions
- [SQL API Reference](../../NeuronDB/docs/sql-api.md) - Complete API documentation
- [CHANGELOG](../../CHANGELOG.md) - Version history and changes
- [Upgrade Guide](../../NeuronDB/docs/operations/playbook.md#upgrades) - Upgrade procedures

