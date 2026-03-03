# Function Stability Policy

NeuronDB classifies all SQL functions, operators, types, and configuration parameters according to their stability level. This policy helps users understand which APIs are safe to use in production and which may change.

## Stability Levels

### Stable

**Definition:** Functions marked as `stable` are intended for general use and guaranteed to maintain backward compatibility across minor and patch releases. The API contract (signature, behavior, and return types) will not change in breaking ways.

**Guarantees:**
- Function signatures (parameter names, types, order) remain constant
- Return types and structures remain constant
- Behavior and semantics remain consistent
- Performance characteristics may improve but won't regress significantly
- Deprecation follows the [Deprecation Policy](deprecation-policy.md) (minimum notice period)

**Use Case:** Safe for production use. Recommended for all production applications.

**Examples:**
- Core vector operations (`<->`, `<=>`, `<#>`)
- Basic embedding functions (`neurondb_embed`, `neurondb_embed_batch`)
- Standard distance metrics
- Primary ML training functions
- Core configuration parameters

### Experimental

**Definition:** Functions marked as `experimental` are available for testing and evaluation but may change or be removed without following the standard deprecation policy. These are new features under active development.

**Characteristics:**
- API may change in any release (minor or patch)
- Function signatures may be modified
- Behavior may evolve based on feedback
- Performance characteristics may vary
- May be promoted to `stable` in future releases
- May be removed without deprecation notice

**Use Case:** Suitable for development, testing, and evaluation. Not recommended for production use without careful consideration.

**Examples:**
- New algorithms in early development
- Advanced optimization features
- Beta GPU features
- Cutting-edge research implementations

### Internal

**Definition:** Functions marked as `internal` are implementation details and should not be used by external applications. These are subject to change or removal at any time without notice.

**Characteristics:**
- No API stability guarantees
- May be removed without notice
- Implementation details may change
- Not documented for external use
- Reserved for internal extension operations

**Use Case:** Should not be used by external applications. These are reserved for internal extension functionality.

**Examples:**
- Internal helper functions
- Debug utilities
- Internal type conversion functions
- Extension maintenance functions

## Identifying Function Stability

### In Documentation

All functions are clearly marked with their stability level:

```markdown
### `function_name()` {#stability-stable}
Generate embeddings using configured LLM providers.

**Stability:** Stable  
**Parameters:** ...  
**Returns:** ...
```

### In SQL Comments

Function stability is documented in SQL comments using PostgreSQL's `COMMENT ON FUNCTION`:

```sql
COMMENT ON FUNCTION neurondb.embed IS 
  'Stability: STABLE - Generate embeddings using configured LLM providers';
```

### Querying Stability Information

You can query function stability information from the PostgreSQL catalog:

```sql
-- Get stability information for all NeuronDB functions
SELECT 
  p.proname AS function_name,
  pg_get_function_identity_arguments(p.oid) AS arguments,
  obj_description(p.oid, 'pg_proc') AS description
FROM pg_proc p
JOIN pg_namespace n ON p.pronamespace = n.oid
WHERE n.nspname = 'neurondb'
  AND obj_description(p.oid, 'pg_proc') LIKE '%Stability:%'
ORDER BY p.proname;
```

## Stability Promotion Process

### Experimental → Stable

1. Function must be available in experimental state for at least one minor release
2. API must be stable and well-tested
3. Documentation must be complete
4. No breaking changes planned
5. Community feedback considered
6. Announcement in release notes

### Internal → Public

Internal functions are generally not promoted. If needed, they will first move to `experimental`, then to `stable` after the standard promotion process.

## Migration Between Stability Levels

When a function moves between stability levels:

1. **Experimental → Stable:** No action required. The function API remains the same.
2. **Stable → Deprecated:** Follows the [Deprecation Policy](deprecation-policy.md)
3. **Any → Internal:** Functions moved to internal status should be avoided in new code

## Configuration Parameters

Configuration parameters (GUCs) follow the same stability classifications:

- **Stable:** Parameters safe for production use with stable behavior
- **Experimental:** Parameters that may change or be removed
- **Internal:** Parameters reserved for internal use

## Breaking Changes Policy

### Stable Functions

Breaking changes to stable functions follow the deprecation process:
- Deprecation announced in release notes
- Minimum notice period before removal
- Migration path provided

### Experimental Functions

Experimental functions may have breaking changes in any release. Check release notes for changes.

### Internal Functions

Internal functions may change at any time without notice.

## Recommendations

1. **Production Applications:** Use only `stable` functions
2. **Early Adopters:** Can use `experimental` functions with awareness of potential changes
3. **All Applications:** Never use `internal` functions

## Version Compatibility

Stability guarantees apply within major versions:
- **Major version (X.0.0):** May include breaking changes with migration guide
- **Minor version (X.Y.0):** Stable functions remain stable, experimental may change
- **Patch version (X.Y.Z):** No breaking changes to stable functions

## Questions and Support

If you have questions about function stability:
- Check this policy document
- Review function documentation
- Check release notes for changes
- Report issues or ask questions via GitHub Issues

## Related Documentation

- [Deprecation Policy](deprecation-policy.md) - How functions are deprecated and removed
- [SQL API Reference](sql-api.md) - Complete function reference
- [API Snapshots](api-snapshots/readme.md) - Versioned API snapshots

