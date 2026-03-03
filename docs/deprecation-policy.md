# Deprecation Policy

This policy defines how NeuronDB handles function deprecation, removal, and breaking changes to ensure predictable API evolution.

## Overview

NeuronDB is committed to maintaining API stability while allowing the codebase to evolve. This policy ensures that users have adequate notice and migration paths when functions are deprecated or removed.

## Deprecation Process

### Step 1: Deprecation Announcement

When a function is marked for deprecation:

1. **Announcement:** Deprecation is announced in release notes
2. **Documentation:** Function documentation is updated with deprecation notices
3. **Warning:** Deprecated functions may emit warnings when used (optional, configurable)
4. **Migration Guide:** Clear migration path to replacement function is provided

### Step 2: Deprecation Period

**Minimum Deprecation Period:** At least one minor release (e.g., if deprecated in 1.5.0, removal no earlier than 1.6.0)

**Extended Period for Stable Functions:** 
- Functions marked as `stable` receive extended deprecation periods (at least 2 minor releases)
- Critical functions may have extended periods (3+ minor releases)

**Deprecation Duration:**
- **Experimental functions:** Minimum 1 minor release
- **Stable functions:** Minimum 2 minor releases
- **Critical/core functions:** Minimum 3 minor releases or longer

### Step 3: Removal

After the deprecation period:
1. Function is removed in a minor or major release
2. Removal is clearly documented in release notes
3. Breaking change warnings are provided
4. Migration guide is updated with final removal notice

## Deprecation Notices

### In Documentation

Deprecated functions are clearly marked:

```markdown
### `deprecated_function()` {#stability-stable}
**⚠️ DEPRECATED:** This function is deprecated and will be removed in version X.Y.0.

**Replacement:** Use `new_function()` instead.

**Deprecated Since:** 1.5.0  
**Removal Target:** 1.7.0

[Original documentation...]
```

### In SQL Comments

Deprecation is documented in SQL function comments:

```sql
COMMENT ON FUNCTION neurondb.deprecated_function IS 
  'Stability: STABLE - DEPRECATED since 1.5.0, will be removed in 1.7.0. Use new_function() instead.';
```

### Runtime Warnings

Deprecated functions may optionally emit warnings (controlled by configuration):

```sql
-- Enable deprecation warnings (default: false)
SET neurondb.warn_on_deprecated = true;

-- Using deprecated function will emit warning
SELECT deprecated_function();  -- WARNING: Function deprecated_function() is deprecated
```

## Types of Deprecations

### Function Removal

A function is removed entirely:

**Example:**
- Old: `neurondb.old_embed(text)` 
- New: `neurondb.embed(text, model)`

**Migration:**
- Clear replacement function provided
- Migration script or guide available
- Parameters mapped to new function

### Function Signature Change

Function parameters change (breaking change):

**Example:**
- Old: `train_model(algorithm, data)`
- New: `train_model(algorithm, data, hyperparameters JSONB)`

**Migration:**
- Old signature marked deprecated
- New signature available
- Both coexist during deprecation period
- Default values provided where possible

### Behavior Change

Function behavior changes significantly:

**Example:**
- Distance calculation algorithm changes
- Default parameter values change

**Migration:**
- Old behavior maintained with compatibility flag
- New behavior becomes default
- Compatibility mode deprecated

### Function Renaming

Function is renamed:

**Example:**
- Old: `neurondb_embed()`
- New: `neurondb.embed()`

**Migration:**
- Old name marked as deprecated alias
- New name available immediately
- Both work during deprecation period

## Emergency Deprecations

In rare cases (security, critical bugs), functions may be deprecated with shorter notice periods:

1. **Security Issues:** Immediate deprecation and removal
2. **Critical Bugs:** Shortened deprecation period (minimum 1 patch release)
3. **Compatibility Issues:** Standard deprecation process

Emergency deprecations are clearly documented with rationale.

## Migration Support

### Migration Guides

Comprehensive migration guides are provided for all deprecations:

1. **What changed:** Clear explanation of the change
2. **Why changed:** Rationale for the deprecation
3. **How to migrate:** Step-by-step migration instructions
4. **Code examples:** Before/after code examples
5. **Common issues:** Known migration issues and solutions

### Compatibility Layers

Where feasible, compatibility layers are provided:

- Deprecated functions may be implemented as wrappers
- Compatibility modes may be available
- Gradual migration supported

### Automated Migration Tools

For complex migrations, automated tools may be provided:

- SQL migration scripts
- Function alias helpers
- Configuration migration utilities

## Versioning and Compatibility

### Version Numbering

- **Major (X.0.0):** May include breaking changes, removal of deprecated functions
- **Minor (X.Y.0):** New features, deprecations announced, functions removed after deprecation period
- **Patch (X.Y.Z):** Bug fixes, security patches, no breaking changes

### Compatibility Guarantees

- **Same Major Version:** Deprecated functions available during deprecation period
- **Next Major Version:** Deprecated functions may be removed
- **Between Major Versions:** Breaking changes documented in migration guide

## Communication Channels

Deprecations are communicated through:

1. **Release Notes:** All deprecations listed in release notes
2. **Documentation:** Function documentation updated with deprecation notices
3. **GitHub Issues:** Major deprecations may be announced via GitHub Issues
4. **Migration Guides:** Dedicated migration guides for complex changes

## Querying Deprecated Functions

You can identify deprecated functions:

```sql
-- Find all deprecated NeuronDB functions
SELECT 
  p.proname AS function_name,
  pg_get_function_identity_arguments(p.oid) AS arguments,
  obj_description(p.oid, 'pg_proc') AS description
FROM pg_proc p
JOIN pg_namespace n ON p.pronamespace = n.oid
WHERE n.nspname = 'neurondb'
  AND obj_description(p.oid, 'pg_proc') LIKE '%DEPRECATED%'
ORDER BY p.proname;
```

## Exceptions and Special Cases

### Experimental Functions

Experimental functions may be removed without following standard deprecation policy:
- Minimum notice: 1 patch release
- May be removed in any minor release
- Users should expect changes

### Internal Functions

Internal functions may be removed without notice:
- No deprecation process required
- Should not be used by external applications

### Configuration Parameters

Configuration parameters follow the same deprecation policy:
- Same minimum notice periods
- Same migration support
- Documented in configuration reference

## Timeline Examples

### Standard Deprecation (Stable Function)

- **1.5.0:** Function deprecated, announcement in release notes
- **1.6.0:** Still available, warnings enabled
- **1.7.0:** Function removed (minimum 2 minor releases)

### Extended Deprecation (Critical Function)

- **1.5.0:** Function deprecated, announcement in release notes
- **1.6.0:** Still available, migration guide provided
- **1.7.0:** Still available, warnings emphasized
- **1.8.0:** Still available (extended period)
- **2.0.0:** Function removed (major version bump)

### Emergency Deprecation (Security Issue)

- **1.5.0:** Security issue discovered, function deprecated immediately
- **1.5.1:** Function removed (next patch release)

## Best Practices for Users

1. **Monitor Release Notes:** Check release notes for deprecation announcements
2. **Plan Migrations:** Plan migrations during deprecation period, not at removal
3. **Test Migration:** Test migration in development before production
4. **Use Migration Guides:** Follow provided migration guides
5. **Enable Warnings:** Enable deprecation warnings in development
6. **Update Dependencies:** Keep NeuronDB updated to avoid surprises

## Feedback and Questions

If you have concerns about a deprecation:
- Review the migration guide
- Check GitHub Issues for discussions
- Open an issue if migration path is unclear
- Provide feedback on the deprecation rationale

## Related Documentation

- [Function Stability Policy](function-stability.md) - Function stability classifications
- [SQL API Reference](sql-api.md) - Complete function reference with deprecation notices
- [API Snapshots](api-snapshots/readme.md) - Versioned API snapshots for reference
- [Release Notes](../whats-new.md) - Deprecation announcements

