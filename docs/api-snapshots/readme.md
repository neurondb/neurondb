# SQL API Reference Snapshots

This directory contains versioned snapshots of the NeuronDB SQL API reference documentation. Each snapshot represents the complete API surface for a specific release, providing a historical record of functions, operators, types, and configuration parameters.

## Purpose

API snapshots serve several important purposes:

1. **Historical Reference:** Understand what functions were available in previous versions
2. **Migration Planning:** Compare API changes between versions
3. **Offline Reference:** Access API documentation without connecting to a database
4. **Version-Specific Documentation:** Reference exact API for deployed versions
5. **Compatibility Checking:** Verify function availability in specific versions

## Snapshot Structure

Each snapshot is a self-contained markdown document or directory containing:

- Complete function reference with signatures
- Parameter descriptions and types
- Return type documentation
- Function stability classifications
- Deprecation notices (if applicable)
- Code examples
- Configuration parameters (GUCs)
- Operators and type definitions

## Naming Convention

Snapshots are named according to the NeuronDB version:

- `v1.0.0.md` - API snapshot for version 1.0.0
- `v1.5.0.md` - API snapshot for version 1.5.0
- `v2.0.0/` - Directory for version 2.0.0 (if snapshot is large)

Format: `v{MAJOR}.{MINOR}.{PATCH}.md` or `v{MAJOR}.{MINOR}.{PATCH}/`

## Current Snapshots

Snapshots are generated at each release and included in the repository. The latest snapshot represents the current stable release.

### Available Snapshots

*Note: Snapshot files will be added during the release process. This section will list all available snapshots once they are generated.*

Example structure (when snapshots are available):
- `v1.0.0.md` - Initial release API snapshot
- `v1.1.0.md` - Minor release with new functions
- `v1.5.0.md` - Feature release snapshot
- `v2.0.0.md` - Major release snapshot

## Generating Snapshots

Snapshots are automatically generated during the release process. The generation process:

1. Extracts all functions from the PostgreSQL catalog
2. Includes function signatures, parameters, and return types
3. Incorporates stability classifications
4. Includes deprecation notices
5. Documents configuration parameters
6. Formats as markdown

### Manual Generation

To generate a snapshot manually:

```bash
# Connect to database with NeuronDB extension
psql -d neurondb -f scripts/generate_api_snapshot.sql > docs/api-snapshots/vX.Y.Z.md

# Or use the automated script (when available)
./scripts/generate_api_snapshot.sh vX.Y.Z
```

## Using Snapshots

### For Development

- Reference specific version API during development
- Understand API evolution between versions
- Plan migrations using historical snapshots

### For Migration

- Compare snapshots between versions to identify changes
- Find deprecated functions and their replacements
- Understand breaking changes

### For Documentation

- Link to version-specific API documentation
- Provide offline API reference
- Support version-specific user documentation

## Snapshot Format

Each snapshot follows a standard structure:

```markdown
# NeuronDB SQL API Reference - Version X.Y.Z

## Overview
[Brief overview of the API in this version]

## Function Stability
[Reference to stability policy and classifications]

## Functions by Category

### Vector Operations
[List of vector functions with signatures]

### Machine Learning
[List of ML functions with signatures]

### Configuration Parameters
[List of GUC parameters]

## Deprecations
[List of deprecated functions in this version]

## Changelog
[Changes from previous version]
```

## Version Compatibility

- Snapshots correspond to NeuronDB extension versions
- Each snapshot documents the API available at that version
- Snapshot format may evolve, but historical snapshots are preserved

## Latest API Reference

For the latest API reference, see:
- [Current SQL API Reference](../sql-api.md) - Latest development version
- [Official Documentation](https://www.neurondb.ai/docs/api) - Official site with interactive API reference

## Snapshot Maintenance

### Adding New Snapshots

1. Generate snapshot during release process
2. Name according to version number
3. Add entry to this README
4. Commit with release

### Updating This README

Update the "Available Snapshots" section when new snapshots are added.

## Related Documentation

- [SQL API Reference](../sql-api.md) - Current API documentation
- [Function Stability Policy](../function-stability.md) - Stability classifications
- [Deprecation Policy](../deprecation-policy.md) - Deprecation process
- [Release Notes](../whats-new.md) - Version release information

## Notes

- Snapshots are generated from the actual extension installation
- Snapshots represent the API at the time of release
- For the most current API, refer to the latest snapshot or current documentation
- Snapshots may be updated to fix documentation errors without changing version numbers

