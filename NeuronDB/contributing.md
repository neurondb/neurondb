# Contributing to NeuronDB

**For comprehensive contributing guidelines covering all NeuronDB ecosystem components, please see the root [CONTRIBUTING.md](../CONTRIBUTING.md).**

This document provides NeuronDB-specific information for contributors working on the PostgreSQL extension.

## NeuronDB-Specific Guidelines

### C Code Standards
- **Style**: 100% PostgreSQL C coding standards
- **Comments**: Only C-style `/* */` comments
- **Variables**: Declared at start of function (C89/C99 compliance)
- **Formatting**: Tabs for indentation (PostgreSQL standard)
- **Naming**: Prefix all functions with `neurondb_`
- **Headers**: Include copyright and file description

### Build Requirements
- **Zero warnings**: Code must compile with `-Wall -Wextra` clean
- **Zero errors**: All compilation must succeed
- **All platforms**: Test on Linux and macOS
- **PostgreSQL versions**: Support PG 16, 17, 18

### Testing
- Add regression tests in `sql/` with expected output in `expected/`
- Add TAP tests in `t/` for integration testing
- Run `make installcheck` before submitting PR
- Document test cases clearly

### Formatting
```bash
# Format all C files
./scripts/neurondb_format.sh

# Check formatting without modifying files
./scripts/neurondb_format.sh --check

# Show diff of formatting changes
./scripts/neurondb_format.sh --diff
```

## See Also

- [Root CONTRIBUTING.md](../CONTRIBUTING.md) - Complete contributing guidelines
- [NeuronDB README.md](README.md) - Component overview
- [Development Guide](../Docs/development/development-guide.md) - Development setup

