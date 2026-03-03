# Crash Prevention Test Suite

This directory contains systematic tests for crash prevention in NeuronDB.

## Test Categories

### 001_null_parameters.sql
Tests NULL parameter handling across all function types:
- NULL model_id
- NULL table_name
- NULL feature columns
- NULL label columns (where applicable)
- NULL arrays and inputs

### 002_invalid_models.sql
Tests handling of invalid or non-existent models:
- Non-existent model_id
- Negative/zero model_id
- Models with NULL payloads
- Models with corrupted metadata

### 003_spi_failures.sql
Tests SPI (Server Programming Interface) failure scenarios:
- Non-existent tables
- Non-existent columns
- Empty tables
- Tables with all NULL values
- Query execution failures

### 004_memory_contexts.sql
Tests memory context handling under stress:
- Large batch evaluations
- Multiple rapid evaluations
- Nested function calls
- Memory pressure scenarios

### 005_array_bounds.sql
Tests array bounds and dimension validation:
- Empty arrays
- Wrong dimension arrays
- Arrays with NULL elements
- Very large arrays
- Dimension mismatches

## Running the Tests

```bash
# Run all crash prevention tests
psql -d your_database -f tests/sql/crash_prevention/001_null_parameters.sql
psql -d your_database -f tests/sql/crash_prevention/002_invalid_models.sql
# ... etc
```

## Expected Behavior

All tests should:
1. **NOT crash the PostgreSQL server**
2. Return appropriate error messages (not segfaults)
3. Handle errors gracefully
4. Clean up resources properly

## Notes

- Some tests may require specific setup (e.g., existing models, tables)
- Tests are designed to trigger error conditions, not to pass
- The goal is to ensure errors are handled gracefully, not to test normal operation
- Memory context tests may require special monitoring tools

## Related Code

These tests validate the crash prevention patterns implemented in:
- `src/util/neurondb_safe_memory.c` - Safe memory management
- `src/util/neurondb_spi_safe.c` - Safe SPI operations
- `include/neurondb_validation.h` - Input validation macros

