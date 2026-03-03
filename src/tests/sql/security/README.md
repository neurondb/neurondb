# Security SQL Tests

This directory contains SQL tests that verify security properties of NeuronDB, in particular resistance to SQL injection via identifier and literal handling.

## Tests

- **001_sql_injection_identifiers.sql** – Ensures that table and column names passed to `neurondb.train()` (and similar APIs) are safely quoted. Attempts to pass strings that look like SQL fragments (e.g. `x; DROP TABLE victim; --`) must not result in execution of that SQL; the victim table must still exist after the call.

## Running

Run against a running Postgres instance with NeuronDB installed, for example:

```bash
psql -v ON_ERROR_STOP=1 -f NeuronDB/tests/sql/security/001_sql_injection_identifiers.sql
```

Or use the project's standard test runner if it executes `tests/sql/**/*.sql`.
