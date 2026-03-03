# NeuronDB Testing Foundation

This document describes the testing setup for NeuronDB and how to run tests. The goal is to maintain and grow test coverage (target: 40%+ for the audit roadmap, 60%+ later, 80% at launch).

## Test Layout

- **`t/`** – TAP-style Perl tests (PostgresNode, etc.). Numbered `001_*.t` … `029_*.t`. Cover extension load, vectors, distances, ML, indexes, workers, edge cases.
- **`tests/sql/`** – Standalone SQL test scripts run by `run_test.py`:
  - **`basic/`** – Basic extension and feature tests
  - **`negative/`** – Negative/error-path tests
  - **`crash_prevention/`** – NULL handling, invalid models, SPI failures, overflow, concurrency
  - **`security/`** – Security tests (e.g. SQL injection resistance for identifiers)
  - **`utils/`** – Schema and session helpers
  - **`perf/`** – Performance-related scripts

## Running Tests

- **SQL tests**: Use the project test runner, e.g. `./run_test.py` (see `run_comprehensive_plan.sh` or Makefile targets if available). Or run a single file:
  ```bash
  psql -v ON_ERROR_STOP=1 -f tests/sql/security/001_sql_injection_identifiers.sql
  ```
- **TAP tests**: Follow `NeuronDB/t/README.md` and project INSTALL/contributing docs (e.g. `make installcheck` or `prove`).

## Security Tests

- **`tests/sql/security/001_sql_injection_identifiers.sql`** – Checks that table/column names passed to `neurondb.train()` are safely quoted so that SQL fragments in identifiers cannot be executed (e.g. no `DROP TABLE` via malicious table name).

## Coverage Target

- **Current roadmap**: Establish a baseline and aim for **40% code coverage** by the end of the testing foundation phase.
- Increase to 60% in later quality phases and 80% at launch readiness.
- Add SQL injection fuzz tests, algorithm correctness tests, and edge-case tests as the suite grows.

## Adding New Tests

- **SQL**: Add `.sql` under the appropriate `tests/sql/` subdirectory; use `\set ON_ERROR_STOP on` and deterministic assertions.
- **TAP**: Add or extend numbered `.t` files in `t/` using the existing PostgresNode/TapTest/NeuronDB helpers; keep tests deterministic and focused.
