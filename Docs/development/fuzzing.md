# Fuzzing NeuronDB

## Go fuzz tests (NeuronAgent)

NeuronAgent includes a Go fuzz target for SQL query validation:

```bash
cd NeuronAgent
go test -fuzz=FuzzValidateSQLQuery -fuzztime=30s ./internal/validation/
```

This exercises `ValidateSQLQuery` with random inputs to find panics or invalid states. Run longer (e.g. `-fuzztime=5m`) in CI for regression coverage.

## C / libFuzzer (optional)

NeuronDB C code runs inside PostgreSQL; most functions depend on `MemoryContext`, SPI, and backend state. A standalone libFuzzer harness would require:

1. **Stub backend**: Build a minimal stub that provides `palloc`, `quote_literal`, etc., or link against a minimal PG build.
2. **Harness**: One C file that includes the function under test (or a copy that does not use PG macros), e.g. a string-escaping helper used before calling `quote_literal`.
3. **Build**: `clang -fsanitize=fuzzer,address -g -O2 harness.c -o fuzz_harness`.

The extension’s `neurondb_quote_literal_cstr` wraps PostgreSQL’s `quote_literal()` and is not easy to fuzz in isolation. Prefer:

- **ASan/UBSan** on the full extension (see [asan-ubsan.md](asan-ubsan.md)) and run the SQL test suite and regression tests.
- **Go fuzz** for any logic that is mirrored in Go (e.g. SQL validation, parsing) in NeuronAgent.

If you introduce a standalone C helper that does not depend on PG (e.g. a small parser or escape function), add a `fuzz/` directory with a libFuzzer harness and document the build and run steps here.
