# Troubleshooting (Simple)

This page lists common setup issues and quick checks.

## Postgres can’t load the extension

- Confirm the extension was installed into the correct Postgres installation.
- Check Postgres logs for the exact dynamic loader error:
  - missing `.so` (Linux)
  - missing symbols (ABI mismatch)
- Validate:
  - Postgres version compatibility
  - correct `shared_preload_libraries` (if required by features)

## Docker container starts but health checks fail

- `docker compose ps` to see which service is unhealthy
- `docker compose logs -f <service>`
- Confirm required env vars exist (see `env.example`)

## GPU features not available

- Ensure you built with the right backend:
  - CUDA / Metal / ROCm
- Check build flags and platform constraints.
- Fall back to CPU path and confirm correctness first.

## Queries are slow

- Confirm an index exists and is used by the planner.
- Use `EXPLAIN (ANALYZE, BUFFERS)` to see plan details.
- Check:
  - index parameters
  - vector dimensionality
  - filter selectivity

## Where to get more help

- Repo docs: `documentation.md` and `NeuronDB/docs/`
- Component docs:
  - `NeuronAgent/README.md`
  - `NeuronMCP/README.md`
  - `NeuronDesktop/README.md`


