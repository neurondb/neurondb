# Load Testing and Optimization

## Goals

- Validate performance at **2x expected production traffic**.
- Establish baselines for latency (P50, P95, P99) and throughput (req/s).
- Identify bottlenecks (DB, LLM, CPU, memory) and tune.

## Tools

- **API load**: k6, hey, or Gatling against NeuronAgent and NeuronDesktop APIs. Script scenarios: create agent, send message, list sessions.
- **Database**: pgbench or custom SQL scripts for NeuronDB (vector queries, ML train/predict). Run from multiple clients to simulate concurrency.
- **End-to-end**: Run a mix of API and DB workloads to mimic production.

## Procedure

1. Define scenarios and success criteria (e.g. P99 < 2s, error rate < 0.1%).
2. Run baseline at 1x load; record metrics.
3. Ramp to 2x; fix any regressions (query tuning, indexes, connection pools, scaling).
4. Store results (e.g. in `benchmark/results/`) and track over time in CI or a dashboard.

## Optimization

- **PostgreSQL**: Tune `shared_buffers`, `work_mem`, `max_connections`; add indexes for hot queries.
- **Connection pooling**: Size pools (NeuronAgent → DB, NeuronDesktop → DB) to match concurrency and DB limits.
- **Docker images**: Multi-stage builds, slim base images, and avoid unnecessary layers to reduce image size.
- **CDN**: Serve NeuronDesktop frontend static assets from a CDN in production.
