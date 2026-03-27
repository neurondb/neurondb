# Observability Setup Guide

Observability for the NeuronDB extension and PostgreSQL using Prometheus, OpenTelemetry, and Grafana.

## Architecture

```
┌─────────────────────┐     ┌──────────────┐     ┌─────────────┐
│ PostgreSQL/NeuronDB │────▶│ Prometheus   │────▶│  Grafana    │
│ (postgres_exporter)  │     │  (Scrape)    │     │ (Dashboards)│
└─────────────────────┘     └──────────────┘     └─────────────┘
```

## Components

### Prometheus metrics

- **PostgreSQL:** Use [postgres_exporter](https://github.com/prometheus-community/postgres_exporter) for standard PG metrics.
- **NeuronDB extension:** Use `pg_stat_statements`, `pg_stat_user_tables`, `pg_stat_user_indexes`, and extension functions such as `neurondb.version()`, `neurondb.gpu_enabled()` (see [Operational Playbook](playbook.md#observability)).

### OpenTelemetry (optional)

If you use OTLP for tracing, configure your application or middleware to export traces; the extension itself runs inside PostgreSQL and does not expose a separate HTTP `/metrics` or trace endpoint.

## Setup

### Docker Compose

From the repository root:

```bash
docker compose -f docker/docker-compose.observability.yml up -d
```

This can start Prometheus, Grafana, Postgres Exporter, and related services. See `docker/docker-compose.observability.yml` for exact services and ports.

### Kubernetes / Helm

If you use the Helm charts in this repo, enable monitoring options as documented in the chart (e.g. `monitoring.enabled`, Prometheus/Grafana). For other services (Agent, Desktop, MCP), see their own repositories.

## NeuronDB and PostgreSQL metrics

Use standard PostgreSQL statistics and extension queries:

- **Extension version:** `SELECT neurondb.version();`
- **GPU:** `SELECT neurondb.gpu_enabled();`, `SELECT neurondb.gpu_device_count();`
- **Index usage:** Query `pg_stat_user_indexes` for indexes whose names contain `hnsw` or `ivf`
- **Query performance:** Enable `pg_stat_statements` and filter by queries touching neurondb objects

## Dashboards

- **Database:** Connection pool, query performance, index health, vector-related activity.
- **System:** CPU, memory, disk (standard node/postgres exporter metrics).

## Alerts

Configure alerts on:

- PostgreSQL/NeuronDB unreachable
- High error rate or failed connections
- High CPU/memory/disk

Exact metric names and labels depend on your postgres_exporter and Prometheus configuration.

## Best practices

1. Set scrape intervals (e.g. 15–30 seconds) and retention (e.g. 30 days) appropriately.
2. Avoid high-cardinality labels.
3. Set up a small set of critical alerts to avoid alert fatigue.

---

[Operations](README.md) · [Documentation](../readme.md)
