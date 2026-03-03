# Metering Collection (neuron-cloud)

This document describes how to implement metering collection for **neuron-cloud** so that usage can be tracked per tenant and used for billing and limits.

## Goals

- Collect usage metrics from provisioned NeuronDB instances (or from the control plane’s view of tenant activity).
- Store metrics in the control plane (e.g. `usage_events` or `metering_records` table).
- Expose aggregated usage via API for billing and dashboards.

## Suggested approach

### 1. Data model (control plane DB)

Example schema addition:

```sql
-- metering_records: one row per usage event or per aggregation window
CREATE TABLE IF NOT EXISTS metering_records (
  id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  org_id        UUID NOT NULL REFERENCES organizations(id),
  tenant_id     UUID NOT NULL REFERENCES tenants(id),
  metric_type   TEXT NOT NULL,   -- e.g. 'vector_search', 'embedding', 'storage_gb', 'api_calls'
  quantity      NUMERIC NOT NULL,
  window_start  TIMESTAMPTZ NOT NULL,
  window_end    TIMESTAMPTZ NOT NULL,
  metadata      JSONB,
  created_at    TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_metering_tenant_time ON metering_records(tenant_id, window_start, window_end);
CREATE INDEX idx_metering_org_time ON metering_records(org_id, window_start, window_end);
```

### 2. Metric collector service

- **Option A (pull):** A control-plane job periodically connects to each tenant’s NeuronDB (or NeuronAgent) and reads metrics (e.g. from Prometheus or a dedicated `/metrics` or `/usage` endpoint), then writes to `metering_records`.
- **Option B (push):** Each NeuronAgent/NeuronDB instance pushes usage to the control plane (e.g. `POST /api/v1/usage` with tenant key); control plane validates and writes to `metering_records`.

Implement a small **metering collector** in `services/metering` (or similar) that:

1. Lists active tenants from the control plane DB.
2. For each tenant, fetches usage (pull from tenant API or from a shared metrics store).
3. Aggregates into time windows (e.g. hourly) and inserts/updates `metering_records`.

### 3. API

- `GET /api/v1/orgs/{org_id}/usage?from=&to=&metric_type=` – return aggregated usage for billing/dashboards (protected by RBAC).

### 4. Configuration

- `METERING_WINDOW_MINUTES` (e.g. 60).
- `METERING_TENANT_METRICS_URL` or per-tenant config for where to pull metrics.

When the **neuron-cloud** repo is present, implement the above in the control plane and run the metering collector as a cron or long-running service.
