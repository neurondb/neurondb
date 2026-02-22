# Metering Collector (neurondb-cloud)

Production-ready metering collector for the control plane. Copy this directory into your `neurondb-cloud` repo (e.g. under `services/metering` or `internal/metering`).

## Database schema

Run in the control plane database:

```sql
CREATE TABLE IF NOT EXISTS metering_records (
  id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  org_id        UUID NOT NULL REFERENCES organizations(id),
  tenant_id     UUID NOT NULL REFERENCES tenants(id),
  metric_type   TEXT NOT NULL,
  quantity      NUMERIC NOT NULL,
  window_start  TIMESTAMPTZ NOT NULL,
  window_end    TIMESTAMPTZ NOT NULL,
  metadata      JSONB,
  created_at    TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_metering_tenant_time ON metering_records(tenant_id, window_start, window_end);
CREATE INDEX IF NOT EXISTS idx_metering_org_time ON metering_records(org_id, window_start, window_end);
```

## Configuration

| Env | Description |
|-----|-------------|
| DATABASE_URL | Control plane PostgreSQL connection string |
| METERING_WINDOW_MINUTES | Aggregation window (default 60) |
| METERING_DRY_RUN | If set, log only, do not insert |

## Build and run

```bash
go build -o metering-collector ./cmd/collector
./metering-collector
```

Run as a cron job (e.g. every hour) or as a long-running service with a ticker.
