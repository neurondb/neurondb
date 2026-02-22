# Contrib – production-ready implementations

This directory contains **complete, copy-paste-ready** implementations for features that live in sibling or downstream repos (neurondb-cloud, neurondb-hub). Each subdirectory is self-contained and verified (builds and, where applicable, tests pass).

## neurondb-cloud

| Path | Description |
|------|-------------|
| **metering/** | Metering collector: lists tenants from control plane DB, gathers usage, writes to `metering_records`. Run as cron or daemon. Requires `DATABASE_URL` and optional `METERING_WINDOW_MINUTES`, `METERING_DRY_RUN`. |
| **notifications/delivery/** | Email (SMTP) and Slack delivery. Use `SendEmail(ctx, Email{...})` and `SendSlack(ctx, SlackMessage{...})`. Configure via `SMTP_*` and `SLACK_WEBHOOK_URL`. |

Copy into your control plane repo (e.g. `services/metering`, `internal/notifications/delivery`) and adjust module paths.

## neurondb-hub

| Path | Description |
|------|-------------|
| **metrics/** | Prometheus metrics for Hub backend/gateway: `Register()`, `Middleware(next)`, `Handler()` for `/metrics`, and counters `CountAgentCreation()`, `CountKnowledgeIngest()`, `CountAuthFailure()`. |

Copy into your Hub backend (e.g. `internal/metrics`) and wire in `main.go`.

## Verification

- **metering**: `cd contrib/neurondb-cloud/metering && go build ./cmd/collector`
- **notifications**: stdlib only; no build required.
- **neurondb-hub/metrics**: `cd contrib/neurondb-hub && go build ./metrics/`
