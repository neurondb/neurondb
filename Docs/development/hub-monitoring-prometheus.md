# Hub Monitoring with Prometheus

This document describes how to add **monitoring and metrics** (Prometheus) to **neurondb-hub** so that backend and gateway can be observed in production.

## Goals

- Expose Prometheus metrics from Hub backend and gateway (e.g. `/metrics`).
- Optionally add structured logging (JSON) for easier aggregation.
- Use the same patterns as NeuronAgent/NeuronMCP where applicable.

## Suggested approach

### 1. Metrics endpoint (backend and gateway)

- Add `github.com/prometheus/client_golang` to the Hub backend and gateway (both Go).
- Register default Go metrics (`prometheus.DefaultCollectors`) and a custom registry.
- Add HTTP handler for `GET /metrics` (only on a dedicated port or path; restrict access in production).
- Counters: `http_requests_total`, `agent_creations_total`, `knowledge_ingest_total`, `auth_failures_total`.
- Histograms: `http_request_duration_seconds` (by method, path, status).

### 2. Structured logging

- Use a logger that supports JSON output (e.g. `zerolog`, `zap`) and set level via `LOG_LEVEL`.
- Include `request_id`, `org_id`, `user_id` where available for tracing.

### 3. Docker and deployment

- In `docker-compose` and any Helm chart, expose the metrics port (e.g. 9090) and add a Prometheus scrape config for `neurondb-hub-backend` and `neurondb-hub-gateway`.
- Document required Prometheus scrape config in Hub’s deployment docs.

### 4. Alerts (optional)

- Define Alertmanager rules for Hub (e.g. high error rate, high latency) when Hub is deployed with the same observability stack as the rest of the ecosystem.

When the **neurondb-hub** repo is present, implement the `/metrics` endpoint and structured logging in the backend and gateway, then add the corresponding scrape configuration to your Prometheus setup.
