# Observability Setup Guide

## Overview

This guide explains how to set up end-to-end observability for the NeuronDB ecosystem using Prometheus, OpenTelemetry, and Grafana.

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  Services   │────▶│ Prometheus   │────▶│  Grafana    │
│ (Metrics)   │     │  (Scrape)    │     │ (Dashboards)│
└─────────────┘     └──────────────┘     └─────────────┘
                            │
                            ▼
                    ┌──────────────┐
                    │ OpenTelemetry │
                    │   (Tracing)   │
                    └──────────────┘
```

## Components

### 1. Prometheus Metrics

All services expose Prometheus metrics at `/metrics`:

- **NeuronAgent**: `http://localhost:8080/metrics`
- **NeuronDesktop API**: `http://localhost:8081/metrics`
- **NeuronDB**: Custom metrics via PostgreSQL extension

### 2. OpenTelemetry Tracing

Distributed tracing across all components:
- Request correlation IDs
- Span propagation
- Trace export to Jaeger/OTLP

### 3. Grafana Dashboards

Pre-built dashboards for:
- System health
- Performance metrics
- Error rates
- Database metrics

## Setup

### Quick Start - Docker Compose

```bash
# Start the complete observability stack
docker compose -f docker-compose.observability.yml up -d

# This starts:
# - Prometheus (localhost:9090)
# - Grafana (localhost:3001, admin/admin)
# - Alertmanager (localhost:9093)
# - Postgres Exporter (localhost:9187)
# - Node Exporter (localhost:9100)
# - cAdvisor (localhost:8080)
# - Jaeger (localhost:16686)
```

### Quick Start - Helm/Kubernetes

```bash
# Install with monitoring enabled
helm install neurondb ./helm/neurondb \
  --set monitoring.enabled=true \
  --set monitoring.prometheus.enabled=true \
  --set monitoring.grafana.enabled=true \
  --set monitoring.prometheus.alertmanager.enabled=true

# Access dashboards
kubectl port-forward svc/neurondb-grafana 3001:3000
kubectl port-forward svc/neurondb-prometheus 9090:9090
```

## Metrics

### NeuronAgent Metrics

- `neurondb_agent_http_requests_total` - Total HTTP requests (with `method`, `endpoint`, `status` labels)
- `neurondb_agent_http_request_duration_seconds` - Request duration histogram
- `neurondb_agent_executions_total` - Agent execution count
- `neurondb_agent_llm_calls_total` - LLM API calls
- `neurondb_agent_tool_executions_total` - Tool execution count

### NeuronDesktop Metrics

- `neurondesktop_api_http_requests_total` - Total HTTP requests (with `method`, `endpoint`, `status` labels)
- `neurondesktop_api_http_request_duration_seconds` - Request duration histogram
- `neurondesktop_api_active_connections` - Active connections by type

### NeuronDB Metrics

- `neurondb_queries_total` - Total queries
- `neurondb_query_duration_seconds` - Query duration
- `neurondb_indexes_total` - Total indexes
- `neurondb_vectors_stored` - Vectors stored
- `neurondb_embedding_requests_total` - Embedding requests

## Tracing

### Trace Context Propagation

All services propagate trace context via headers:
- `traceparent` (W3C Trace Context)
- `tracestate` (W3C Trace State)

### Example Trace

```
HTTP Request → NeuronDesktop API
  ├─ SQL Query → NeuronDB
  ├─ Agent Call → NeuronAgent
  │   ├─ Tool Execution
  │   └─ LLM Call
  └─ Response
```

## Dashboards

### System Health Dashboard

- Service uptime
- Health check status
- Error rates
- Request rates

### Performance Dashboard

- Request latency (p50, p95, p99)
- Throughput
- Database query performance
- Vector search performance

### Database Dashboard

- Connection pool stats
- Query performance
- Index health
- Vector operations

## Alerts

Alerts are configured in `prometheus/alerts.yml` with proper metric names:

### Critical Alerts

- **ServiceDown**: Service unavailable (checks `up` metric)
- **NeuronAgentHighErrorRate**: Error rate > 5% (uses `status="5xx"` label)
- **NeuronDesktopHighErrorRate**: Error rate > 5% (uses `status="5xx"` label)
- **NeuronDBConnectionFailure**: Database unreachable
- **HighCPUUsage**: CPU > 80%
- **HighMemoryUsage**: Memory > 85%
- **HighDiskUsage**: Disk > 85%

### Warning Alerts

- **NeuronAgentHighLatency**: P95 latency > 1s
- **NeuronDesktopHighLatency**: P95 latency > 1s

## Verification

See `prometheus/VERIFICATION.md` for a complete checklist to verify your observability stack.

## Best Practices

1. **Set appropriate scrape intervals**: 15-30 seconds
2. **Retain metrics**: 30 days for metrics, 7 days for traces
3. **Use labels wisely**: Don't create high cardinality
4. **Monitor cardinality**: Watch for label explosion
5. **Set up alerts**: But avoid alert fatigue

## Troubleshooting

### Metrics Not Appearing

1. Check service is exposing `/metrics`
2. Verify Prometheus can reach service
3. Check Prometheus targets page
4. Review service logs

### Traces Missing

1. Verify OTEL exporter endpoint
2. Check trace context propagation
3. Review OpenTelemetry logs
4. Verify sampling configuration

### High Cardinality

1. Review label usage
2. Use aggregation where possible
3. Consider reducing label dimensions
4. Monitor Prometheus memory usage




