# Prometheus Observability Ecosystem for NeuronDB

This directory contains the complete Prometheus observability configuration for the NeuronDB ecosystem, including all modules (NeuronDB, NeuronAgent, NeuronDesktop, NeuronMCP) and their variants (CPU, CUDA, ROCM, Metal).

## Table of Contents

1. [Overview](#overview)
2. [File Structure](#file-structure)
3. [Configuration Files](#configuration-files)
4. [Metrics Reference](#metrics-reference)
5. [Alert Rules](#alert-rules)
6. [Recording Rules](#recording-rules)
7. [Alertmanager Configuration](#alertmanager-configuration)
8. [Exporter Configurations](#exporter-configurations)
9. [Service Discovery](#service-discovery)
10. [Setup and Deployment](#setup-and-deployment)
11. [Usage Examples](#usage-examples)
12. [Troubleshooting](#troubleshooting)

## Overview

The Prometheus observability ecosystem provides comprehensive monitoring for:

- **NeuronDB**: Database instances (CPU, CUDA, ROCM, Metal variants)
- **NeuronAgent**: Agent server instances (all variants)
- **NeuronDesktop**: API and frontend services
- **NeuronMCP**: MCP server instances (all variants)
- **Infrastructure**: System metrics, container metrics, and PostgreSQL metrics

### Key Features

- **Complete Coverage**: All modules and variants monitored
- **Detailed Metrics**: Module-specific metrics with proper labeling
- **Comprehensive Alerts**: Alerts for all critical failure modes
- **Performance Optimization**: Recording rules for common queries
- **Production Ready**: Alertmanager integration with notification routing
- **Extensible**: Easy to add new services and metrics

## File Structure

```
prometheus/
├── prometheus.yml              # Main Prometheus configuration
├── alerts.yml                  # Alert rules (organized by module)
├── recording_rules.yml         # Pre-computed metrics
├── alertmanager.yml            # Alertmanager configuration
├── postgres_exporter.yml       # PostgreSQL exporter queries
├── node_exporter.yml           # Node exporter configuration
├── service_discovery.yml       # Service discovery reference
└── README.md                   # This file
```

## Configuration Files

### prometheus.yml

Main Prometheus configuration file containing:

- **Global Settings**: Scrape intervals, evaluation intervals, external labels
- **Scrape Configs**: All service endpoints with proper labeling
  - `neuronagent`: Scrapes from `neuronagent:8080/metrics`
  - `neurondesktop-api`: Scrapes from `neurondesk-api:8081/metrics`
  - `neurondb-postgres`: Scrapes from `postgres-exporter:9187/metrics`
  - `node-exporter`: Scrapes from `node-exporter:9100/metrics`
  - `cadvisor`: Scrapes from `cadvisor:8080/metrics`
- **Rule Files**: References to alerts.yml and recording_rules.yml
- **Alertmanager**: Configuration for alert routing to `alertmanager:9093`

**Key Scrape Intervals**:
- Default: 15 seconds
- Frontend: 30 seconds (optional metrics)
- Infrastructure: 15 seconds

### alerts.yml

Comprehensive alert rules organized by module:

- **Service Availability**: `ServiceDown` - Detects when services are unreachable
- **NeuronAgent Alerts**: 
  - `NeuronAgentHighErrorRate` - Error rate > 5% (uses 5xx status label)
  - `NeuronAgentHighLatency` - P95 latency > 1s
  - `NeuronAgentHighMemoryUsage` - Memory usage > 80%
- **NeuronDesktop Alerts**:
  - `NeuronDesktopHighErrorRate` - Error rate > 5% (uses 5xx status label)
  - `NeuronDesktopHighLatency` - P95 latency > 1s
  - `NeuronDesktopHighMemoryUsage` - Memory usage > 80%
- **NeuronDB Alerts**: 
  - `NeuronDBConnectionFailure` - Database connection failed (pg_up=0)
- **Infrastructure Alerts**: 
  - `HighCPUUsage` - CPU usage > 80%
  - `HighMemoryUsage` - Memory usage > 85%
  - `HighDiskUsage` - Disk usage > 85%

All alerts use proper metric names matching the actual emitted metrics:
- `neurondb_agent_http_requests_total` (not `neurondb_agent_requests_total`)
- `neurondesktop_api_http_requests_total` (not `neurondesktop_api_requests_total`)
- Error rates calculated from `status="5xx"` label (not separate error counters)

### recording_rules.yml

Pre-computed metrics for performance optimization:

- **Request Rate Aggregations**: 
  - `neuronagent:requests:rate5m` - Agent request rate over 5 minutes
  - `neurondesktop:api:requests:rate5m` - API request rate over 5 minutes
- **Error Rate Calculations**:
  - `neuronagent:errors:ratio5m` - Agent error ratio (5xx / total)
  - `neurondesktop:api:errors:ratio5m` - API error ratio (5xx / total)
- **Latency Percentiles**:
  - `neuronagent:request_duration:p50/p95/p99` - Agent latency percentiles
  - `neurondesktop:api:request_duration:p50/p95/p99` - API latency percentiles
- **Resource Utilization**:
  - `infrastructure:cpu:usage:percent` - CPU usage percentage
  - `infrastructure:memory:usage:percent` - Memory usage percentage
  - `infrastructure:disk:usage:percent` - Disk usage percentage

All recording rules use the correct metric names matching actual emitted metrics.

### alertmanager.yml

Alert routing and notification configuration:

- **Route Configuration**: Module-based routing with severity levels
- **Inhibition Rules**: Prevent alert flooding
- **Receivers**: Email, webhook, Slack, PagerDuty templates (commented examples)
- **Grouping**: Alerts grouped by alertname, service, module

### postgres_exporter.yml

Custom PostgreSQL queries for NeuronDB-specific metrics:

- Extension statistics
- Vector statistics
- Index health metrics
- Query performance
- Connection pool statistics
- Table bloat detection
- Cache statistics

### node_exporter.yml

Node exporter textfile collector configuration and examples for custom metrics.

### service_discovery.yml

Reference documentation for service discovery configurations (Docker, static, file-based, Kubernetes).

## Metrics Reference

### NeuronDB Metrics

**Endpoint**: `http://neurondb-{variant}:9187/metrics`

| Metric | Type | Description | Labels |
|--------|------|-------------|--------|
| `neurondb_queries_total` | Counter | Total number of queries | `query_type`, `index_type` |
| `neurondb_query_duration_seconds` | Histogram | Query duration | `query_type` |
| `neurondb_index_size_bytes` | Gauge | Index size in bytes | `index_name`, `index_type` |
| `neurondb_vector_count` | Gauge | Number of vectors | `table_name` |
| `neurondb_cache_hits_total` | Counter | Cache hits | `cache_type` |
| `neurondb_cache_misses_total` | Counter | Cache misses | `cache_type` |
| `neurondb_worker_status` | Gauge | Worker status | `worker_id`, `status` |
| `neurondb_errors_total` | Counter | Total errors | `error_type` |

### NeuronAgent Metrics

**Endpoint**: `http://neuronagent-{variant}:8080/metrics`

| Metric | Type | Description | Labels |
|--------|------|-------------|--------|
| `neurondb_agent_http_requests_total` | Counter | Total HTTP requests | `method`, `endpoint`, `status` |
| `neurondb_agent_http_request_duration_seconds` | Histogram | HTTP request duration | `method`, `endpoint` |
| `neurondb_agent_executions_total` | Counter | Agent executions | `agent_id`, `status` |
| `neurondb_agent_execution_duration_seconds` | Histogram | Execution duration | `agent_id` |
| `neurondb_agent_llm_calls_total` | Counter | LLM API calls | `model`, `status` |
| `neurondb_agent_llm_tokens_total` | Counter | LLM tokens | `model`, `type` |
| `neurondb_agent_memory_chunks_stored_total` | Counter | Memory chunks stored | `agent_id` |
| `neurondb_agent_tool_executions_total` | Counter | Tool executions | `tool_name`, `status` |
| `neurondb_agent_database_connections_active` | Gauge | Active DB connections | - |
| `neurondb_agent_database_connection_errors_total` | Counter | DB connection errors | - |

### NeuronDesktop Metrics

**Endpoint**: `http://neurondesk-api:8081/metrics`

| Metric | Type | Description | Labels |
|--------|------|-------------|--------|
| `neurondesktop_api_http_requests_total` | Counter | Total API requests | `method`, `endpoint`, `status` |
| `neurondesktop_api_http_request_duration_seconds` | Histogram | Request duration | `method`, `endpoint` |
| `neurondesktop_api_active_connections` | Gauge | Active connections | `type` |
| `neurondesktop_active_neurondb_connections` | Gauge | Active NeuronDB connections | - |
| `neurondesktop_active_agent_connections` | Gauge | Active agent connections | - |
| `neurondesktop_database_connection_errors_total` | Counter | DB connection errors | - |

### NeuronMCP Metrics

**Endpoint**: `http://neurondb-mcp-{variant}:9091/metrics` (if HTTP endpoint available)

| Metric | Type | Description | Labels |
|--------|------|-------------|--------|
| `neurondb_mcp_requests_total` | Counter | Total requests | - |
| `neurondb_mcp_errors_total` | Counter | Total errors | - |
| `neurondb_mcp_request_duration_seconds` | Counter | Total request duration | - |
| `neurondb_mcp_request_duration_seconds_avg` | Gauge | Average request duration | - |
| `neurondb_mcp_method_requests_total` | Counter | Requests per method | `method` |
| `neurondb_mcp_tool_requests_total` | Counter | Requests per tool | `tool` |
| `neurondb_mcp_pool_connections_total` | Gauge | Total pool connections | - |
| `neurondb_mcp_pool_connections_active` | Gauge | Active pool connections | - |
| `neurondb_mcp_pool_connections_idle` | Gauge | Idle pool connections | - |
| `neurondb_mcp_pool_connections_max` | Gauge | Maximum pool connections | - |
| `neurondb_mcp_pool_utilization` | Gauge | Pool utilization ratio | - |

**Note**: NeuronMCP uses stdio transport by default. HTTP metrics endpoint may require a sidecar or wrapper.

### Infrastructure Metrics

**Node Exporter**: `http://node-exporter:9100/metrics`
- Standard node_exporter metrics (CPU, memory, disk, network)

**cAdvisor**: `http://cadvisor:8080/metrics`
- Container metrics (CPU, memory, network, filesystem)

**PostgreSQL Exporter**: `http://postgres-exporter-{variant}:9187/metrics`
- Standard PostgreSQL metrics + custom queries from `postgres_exporter.yml`

## Alert Rules

### NeuronDB Alerts

| Alert Name | Severity | Condition | Description |
|------------|----------|-----------|-------------|
| `NeuronDBServiceDown` | Critical | Service down > 1m | NeuronDB instance unavailable |
| `NeuronDBConnectionFailure` | Critical | >5 failures in 5m | Database connection failures |
| `NeuronDBHighQueryLatency` | Warning | P95 > 1s for 5m | High query latency |
| `NeuronDBIndexHealthDegraded` | Warning | Health < 80% for 5m | Vector index health degraded |
| `NeuronDBCacheHitRateLow` | Warning | Hit rate < 70% for 5m | Low cache hit rate |
| `NeuronDBConnectionPoolExhausted` | Critical | Utilization > 90% for 5m | Connection pool nearly full |
| `NeuronDBHighVectorQueryRate` | Info | >100 queries/sec for 5m | High vector query rate |
| `NeuronDBDiskSpaceLow` | Warning | Available < 15% for 5m | Low disk space |

### NeuronAgent Alerts

| Alert Name | Severity | Condition | Description |
|------------|----------|-----------|-------------|
| `NeuronAgentServiceDown` | Critical | Service down > 1m | NeuronAgent unavailable |
| `NeuronAgentHighErrorRate` | Critical | Error rate > 5% for 5m | High error rate |
| `NeuronAgentHighLatency` | Warning | P95 > 1s for 5m | High request latency |
| `NeuronAgentExecutionFailure` | Critical | >10 failures in 5m | High execution failure rate |
| `NeuronAgentDatabaseConnectionIssue` | Warning | >5 errors in 5m | Database connection issues |
| `NeuronAgentHighMemoryUsage` | Warning | Usage > 80% for 5m | High memory usage |
| `NeuronAgentHighCPUUsage` | Warning | Usage > 80% for 5m | High CPU usage |
| `NeuronAgentRequestRateSpike` | Info | 2x increase for 2m | Request rate spike |

### NeuronDesktop Alerts

| Alert Name | Severity | Condition | Description |
|------------|----------|-----------|-------------|
| `NeuronDesktopAPIDown` | Critical | API down > 1m | API unavailable |
| `NeuronDesktopHighErrorRate` | Critical | Error rate > 5% for 5m | High error rate |
| `NeuronDesktopFrontendDown` | Warning | Frontend down > 2m | Frontend unavailable |
| `NeuronDesktopDatabaseConnectionIssue` | Warning | >5 errors in 5m | Database connection issues |
| `NeuronDesktopHighResponseTime` | Warning | P95 > 2s for 5m | High response time |
| `NeuronDesktopHighActiveConnections` | Warning | >100 connections for 5m | High connection count |

### NeuronMCP Alerts

| Alert Name | Severity | Condition | Description |
|------------|----------|-----------|-------------|
| `NeuronMCPServiceDown` | Critical | Service down > 1m | NeuronMCP unavailable |
| `NeuronMCPHighErrorRate` | Critical | Error rate > 5% for 5m | High error rate |
| `NeuronMCPToolExecutionFailure` | Warning | >10 failures in 5m | High tool execution failures |
| `NeuronMCPConnectionPoolIssue` | Warning | Utilization > 90% for 5m | Connection pool high |
| `NeuronMCPHighLatency` | Warning | Avg > 1s for 5m | High request latency |
| `NeuronMCPMethodHighErrorRate` | Warning | Error rate > 10% for 5m | High method error rate |

### Infrastructure Alerts

| Alert Name | Severity | Condition | Description |
|------------|----------|-----------|-------------|
| `HighCPUUsage` | Warning | CPU > 80% for 5m | High CPU usage |
| `HighMemoryUsage` | Warning | Memory > 85% for 5m | High memory usage |
| `HighDiskUsage` | Warning | Disk > 85% for 5m | High disk usage |
| `ContainerHighCPUUsage` | Warning | Container CPU > 80% for 5m | High container CPU |
| `ContainerHighMemoryUsage` | Warning | Container memory > 90% for 5m | High container memory |
| `HighNetworkErrorRate` | Warning | >10 errors/sec for 5m | High network errors |
| `HighDiskIO` | Warning | I/O time > 90% for 5m | High disk I/O |
| `PrometheusTargetDown` | Critical | Target down > 2m | Prometheus target unavailable |
| `AlertmanagerDown` | Critical | Alertmanager down > 1m | Alertmanager unavailable |

## Recording Rules

Recording rules pre-compute common queries for better performance. Key recording rules include:

### Request Rates
- `neurondb:queries:rate5m` - Query rate over 5 minutes
- `neuronagent:requests:rate5m` - Agent request rate
- `neurondesktop:api:requests:rate5m` - API request rate
- `neuronmcp:requests:rate5m` - MCP request rate

### Error Rates
- `neurondb:errors:ratio5m` - Error ratio
- `neuronagent:errors:ratio5m` - Agent error ratio
- `neurondesktop:api:errors:ratio5m` - API error ratio
- `neuronmcp:errors:ratio5m` - MCP error ratio

### Latency Percentiles
- `neurondb:query_duration:p95` - P95 query duration
- `neuronagent:request_duration:p95` - P95 request duration
- `neurondesktop:api:request_duration:p95` - P95 API duration

### Resource Utilization
- `infrastructure:cpu:usage:percent` - CPU usage percentage
- `infrastructure:memory:usage:percent` - Memory usage percentage
- `infrastructure:disk:usage:percent` - Disk usage percentage

## Alertmanager Configuration

### Alert Routing

Alerts are routed based on:
- **Severity**: Critical alerts go to `critical-alerts` receiver
- **Module**: Module-specific receivers (neurondb-alerts, neuronagent-alerts, etc.)
- **Grouping**: Alerts grouped by alertname, service, module

### Notification Channels

The configuration includes commented examples for:
- **Email**: SMTP configuration
- **Slack**: Webhook integration
- **PagerDuty**: Service integration
- **Webhook**: Custom webhook endpoints

**To enable notifications**:
1. Uncomment and configure the desired notification channel in `alertmanager.yml`
2. Update receiver configurations with your credentials/endpoints
3. Restart Alertmanager

### Inhibition Rules

Inhibition rules prevent alert flooding:
- Warning alerts are suppressed if a critical alert with the same labels is firing
- Module-specific alerts are suppressed if the service is down

## Exporter Configurations

### PostgreSQL Exporter

The `postgres_exporter.yml` file contains custom queries for NeuronDB-specific metrics:

- Extension statistics
- Vector statistics
- Index health
- Query performance
- Connection pool metrics
- Table bloat detection
- Cache statistics

**To use custom queries**:
1. Mount `postgres_exporter.yml` to the postgres_exporter container
2. Configure postgres_exporter with `--extend.query-path=/path/to/postgres_exporter.yml`
3. Ensure queries are compatible with your PostgreSQL version

### Node Exporter

The `node_exporter.yml` file documents the textfile collector configuration:

- Custom metrics via textfile collector
- Service health checks
- External dependency monitoring

**To add custom metrics**:
1. Write Prometheus-formatted metrics to `/var/lib/node_exporter/textfile_collector/*.prom`
2. Use systemd timers or cron jobs to update metrics files
3. Metrics are automatically collected by node_exporter

## Service Discovery

The `service_discovery.yml` file provides reference documentation for:

- **Docker Service Discovery**: Automatic container discovery
- **Static Service Discovery**: Current implementation (most reliable)
- **File-based Discovery**: Dynamic service registration
- **Kubernetes Discovery**: For Kubernetes deployments

Current implementation uses static service discovery for reliability.

## Setup and Deployment

### Prerequisites

- Docker and Docker Compose
- Access to `neurondb-network` Docker network
- All NeuronDB ecosystem services running

### Quick Start

1. **Start the full observability stack**:
   ```bash
   docker compose -f docker-compose.observability.yml up -d
   ```

2. **Verify Prometheus**:
   - Access Prometheus UI: http://localhost:9090
   - Check targets: http://localhost:9090/targets
   - Verify all services are "UP" (neuronagent, neurondesktop-api, postgres-exporter, node-exporter, cadvisor)
   - Check rules: http://localhost:9090/rules (should show alerts and recording rules)

3. **Verify Alertmanager**:
   - Access Alertmanager UI: http://localhost:9093
   - Check alert status and routing
   - Configure webhook URL via `ALERTMANAGER_WEBHOOK_URL` environment variable

4. **Verify Grafana**:
   - Access Grafana UI: http://localhost:3001
   - Login: admin/admin (change in production!)
   - Verify dashboards are loaded: NeuronAgent, NeuronDesktop API, NeuronDB/PostgreSQL, Infrastructure

### Verification Checklist

#### Docker Compose Deployment

- [ ] All services are running: `docker compose -f docker-compose.observability.yml ps`
- [ ] Prometheus targets are UP: http://localhost:9090/targets
- [ ] Prometheus rules are loaded: http://localhost:9090/rules
- [ ] Alertmanager is accessible: http://localhost:9093
- [ ] Grafana dashboards are visible: http://localhost:3001
- [ ] Metrics endpoints are accessible:
  - `curl http://neuronagent:8080/metrics` (should return Prometheus format)
  - `curl http://neurondesk-api:8081/metrics` (should return Prometheus format)
- [ ] Recording rules are working: Query `neuronagent:requests:rate5m` in Prometheus

#### Helm/Kubernetes Deployment

- [ ] Prometheus pods are running: `kubectl get pods -l app.kubernetes.io/component=prometheus`
- [ ] Alertmanager pods are running (if enabled): `kubectl get pods -l app.kubernetes.io/component=alertmanager`
- [ ] Grafana pods are running: `kubectl get pods -l app.kubernetes.io/component=grafana`
- [ ] ServiceMonitors are created (if using Prometheus Operator): `kubectl get servicemonitors`
- [ ] PrometheusRules are created (if using Prometheus Operator): `kubectl get prometheusrules`
- [ ] Metrics endpoints are accessible from within cluster:
  - `kubectl exec -it <pod> -- curl http://<service>:<port>/metrics`
- [ ] Webhook URL is configured in Alertmanager (if using webhooks)

### Docker Compose Integration

Update `docker-compose.observability.yml` to include:

```yaml
services:
  prometheus:
    volumes:
      - ./prometheus/prometheus.yml:/etc/prometheus/prometheus.yml
      - ./prometheus/alerts.yml:/etc/prometheus/alerts.yml
      - ./prometheus/recording_rules.yml:/etc/prometheus/recording_rules.yml
      - prometheus-data:/prometheus
  
  alertmanager:
    volumes:
      - ./prometheus/alertmanager.yml:/etc/alertmanager/alertmanager.yml
      - alertmanager-data:/alertmanager
  
  postgres-exporter-cpu:
    image: prometheuscommunity/postgres-exporter:latest
    environment:
      DATA_SOURCE_NAME: "postgresql://neurondb:neurondb@neurondb-cpu:5432/neurondb?sslmode=disable"
    command:
      - '--extend.query-path=/etc/postgres_exporter/queries.yml'
    volumes:
      - ./prometheus/postgres_exporter.yml:/etc/postgres_exporter/queries.yml
    ports:
      - "9187:9187"
  
  node-exporter:
    image: prom/node-exporter:latest
    volumes:
      - /proc:/host/proc:ro
      - /sys:/host/sys:ro
      - /:/rootfs:ro
    command:
      - '--path.procfs=/host/proc'
      - '--path.sysfs=/host/sys'
      - '--collector.filesystem.mount-points-exclude=^/(sys|proc|dev|host|etc)($$|/)'
    ports:
      - "9100:9100"
  
  cadvisor:
    image: gcr.io/cadvisor/cadvisor:latest
    volumes:
      - /:/rootfs:ro
      - /var/run:/var/run:ro
      - /sys:/sys:ro
      - /var/lib/docker/:/var/lib/docker:ro
    ports:
      - "8080:8080"
```

### Configuration Updates

After modifying configuration files:

1. **Reload Prometheus** (without restart):
   ```bash
   curl -X POST http://localhost:9090/-/reload
   ```

2. **Or restart services**:
   ```bash
   docker compose -f docker-compose.observability.yml restart prometheus alertmanager
   ```

## Usage Examples

### Query Examples

**Request rate for NeuronAgent**:
```promql
rate(neurondb_agent_http_requests_total[5m])
```

**Error rate percentage** (using 5xx status label):
```promql
sum(rate(neurondb_agent_http_requests_total{status="5xx"}[5m])) / sum(rate(neurondb_agent_http_requests_total[5m])) * 100
```

**P95 latency**:
```promql
histogram_quantile(0.95, rate(neurondb_agent_http_request_duration_seconds_bucket[5m]))
```

**Using recording rules**:
```promql
neuronagent:request_duration:p95
neuronagent:requests:rate5m
neuronagent:errors:ratio5m
```

### Dashboard Queries

**Service availability**:
```promql
up{job="neuronagent"}
```

**Total requests by method and status**:
```promql
sum(rate(neurondb_agent_http_requests_total[5m])) by (method, status)
```

**Error rate by endpoint**:
```promql
sum(rate(neurondb_agent_http_requests_total{status="5xx"}[5m])) by (endpoint) / 
sum(rate(neurondb_agent_http_requests_total[5m])) by (endpoint) * 100
```

## Troubleshooting

### Services Not Scraping

1. **Check service is running**:
   ```bash
   docker ps | grep neurondb
   ```

2. **Check network connectivity**:
   ```bash
   docker exec neurondb-prometheus wget -O- http://neuronagent:8080/metrics
   ```

3. **Check Prometheus targets**:
   - Visit http://localhost:9090/targets
   - Look for "DOWN" status
   - Check error messages

4. **Verify service exposes metrics**:
   ```bash
   curl http://neuronagent:8080/metrics
   ```

### Alerts Not Firing

1. **Check alert rules are loaded**:
   - Visit http://localhost:9090/rules
   - Verify rules are listed

2. **Check alert evaluation**:
   - Visit http://localhost:9090/alerts
   - Check alert state (pending/firing)

3. **Test alert expression**:
   - Use Prometheus query UI to test alert expressions
   - Verify metrics exist and have expected values

### Alertmanager Not Receiving Alerts

1. **Check Prometheus configuration**:
   - Verify `alertmanager.yml` has correct Alertmanager URL
   - Check `alerting.alertmanagers` section in `prometheus.yml`

2. **Check Alertmanager status**:
   - Visit http://localhost:9093
   - Check Alertmanager is running

3. **Check network connectivity**:
   ```bash
   docker exec neurondb-prometheus wget -O- http://alertmanager:9093/-/healthy
   ```

### Metrics Not Available

1. **Check service metrics endpoint**:
   ```bash
   curl http://service:port/metrics
   ```

2. **Verify service exposes Prometheus format**:
   - Metrics should start with `# HELP` and `# TYPE`
   - Check for proper metric naming

3. **Check scrape configuration**:
   - Verify job name and targets in `prometheus.yml`
   - Check metrics_path is correct

### High Memory Usage

1. **Reduce retention**:
   - Update `--storage.tsdb.retention.time` in Prometheus command
   - Default is 30 days

2. **Reduce scrape frequency**:
   - Increase `scrape_interval` in `prometheus.yml`
   - Note: This reduces metric resolution

3. **Limit metrics collected**:
   - Use metric relabeling to drop unnecessary metrics
   - Configure exporters to collect only needed metrics

## Additional Resources

- [Prometheus Documentation](https://prometheus.io/docs/)
- [Alertmanager Documentation](https://prometheus.io/docs/alerting/latest/alertmanager/)
- [PostgreSQL Exporter](https://github.com/prometheus-community/postgres_exporter)
- [Node Exporter](https://github.com/prometheus/node_exporter)
- [cAdvisor](https://github.com/google/cadvisor)

## Support

For issues or questions:
- Check service logs: `docker logs <container-name>`
- Review Prometheus logs: `docker logs neurondb-prometheus`
- Check Alertmanager logs: `docker logs neurondb-alertmanager`
- Consult NeuronDB documentation

