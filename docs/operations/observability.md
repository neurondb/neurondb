# Observability

## OpenTelemetry distributed tracing

- **NeuronAgent**: Use `go.opentelemetry.io/otel` and export spans to an OTLP collector. Create a TracerProvider at startup and inject trace context into HTTP handlers and DB/LLM calls.
- **NeuronDesktop API**: Add OpenTelemetry SDK for Go; create spans for each request and propagate trace ID in response headers (e.g. `X-Trace-ID`).
- **NeuronMCP**: Instrument MCP request handling with spans; propagate context from clients.
- **NeuronDB**: Use PostgreSQL `pg_tracing` or application-level spans for training/prediction calls from application code.

Configure via env: `OTEL_EXPORTER_OTLP_ENDPOINT`, `OTEL_SERVICE_NAME=neurondesktop-api`, etc.

## Structured logging and correlation IDs

- **NeuronAgent**: Already uses structured logging (zerolog). Ensure every log entry includes `request_id` (from middleware) so logs can be correlated. Add `trace_id` and `span_id` when OpenTelemetry is enabled.
- **NeuronDesktop**: Add a request ID middleware and include `request_id` in all log fields. Use the same ID in responses (header `X-Request-ID`).
- **NeuronMCP**: Add request ID to log context for each MCP request.

## Grafana dashboards

- Create dashboards per service:
  - **NeuronDB**: Connection count, query latency (p99), ML training duration, vector index size.
  - **NeuronAgent**: Request rate, latency by endpoint, LLM call count and latency, error rate.
  - **NeuronDesktop**: API latency, auth success/failure, session count.
  - **NeuronMCP**: Tool invocations, latency, errors.
- Store dashboard JSON in `monitoring/grafana/dashboards/` and provision via Grafana config or ConfigMaps.

## Alerting rules

- **Error rate**: Alert when 5xx or error rate > 5% over 5 minutes.
- **Latency P99**: Alert when P99 latency exceeds SLA (e.g. > 2s for API).
- **Resource usage**: Alert when CPU > 80% or memory > 85% for critical pods.
- **Security**: Alert on repeated auth failures, unusual request patterns.

Define in Prometheus Alertmanager or Grafana (e.g. `monitoring/prometheus/alerts/`).

## Audit logging

- Log all data access and admin operations: who (user/session), what (resource, action), when, result.
- Store audit events in a dedicated table or stream to a log aggregator. Do not log request bodies that may contain PII; log only resource IDs and action types.
- Implement in each service: NeuronDesktop (API and SQL console), NeuronAgent (agent runs, tool calls), NeuronDB (via PostgreSQL audit extension or application-level logging).
