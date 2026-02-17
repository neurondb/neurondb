# Launch Readiness Checklist

Use this before a production launch or major release.

## Security

- [ ] All critical/high security findings from last audit or scan are fixed.
- [ ] No hardcoded secrets; secrets from Vault or env.
- [ ] TLS enabled for client-facing and inter-service traffic.
- [ ] Security headers and CORS configured correctly.
- [ ] Authentication enforced; JWT secret and passwords strong.

## Reliability

- [ ] Health checks (HTTP/functional) on all services; orchestrator uses them.
- [ ] Graceful shutdown and connection draining implemented and tested.
- [ ] Resource limits (CPU/memory) set on all containers/pods.
- [ ] Database backups automated and restore tested.

## Observability

- [ ] Logging structured with request/correlation IDs.
- [ ] Metrics exported (Prometheus or equivalent); dashboards in place.
- [ ] Alerting rules configured (error rate, latency, resource usage).
- [ ] On-call rotation and runbooks documented.

## Performance

- [ ] Load test at 2x expected traffic passed.
- [ ] P99 latency and error rate within SLA.
- [ ] Database and connection pools tuned.

## Compliance and legal

- [ ] Security disclosure policy published.
- [ ] SBOM generated and retained; dependency scan in CI (fail on critical).
- [ ] SOC 2 or other compliance prep if required.

## Documentation

- [ ] API docs and OpenAPI specs up to date.
- [ ] Runbooks for incident response and DR.
- [ ] Quickstart and deployment docs accurate.

## Final sign-off

- [ ] All CI/CD pipelines green.
- [ ] All tests passing; coverage target met (e.g. 80% for launch).
- [ ] Rollback procedure tested.
- [ ] Launch owner and go-live date confirmed.
