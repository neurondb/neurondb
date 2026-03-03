# Incident Response Runbook

## Severity levels

- **P1 Critical**: Full outage or data loss; respond immediately.
- **P2 High**: Major feature broken or degraded for many users; respond within 1 hour.
- **P3 Medium**: Minor feature broken or workaround exists; respond within 4 hours.
- **P4 Low**: Cosmetic or rare; respond next business day.

## Response steps

1. **Acknowledge**: Assign incident owner, create incident channel or ticket.
2. **Assess**: Determine scope (which service, which tenants), impact (error rate, latency), and likely cause (recent deploy, dependency, load).
3. **Mitigate**: Rollback or feature-flag off if recent change; scale up or restart if resource issue; block bad traffic if abuse.
4. **Communicate**: Update status page and stakeholders at 30min intervals until resolved.
5. **Resolve**: Confirm recovery (metrics, smoke tests), then close incident.
6. **Post-mortem**: Within 48h, write post-mortem (timeline, root cause, action items) and share.

## Common scenarios

- **NeuronDB (Postgres) down**: Check pod/container and logs; restore from backup if data loss; failover to replica if configured.
- **NeuronAgent high latency**: Check LLM provider status and rate limits; check DB connection pool; scale replicas.
- **Auth failures**: Verify JWT secret and OIDC config; check for clock skew; review recent auth config changes.
- **High error rate**: Check logs and traces; rollback last deploy; enable circuit breakers or rate limits if overload.

## Contacts

- Maintain list of on-call and escalation paths in your operations wiki.
